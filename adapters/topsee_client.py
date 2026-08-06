"""拓普视巡检平台 HTTP 客户端（标准库实现）。

设计依据：`docs/design/2026-08-03-dog-integration-plan.md`

已核验的平台行为（对应方案 §2.1）：
  F1  成功返回码是 body 里的 `code:0`，不是 HTTP 200
  F2  未登录返回 HTTP 401，body 是纯文本「未登录」而非 JSON
  F3  登录三步：取公钥 → RSA 加密密码 → POST pc/login
      （云端：参数在 query；离线一体机 :8888：JSON body + webToken）
  F4  token header 名与有效期文档未写 → 同时支持 cookie 会话与可配 header
      （离线一体机抓包确认：`tuopushi_edge_token`）
  F5  账号有授权天数，到期无法登录 → 单独抛 TopseeLicenseError，不可重试
  F9  7 个接口已 deprecated，本模块一律不封装

线程模型（方案 §5.3）：所有网络调用同步阻塞，**禁止**在 MissionBrain 的
tick 循环里直接调用。需要周期状态的调用方用 PollCache 起独立线程，
tick 只读缓存。
"""

from __future__ import annotations

import json
import logging
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Callable, Dict, Generic, Mapping, Optional, Tuple, TypeVar

from adapters.topsee_rsa import encrypt_b64

logger = logging.getLogger(__name__)

# 平台把「未登录」以纯文本 + 401 返回，这些片段用于识别
_UNAUTH_HINTS = ("未登录", "not login", "unauthorized")
# 授权天数到期属于人工问题，重试无用
_LICENSE_HINTS = ("授权", "到期", "过期", "license", "expired")


class TopseeError(Exception):
    """平台交互错误基类。"""


class TopseeUnreachable(TopseeError):
    """网络层失败（连不上 / 超时 / DNS）。可重试。"""


class TopseeAuthError(TopseeError):
    """未登录或登录失败。可尝试重新登录一次。"""


class TopseeLicenseError(TopseeAuthError):
    """账号授权到期（F5）。**不可重试**，须人工联系厂商续期。"""


class TopseeBusyError(TopseeError):
    """机器人控制权被他人占用。"""


class TopseeApiError(TopseeError):
    """HTTP 通了但业务 code != 0。"""

    def __init__(self, code: Any, message: str, path: str) -> None:
        super().__init__(f"{path} 返回 code={code!r}: {message}")
        self.code = code
        self.message = message
        self.path = path


T = TypeVar("T")


class PollCache(Generic[T]):
    """后台线程周期刷新的单值缓存（方案 §5.3）。

    tick 循环只读 `get()`，绝不触发网络。`get()` 返回 (值, 年龄秒)；
    从未成功过则值为 None。超过 stale_s 的判定由调用方做，
    因为不同用途容忍度不同。
    """

    def __init__(
        self,
        fetch: Callable[[], T],
        *,
        interval_s: float = 1.0,
        name: str = "poll",
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s 必须为正")
        self._fetch = fetch
        self.interval_s = float(interval_s)
        self.name = name
        self._lock = threading.Lock()
        self._value: Optional[T] = None
        self._at: float = 0.0
        self._error: Optional[BaseException] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.fetch_count = 0
        self.error_count = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name=f"topsee-{self.name}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        th = self._thread
        self._thread = None
        if th is not None:
            th.join(timeout=timeout_s)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.refresh_once()
            self._stop.wait(self.interval_s)

    def refresh_once(self) -> None:
        """同步刷新一次。测试里可以不起线程直接调这个。"""
        try:
            val = self._fetch()
        except BaseException as exc:  # noqa: BLE001 — 后台线程不能让异常逃逸
            with self._lock:
                self._error = exc
                self.error_count += 1
            logger.debug("PollCache(%s) 刷新失败: %s", self.name, exc)
            return
        with self._lock:
            self._value = val
            self._at = time.monotonic()
            self._error = None
            self.fetch_count += 1

    def get(self, now: Optional[float] = None) -> Tuple[Optional[T], float]:
        """返回 (值, 年龄秒)。从未成功过时年龄为 inf。"""
        t = float(now if now is not None else time.monotonic())
        with self._lock:
            if self._at == 0.0:
                return None, float("inf")
            return self._value, max(0.0, t - self._at)

    @property
    def last_error(self) -> Optional[BaseException]:
        with self._lock:
            return self._error


class TopseeClient:
    """平台 HTTP 会话。

    只封装方案 §3 判定为「必用 / 可用但需包装」的接口；deprecated 接口
    （`selectControllerUser` / `testPoints` / `getRtspUrl` 等）一律不提供，
    以免有人顺手用上。
    """

    ROBOT = "/service/api/robot"
    PERM = "/service/api/permission"
    # 离线一体机（topsee-offline-app :8888）前缀与云端不同
    _OFFLINE_PREFIX = "/service"
    _OFFLINE_TOKEN_HEADER = "tuopushi_edge_token"

    def __init__(
        self,
        base_url: str,
        *,
        account: str,
        password: str,
        timeout_s: float = 10.0,
        abort_timeout_s: float = 2.0,
        insecure_tls: bool = False,
        current_language: str = "zh_CN",
        token_header: Optional[str] = None,
        deployment: str = "cloud",
        opener: Optional[Any] = None,
    ) -> None:
        """
        base_url: 形如 `http://192.168.0.85:8001`（云端）或 `:8888`（离线一体机）。
        deployment: `cloud`（默认）或 `offline`（:8888 一体机 API 形态）。
        abort_timeout_s: 取消/停止类调用的短超时，防止网络卡顿拖慢 abort 路径。
        insecure_tls: 现场 https 常为自签证书；仅在明确知情时打开。
        token_header: 抓包确认 token header 名后填入（G8）；未知时靠 cookie 会话。
        opener: 注入自定义 urllib opener，供测试替换。
        """
        mode = (deployment or "cloud").strip().lower()
        if mode not in ("cloud", "offline"):
            raise ValueError(f"未知 deployment={deployment!r}（cloud|offline）")
        self.deployment = mode
        self.base_url = base_url.rstrip("/")
        self.account = account
        self._password = password
        self.timeout_s = float(timeout_s)
        self.abort_timeout_s = float(abort_timeout_s)
        self.current_language = current_language
        if token_header:
            self.token_header = token_header
        elif mode == "offline":
            self.token_header = self._OFFLINE_TOKEN_HEADER
        else:
            self.token_header = None
        self._token: Optional[str] = None
        self._lock = threading.Lock()
        self.login_count = 0
        self.request_count = 0

        if mode == "offline":
            self.ROBOT = self._OFFLINE_PREFIX
            self.PERM = self._OFFLINE_PREFIX

        if opener is not None:
            self._opener = opener
        else:
            handlers: list[Any] = [urllib.request.HTTPCookieProcessor()]
            if insecure_tls:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                handlers.append(urllib.request.HTTPSHandler(context=ctx))
            self._opener = urllib.request.build_opener(*handlers)

    # ---------- 底层 ----------

    def _raw(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json_body: Optional[Mapping[str, Any]] = None,
        timeout_s: Optional[float] = None,
    ) -> Tuple[int, str]:
        """发一次请求，返回 (http_status, body_text)。不解析业务 code。"""
        url = self.base_url + path
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            if clean:
                url = f"{url}?{urllib.parse.urlencode(clean)}"
        data = None
        headers = {"Accept": "application/json, */*"}
        if json_body is not None:
            data = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json;charset=UTF-8"
        if self.token_header and self._token:
            headers[self.token_header] = self._token

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        self.request_count += 1
        try:
            with self._opener.open(
                req, timeout=float(timeout_s if timeout_s is not None else self.timeout_s)
            ) as resp:
                return int(resp.status), resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", "replace")
            except Exception:  # noqa: BLE001 — body 可选
                pass
            return int(exc.code), body
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise TopseeUnreachable(f"{method} {path} 网络失败: {exc}") from exc

    @staticmethod
    def _classify_unauth(status: int, body: str) -> Optional[TopseeAuthError]:
        """把 401 / 未登录文本分类成 License 或普通 Auth 错误。"""
        low = body.lower()
        looks_unauth = status in (401, 403) or any(h in body or h in low for h in _UNAUTH_HINTS)
        if not looks_unauth:
            return None
        if any(h in body or h in low for h in _LICENSE_HINTS):
            return TopseeLicenseError(f"授权异常（不可重试）: HTTP {status} {body[:120]}")
        return TopseeAuthError(f"未登录: HTTP {status} {body[:120]}")

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json_body: Optional[Mapping[str, Any]] = None,
        timeout_s: Optional[float] = None,
        allow_relogin: bool = True,
    ) -> Any:
        """发请求并解包业务 `data`。

        code != 0 抛 TopseeApiError；未登录先自动重登一次再重试。
        授权到期直接抛 TopseeLicenseError，不重试（F5）。
        """
        status, body = self._raw(
            method, path, params=params, json_body=json_body, timeout_s=timeout_s
        )
        auth_exc = self._classify_unauth(status, body)
        if auth_exc is not None:
            if isinstance(auth_exc, TopseeLicenseError) or not allow_relogin:
                raise auth_exc
            logger.info("检测到未登录，自动重新登录后重试一次: %s", path)
            self.login()
            status, body = self._raw(
                method, path, params=params, json_body=json_body, timeout_s=timeout_s
            )
            again = self._classify_unauth(status, body)
            if again is not None:
                raise again

        if status >= 500:
            raise TopseeApiError(status, body[:200], path)

        try:
            doc = json.loads(body) if body.strip() else {}
        except json.JSONDecodeError as exc:
            raise TopseeApiError(status, f"响应非 JSON: {body[:120]}", path) from exc
        if not isinstance(doc, Mapping):
            raise TopseeApiError(status, f"响应不是对象: {body[:120]}", path)

        code = doc.get("code")
        # F1：成功是 code:0；平台有时给字符串 "0"
        if code is not None and str(code) != "0":
            raise TopseeApiError(code, str(doc.get("message", "")), path)
        return doc.get("data")

    # ---------- 登录 ----------

    def login(self) -> None:
        """RSA 登录（F3）。失败抛 TopseeAuthError / TopseeLicenseError。"""
        with self._lock:
            security_key = str(uuid.uuid4())
            status, body = self._raw(
                "GET",
                f"{self.PERM}/free/security/rsa",
                params={"securityKey": security_key},
            )
            if status != 200:
                raise TopseeAuthError(f"取公钥失败: HTTP {status} {body[:120]}")
            try:
                doc = json.loads(body)
            except json.JSONDecodeError as exc:
                raise TopseeAuthError(f"公钥响应非 JSON: {body[:120]}") from exc
            if str(doc.get("code")) != "0" or not doc.get("data"):
                raise TopseeAuthError(f"取公钥失败: {body[:160]}")
            enc = encrypt_b64(self._password, str(doc["data"]))

            login_payload = {
                "account": self.account,
                "password": enc,
                "securityKey": security_key,
                "currentLanguage": self.current_language,
            }
            if self.deployment == "offline":
                # 离线一体机：POST JSON body；query/form 会 400/415
                status, body = self._raw(
                    "POST",
                    f"{self.PERM}/free/pc/login",
                    json_body=login_payload,
                )
            else:
                status, body = self._raw(
                    "POST",
                    f"{self.PERM}/free/pc/login",
                    params=login_payload,
                )
            lic = self._classify_unauth(status, body)
            if isinstance(lic, TopseeLicenseError):
                raise lic
            if status != 200:
                raise TopseeAuthError(f"登录失败: HTTP {status} {body[:160]}")
            try:
                doc = json.loads(body) if body.strip() else {}
            except json.JSONDecodeError as exc:
                raise TopseeAuthError(f"登录响应非 JSON: {body[:120]}") from exc
            if str(doc.get("code")) != "0":
                msg = str(doc.get("message", ""))
                if any(h in msg for h in _LICENSE_HINTS):
                    raise TopseeLicenseError(f"授权到期（不可重试）: {msg}")
                raise TopseeAuthError(f"登录失败: {body[:160]}")
            data = doc.get("data")
            if isinstance(data, Mapping):
                for k in ("webToken", "token", "accessToken", "access_token"):
                    if data.get(k):
                        self._token = str(data[k])
                        break
            self.login_count += 1
            logger.info(
                "topsee 登录成功 account=%s deployment=%s",
                self.account,
                self.deployment,
            )

    def logout(self) -> None:
        """退出登录。失败只记日志，不抛（收尾路径）。"""
        try:
            self.request("GET", f"{self.PERM}/vueLogout", allow_relogin=False)
        except TopseeError as exc:
            logger.debug("logout 忽略错误: %s", exc)
        self._token = None

    # ---------- 状态与控制权 ----------

    def get_state_data(
        self,
        *,
        robot_name: Optional[str] = None,
        robot_state: Optional[str] = None,
        robot_type: Optional[str] = None,
    ) -> Any:
        """GET state/getStateData — 实时状态列表（电量/温度等）。"""
        return self.request(
            "GET",
            f"{self.ROBOT}/state/getStateData",
            params={
                "robotName": robot_name,
                "robotState": robot_state,
                "robotType": robot_type,
            },
        )

    def update_controller_user(
        self,
        robot_id: str,
        *,
        state: Optional[str] = None,
        force: Optional[str] = None,
    ) -> Any:
        """PUT state/updateControllerUser — 修改控制人。

        `state` / `force` 的取值 OpenAPI 未给枚举（G7），调用方只应传抓包确认过的值。
        控制权被占用时平台如何报错也未知，这里把含「忙」「占用」的业务错误
        归一成 TopseeBusyError，便于上层区分。
        """
        try:
            return self.request(
                "PUT",
                f"{self.ROBOT}/state/updateControllerUser",
                params={"robotId": robot_id, "state": state, "force": force},
            )
        except TopseeApiError as exc:
            if any(h in exc.message for h in ("忙", "占用", "busy", "已被")):
                raise TopseeBusyError(f"控制权被占用: {exc.message}") from exc
            raise

    # ---------- 导航与任务 ----------

    def send_navigate(self, robot_id: str, points_id: str) -> Any:
        """POST point/sendNavigate — 单点拓扑派单。

        F8：返回裸 `Result`，**不含任何任务标识**。是否会产生可查询任务
        属于待验证假设 E1，调用方不得依赖。
        """
        return self.request(
            "POST",
            f"{self.ROBOT}/point/sendNavigate",
            params={"robotId": robot_id, "pointsId": points_id},
        )

    def get_current_task(self, robot_id: str) -> Any:
        """GET instrument/robotTask/getCurrentByRobotId → ShowRobotTaskAllEntity（F6）。

        字段是 `pointsId` / `currentState` / `totalState` / `taskId` / `taskItemId`，
        **没有** `currentPointsId` 与 `inspectionRate`（那两个在 getPagingRobotTask）。
        """
        return self.request(
            "GET",
            f"{self.ROBOT}/instrument/robotTask/getCurrentByRobotId",
            params={"robotId": robot_id},
        )

    def stop_task(self, robot_id: str) -> Any:
        """GET instrument/robotTask/stopTask — 任务级 abort，走短超时。"""
        return self.request(
            "GET",
            f"{self.ROBOT}/instrument/robotTask/stopTask",
            params={"robotId": robot_id},
            timeout_s=self.abort_timeout_s,
        )

    def pause_task(self, robot_id: str) -> Any:
        """GET instrument/robotTask/pauseTask。"""
        return self.request(
            "GET",
            f"{self.ROBOT}/instrument/robotTask/pauseTask",
            params={"robotId": robot_id},
            timeout_s=self.abort_timeout_s,
        )

    def back_to_charge(self, robot_id: str) -> Any:
        """GET instrument/robotTask/back — 一键返桩。依赖反光柱标定（F31）。"""
        return self.request(
            "GET",
            f"{self.ROBOT}/instrument/robotTask/back",
            params={"robotId": robot_id},
        )

    # ---------- 地图与点位 ----------

    def map_update(self, body: Mapping[str, Any]) -> Any:
        """POST 地图更新 — MapUpdateDomain（F15）。

        云端：`mapMan/mapUpdate`；离线一体机：`free/mapUpdate`。
        action: 0 初始化 / 1 重定位 / 2 增量建图 / 3 结束 / 4 取消 / 5 切割 / 6 加载地图
        重定位需带 `xyth`（形如 `{"x":0,"y":0,"th":0}` 的 JSON 字符串）。
        注意：**重定位是否成功无法程序化读取**（G4 无置信度接口），仍需人工确认。
        """
        path = (
            f"{self.ROBOT}/free/mapUpdate"
            if self.deployment == "offline"
            else f"{self.ROBOT}/mapMan/mapUpdate"
        )
        return self.request("POST", path, json_body=body)

    def relocate(self, robot_id: str, keyname: str, x: float, y: float, th: float) -> Any:
        """mapUpdate 的重定位便捷封装（action=1）。"""
        return self.map_update(
            {
                "action": 1,
                "robotId": robot_id,
                "destKeyname": keyname,
                "xyth": json.dumps({"x": x, "y": y, "th": th}),
            }
        )

    def load_map(self, robot_id: str, keyname: str) -> Any:
        """mapUpdate 的加载地图便捷封装（action=6）。"""
        return self.map_update(
            {"action": 6, "robotId": robot_id, "destKeyname": keyname}
        )

    def get_robot_map_all(self, robot_id: str) -> Any:
        """GET 地图 + 线路 + 点位导出。

        云端：`mapMan/getRobotMapAll`；离线一体机有数据的是 `point/getRobotMapAll`
        （`mapMan` 路径会回空点位）。
        G16：OpenAPI 把它的响应也标成 ShowRobotTaskAllEntity，疑似导出标注错误，
        解析方要对结构做防御（见 tools/export_dog_bindings.py）。
        """
        path = (
            f"{self.ROBOT}/point/getRobotMapAll"
            if self.deployment == "offline"
            else f"{self.ROBOT}/mapMan/getRobotMapAll"
        )
        return self.request("GET", path, params={"robotId": robot_id})

    def get_points_by_id(self, points_id: str) -> Any:
        """GET point/getPointsById — 点位详情（含 x/y/th）。"""
        return self.request(
            "GET", f"{self.ROBOT}/point/getPointsById", params={"pointsId": points_id}
        )

    # ---------- 视频与证据 ----------

    def get_stream_url(
        self,
        *,
        device_id: str,
        ip: str,
        stream: str = "main",
        stream_mode: str = "tcp",
        screenshot: str = "0",
    ) -> Any:
        """GET video/getStreamUrl — 取流地址。

        五个参数在 OpenAPI 里都是必填。响应是未结构化 JSONObject。
        不要用已 deprecated 的 archivesMan/getRtspUrl（F9）。
        """
        return self.request(
            "GET",
            f"{self.ROBOT}/video/getStreamUrl",
            params={
                "deviceId": device_id,
                "ip": ip,
                "stream": stream,
                "streamMode": stream_mode,
                "screenshot": screenshot,
            },
        )

    def get_alarm_list(self, **params: Any) -> Any:
        """GET 只读告警，作旁路证据，不驱动状态机。

        云端：`taskAlarm/getAlarmList`；离线一体机有数据的是 `getAlarmHistory`。
        """
        name = "getAlarmHistory" if self.deployment == "offline" else "getAlarmList"
        return self.request("GET", f"{self.ROBOT}/taskAlarm/{name}", params=params)

    # ---------- 气体 ----------

    def get_gas_history(self, **params: Any) -> Any:
        """GET gas/getGasHistory — 只有历史聚合，没有「立即采样」（G9）。"""
        return self.request(
            "GET", f"{self.ROBOT}/gas/getGasHistory", params=params
        )
