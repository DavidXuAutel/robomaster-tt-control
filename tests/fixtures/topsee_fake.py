"""可编排的假拓普视平台，用于无真机回归测试。

刻意复刻已核验的平台怪癖（方案 §2.1），否则测试会给出虚假的安全感：
  F1  成功是 body 里的 `code:0`，不是 HTTP 200
  F2  未登录返回 HTTP 401 + **纯文本**「未登录」，不是 JSON
  F3  登录要先取公钥再用 RSA 加密密码（这里会真的解密校验）
  F8  sendNavigate 返回裸 Result，不含任何任务标识

RSA 密钥是一次性生成的 1024 位测试密钥，只用于单测，不用于任何真实环境。
"""

from __future__ import annotations

import base64
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

# 测试专用密钥（tmp 脚本一次性生成，已删脚本）
TEST_SPKI_B64 = (
    "MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDWFzEDxrvm9uKxppMj8CbKK5gMOYruJPaH97yk"
    "BuyqwNuniqrLlzyfU8b/oaKtr/CHgJVnrTZtFsxKdcCv7HXv/oINo3ul5+GDvwqvkWi+g5cPY1dp"
    "xLaOFflqaM8gmdJcwOR5zRzl8aVdkMtjdDNZWEiKsqwlVZU5KYAyFT2NFwIDAQAB"
)
TEST_PKCS1_B64 = (
    "MIGJAoGBANYXMQPGu+b24rGmkyPwJsormAw5iu4k9of3vKQG7KrA26eKqsuXPJ9Txv+hoq2v8IeA"
    "lWetNm0WzEp1wK/sde/+gg2je6Xn4YO/Cq+RaL6Dlw9jV2nEto4V+WpozyCZ0lzA5HnNHOXxpV2Q"
    "y2N0M1lYSIqyrCVVlTkpgDIVPY0XAgMBAAE="
)
TEST_N = int(
    "150339526116465640456523194014022106749514136396235161244531965476978291700"
    "225156476494159286388601970139960794501916814339189978981149867426397667355"
    "041743280433712610928040084331004396497897675618771034831825361234308957496"
    "781881234368557310636599334868789090667005735720615514425535579495138655802"
    "666880279"
)
TEST_E = 65537
TEST_D = int(
    "146439787123255826324413249557854787807936492746486167941272671622958988163"
    "438566214351551739707175854369664117042569291804184974719924074750125087979"
    "977695007555450367832589947685231906854125470172328976310129794889108344074"
    "382138997734854683505922043922105965702488968714756130243186284651011582275"
    "758045857"
)


def rsa_decrypt(cipher_b64: str) -> str:
    """用测试私钥解 PKCS#1 v1.5，校验我们自己的加密实现是否正确。"""
    raw = base64.b64decode(cipher_b64)
    m = pow(int.from_bytes(raw, "big"), TEST_D, TEST_N)
    k = (TEST_N.bit_length() + 7) // 8
    em = m.to_bytes(k, "big")
    if em[0] != 0x00 or em[1] != 0x02:
        raise ValueError("PKCS#1 头不对")
    sep = em.find(b"\x00", 2)
    if sep < 10:  # PS 至少 8 字节
        raise ValueError("PKCS#1 padding 过短")
    return em[sep + 1 :].decode("utf-8")


class FakeTopseeState:
    """假平台的可编排状态。测试直接改字段来构造场景。"""

    def __init__(self) -> None:
        self.account = "tester"
        self.password = "pa55w0rd!"
        self.public_key_b64 = TEST_SPKI_B64
        self.logged_in = False
        self.require_login = True
        # 授权到期场景：登录直接失败且不可重试
        self.license_expired = False
        # 当前任务；None 表示平台查不到任务（E1 的失败特征）
        self.current_task: Optional[Dict[str, Any]] = None
        self.navigate_calls: List[Tuple[str, str]] = []
        self.stop_calls: List[str] = []
        self.controller_calls: List[Dict[str, Any]] = []
        self.controller_busy = False
        self.gas_rows: List[Dict[str, Any]] = []
        self.map_all: Any = None
        self.alarm_rows: List[Dict[str, Any]] = []
        self.state_rows: Any = None
        # 强制某路径返回业务错误码：{path_suffix: (code, message)}
        self.force_error: Dict[str, Tuple[Any, str]] = {}
        # 让某路径挂起若干秒，用于超时测试
        self.delay_s: Dict[str, float] = {}
        self.request_log: List[Tuple[str, str]] = []


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def st(self) -> FakeTopseeState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, *args: Any) -> None:  # 静音测试输出
        pass

    # ---------- 工具 ----------

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, doc: Any, status: int = 200) -> None:
        self._send(
            status,
            json.dumps(doc, ensure_ascii=False).encode("utf-8"),
            "application/json;charset=UTF-8",
        )

    def _ok(self, data: Any = None) -> None:
        self._json({"code": 0, "message": "成功", "data": data})

    def _unauth(self) -> None:
        # F2：纯文本 + 401，专门用来验证客户端不会当成 JSON 解析
        self._send(401, "未登录".encode("utf-8"), "text/plain;charset=UTF-8")

    # ---------- 路由 ----------

    def do_GET(self) -> None:
        self._route("GET")

    def do_POST(self) -> None:
        self._route("POST")

    def do_PUT(self) -> None:
        self._route("PUT")

    def _route(self, method: str) -> None:
        import time as _time

        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        q = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        st = self.st
        st.request_log.append((method, path))

        for suffix, delay in st.delay_s.items():
            if path.endswith(suffix):
                _time.sleep(delay)

        for suffix, (code, msg) in st.force_error.items():
            if path.endswith(suffix):
                self._json({"code": code, "message": msg, "data": None})
                return

        if path.endswith("/permission/free/security/rsa"):
            self._ok(st.public_key_b64)
            return

        if path.endswith("/permission/free/pc/login"):
            if st.license_expired:
                self._json({"code": 1, "message": "账号授权已到期，请联系厂商", "data": None})
                return
            try:
                pwd = rsa_decrypt(q.get("password", ""))
            except Exception as exc:  # noqa: BLE001
                self._json({"code": 1, "message": f"密码解密失败: {exc}", "data": None})
                return
            if q.get("account") != st.account or pwd != st.password:
                self._json({"code": 1, "message": "账号或密码错误", "data": None})
                return
            st.logged_in = True
            self.send_response(200)
            self.send_header("Set-Cookie", "JSESSIONID=faketoken; Path=/")
            body = json.dumps({"code": 0, "message": "成功", "data": {}}).encode()
            self.send_header("Content-Type", "application/json;charset=UTF-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path.endswith("/permission/vueLogout"):
            st.logged_in = False
            self._ok()
            return

        # 其余接口都需要会话
        if st.require_login and not st.logged_in:
            self._unauth()
            return

        if path.endswith("/robot/state/getStateData"):
            self._ok(st.state_rows)
            return

        if path.endswith("/robot/state/updateControllerUser"):
            st.controller_calls.append(dict(q))
            if st.controller_busy:
                self._json({"code": 1, "message": "机器人忙碌，控制权已被占用", "data": None})
                return
            self._ok()
            return

        if path.endswith("/robot/point/sendNavigate"):
            st.navigate_calls.append((q.get("robotId", ""), q.get("pointsId", "")))
            self._ok({})  # F8：裸 Result，无 taskId
            return

        if path.endswith("/robotTask/getCurrentByRobotId"):
            self._ok(st.current_task)
            return

        if path.endswith("/robotTask/stopTask"):
            st.stop_calls.append(q.get("robotId", ""))
            st.current_task = None
            self._ok()
            return

        if path.endswith("/robotTask/pauseTask") or path.endswith("/robotTask/back"):
            self._ok()
            return

        if path.endswith("/robot/mapMan/mapUpdate"):
            self._ok("ok")
            return

        if path.endswith("/robot/mapMan/getRobotMapAll"):
            self._ok(st.map_all)
            return

        if path.endswith("/robot/gas/getGasHistory"):
            self._ok(st.gas_rows)
            return

        if path.endswith("/robot/taskAlarm/getAlarmList"):
            self._ok({"records": st.alarm_rows})
            return

        self._send(404, b"not found", "text/plain")


class FakeTopseeServer:
    """上下文管理器：`with FakeTopseeServer() as srv: srv.base_url / srv.state`。"""

    def __init__(self) -> None:
        self.state = FakeTopseeState()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "FakeTopseeServer":
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.state = self.state  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    @property
    def base_url(self) -> str:
        assert self._httpd is not None
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    def __enter__(self) -> "FakeTopseeServer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()
