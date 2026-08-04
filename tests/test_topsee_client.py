"""M7b：TopseeClient 对真实 HTTP 的行为。

重点覆盖已核验的平台怪癖（方案 §2.1）与方案 §5.3 的线程/超时纪律。
"""

import time

import pytest

from adapters.topsee_client import (
    PollCache,
    TopseeApiError,
    TopseeAuthError,
    TopseeBusyError,
    TopseeClient,
    TopseeLicenseError,
    TopseeUnreachable,
)
from tests.fixtures.topsee_fake import FakeTopseeServer


@pytest.fixture
def srv():
    with FakeTopseeServer() as s:
        yield s


def _client(srv, **kw):
    kw.setdefault("account", srv.state.account)
    kw.setdefault("password", srv.state.password)
    kw.setdefault("timeout_s", 5.0)
    return TopseeClient(srv.base_url, **kw)


def test_login_encrypts_password_with_platform_key(srv):
    c = _client(srv)
    c.login()
    assert srv.state.logged_in is True
    assert c.login_count == 1


def test_login_wrong_password_rejected(srv):
    c = _client(srv, password="wrong")
    with pytest.raises(TopseeAuthError):
        c.login()


def test_license_expired_is_not_retryable(srv):
    """F5：授权到期是人工问题，必须与可重试的网络错误区分开。"""
    srv.state.license_expired = True
    c = _client(srv)
    with pytest.raises(TopseeLicenseError):
        c.login()


def test_unauthenticated_plaintext_401_triggers_relogin(srv):
    """F2：401 + 纯文本「未登录」不能被当成 JSON 解析，且应自动重登一次。"""
    c = _client(srv)
    data = c.get_state_data()  # 未登录 → 自动 login → 重试
    assert data is None or data == srv.state.state_rows
    assert c.login_count == 1
    assert srv.state.logged_in is True


def test_relogin_disabled_raises(srv):
    c = _client(srv)
    with pytest.raises(TopseeAuthError):
        c.request("GET", "/service/api/robot/state/getStateData", allow_relogin=False)


def test_business_code_nonzero_raises_even_on_http_200(srv):
    """F1：HTTP 200 不代表成功，必须看 body 里的 code。"""
    srv.state.force_error["/robotTask/getCurrentByRobotId"] = (500123, "内部错误")
    c = _client(srv)
    c.login()
    with pytest.raises(TopseeApiError) as ei:
        c.get_current_task("B2000397")
    assert ei.value.code == 500123
    assert "内部错误" in ei.value.message


def test_send_navigate_returns_no_task_id(srv):
    """F8：sendNavigate 只回裸 Result，任何依赖 taskId 的设计都不成立。"""
    c = _client(srv)
    c.login()
    assert c.send_navigate("B2000397", "快速打点-1785465994716") == {}
    assert srv.state.navigate_calls == [("B2000397", "快速打点-1785465994716")]


def test_controller_busy_mapped_to_dedicated_error(srv):
    srv.state.controller_busy = True
    c = _client(srv)
    c.login()
    with pytest.raises(TopseeBusyError):
        c.update_controller_user("B2000397", state="02")


def test_stop_task_uses_short_timeout(srv):
    """方案 §5.3：abort 路径不许被网络拖住。"""
    srv.state.delay_s["/robotTask/stopTask"] = 1.5
    c = _client(srv, abort_timeout_s=0.3)
    c.login()
    t0 = time.monotonic()
    with pytest.raises(TopseeUnreachable):
        c.stop_task("B2000397")
    assert time.monotonic() - t0 < 1.2


def test_unreachable_host_raises_unreachable():
    c = TopseeClient(
        "http://127.0.0.1:1", account="a", password="b", timeout_s=0.5
    )
    with pytest.raises(TopseeUnreachable):
        c.get_state_data()


def test_relocate_serializes_xyth(srv):
    """F15：重定位要带 xyth JSON 字符串。"""
    c = _client(srv)
    c.login()
    assert c.relocate("B2000397", "demo", 1.5, -2.0, 0.3) == "ok"
    assert ("POST", "/service/api/robot/mapMan/mapUpdate") in srv.state.request_log


def test_logout_swallows_errors(srv):
    c = _client(srv)
    c.login()
    srv.state.force_error["/permission/vueLogout"] = (1, "boom")
    c.logout()  # 不抛


# ---------- PollCache ----------


def test_poll_cache_reports_age_and_never_leaks_exceptions():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first fails")
        return calls["n"]

    cache: PollCache[int] = PollCache(flaky, interval_s=0.5, name="t")
    cache.refresh_once()
    val, age = cache.get()
    assert val is None and age == float("inf")
    assert isinstance(cache.last_error, RuntimeError)

    cache.refresh_once()
    val, age = cache.get()
    assert val == 2 and age < 1.0
    assert cache.last_error is None


def test_poll_cache_thread_lifecycle():
    cache: PollCache[int] = PollCache(lambda: 7, interval_s=0.5, name="t")
    cache.start()
    try:
        deadline = time.monotonic() + 3.0
        while cache.fetch_count == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert cache.get()[0] == 7
    finally:
        cache.stop()


def test_poll_cache_rejects_nonpositive_interval():
    with pytest.raises(ValueError):
        PollCache(lambda: 1, interval_s=0.0)
