"""L1 契约：深度后端不得静默回落公网默认地址。"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import main as main_mod


def test_depth_inference_without_service_returns_2():
    args = main_mod.parse_args(
        ["--inference", "depth-anything", "--sim"]
    )
    assert args.depth_service == ""
    with patch.object(main_mod, "App") as App:
        code = main_mod.main(
            ["--inference", "depth-anything", "--sim"]
        )
    assert code == 2
    App.assert_not_called()


def test_start_depth_service_timeout_returns_2_no_fallback():
    """--start-depth-service 失败不得继续 create_backend。"""
    fake_proc = MagicMock()
    fake_proc.pid = 4242
    fake_proc.poll.return_value = 1  # 立刻失败退出
    fake_proc.returncode = 1
    fake_proc.stderr = MagicMock()
    fake_proc.stderr.readline.return_value = b""

    with patch("socket.socket") as sock_cls, patch(
        "subprocess.Popen", return_value=fake_proc
    ), patch.object(main_mod, "App") as App, patch(
        "pathlib.Path.exists", return_value=True
    ):
        # 端口空闲 → 走启动分支
        sock = MagicMock()
        sock.connect_ex.return_value = 1
        sock_cls.return_value = sock
        code = main_mod.main(
            [
                "--inference",
                "depth-anything",
                "--start-depth-service",
                "--sim",
            ]
        )
    assert code == 2
    App.assert_not_called()
