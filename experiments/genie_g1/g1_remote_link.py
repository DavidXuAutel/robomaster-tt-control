"""
本地 FastWAM 与智元 G1 机器人（默认 IP ``10.229.66.60``）之间的 ROS 2 连接配置。

G1 GDK 侧话题仍为同一 DDS 域内的默认名称（如 ``/camera/head_color``）；本机与机器人
需 **同一 ``ROS_DOMAIN_ID``**，且跨网段 / 无组播时常用 **静态 Peer** 发现。

本模块在 ``rclpy.init`` 之前写入配置文件并设置环境变量；**两侧 RMW 需一致**
（例如均为 ``rmw_fastrtps_cpp`` 或均为 ``rmw_cyclonedds_cpp``），否则无法互通。
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Literal, Optional

DEFAULT_G1_ROBOT_IP = "10.229.66.60"

DdsBackend = Literal["cyclonedds", "fastrtps"]


def _config_dir() -> Path:
    d = Path.home() / ".config" / "fastwam" / "g1_remote"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_cyclonedds_peers_xml(robot_ip: str, *, domain_id: Optional[int] = None) -> Path:
    """生成 Cyclone DDS 静态 peer 配置（关闭组播，仅指向机器人）。"""
    domain_open = (
        f' id="{int(domain_id)}"' if domain_id is not None else ""
    )
    body = textwrap.dedent(
        f"""\
        <?xml version="1.0" encoding="UTF-8" ?>
        <CycloneDDS xmlns="https://cyclic.ossi.org/cyclonedds">
          <Domain{domain_open}>
            <General>
              <AllowMulticast>false</AllowMulticast>
            </General>
            <Discovery>
              <ParticipantIndex>auto</ParticipantIndex>
              <Peers>
                <Peer address="{robot_ip.strip()}"/>
              </Peers>
            </Discovery>
          </Domain>
        </CycloneDDS>
        """
    )
    path = _config_dir() / f"cyclonedds_peer_{robot_ip.replace('.', '_')}.xml"
    path.write_text(body.strip() + "\n", encoding="utf-8")
    return path.resolve()


def write_fastdds_peers_xml(robot_ip: str, *, domain_id: int = 0) -> Path:
    """
    生成 Fast DDS 初始对等方列表（指向机器人一侧常见发现端口）。

    端口按 ROS 2 常用约定：``7400 + 250 * domain_id`` 起；多写几个 locator 提高发现概率。
    若仍无法发现，请在机器人上 ``ros2 daemon stop`` 后用 ``ros2 topic list`` 对照端口。
    """
    ip = robot_ip.strip()
    d = int(domain_id)
    base = 7400 + 250 * d
    ports = [base, base + 1, base + 2, 7410 + 250 * d, 7411 + 250 * d]
    locator_lines = []
    for p in ports:
        locator_lines.append(f"""                <locator>
                  <udpv4>
                    <address>{ip}</address>
                    <port>{p}</port>
                  </udpv4>
                </locator>""")
    locators = "\n".join(locator_lines)
    body = f"""<?xml version="1.0" encoding="UTF-8" ?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
  <participant profile_name="fastwam_g1_client" is_default_profile="true">
    <rtps>
      <builtin>
        <discovery_config>
          <discoveryProtocol>SIMPLE</discoveryProtocol>
        </discovery_config>
        <initialPeersList>
{locators}
        </initialPeersList>
      </builtin>
    </rtps>
  </participant>
</profiles>
"""
    path = _config_dir() / f"fastdds_peers_{ip.replace('.', '_')}_d{d}.xml"
    path.write_text(body, encoding="utf-8")
    return path.resolve()


def remote_ros_env_dict(
    robot_ip: str,
    *,
    domain_id: Optional[int] = None,
    dds: DdsBackend = "fastrtps",
) -> dict[str, str]:
    """
    计算连接 G1 所需的环境变量（不写 os.environ），便于 ``eval`` 或测试。
    """
    ip = robot_ip.strip()
    if not ip:
        raise ValueError("robot_ip must be non-empty.")

    rid = int(domain_id) if domain_id is not None else int(os.environ.get("ROS_DOMAIN_ID", "0"))
    out: dict[str, str] = {
        "G1_ROBOT_IP": ip,
        "ROS_DOMAIN_ID": str(rid),
        "ROS_LOCALHOST_ONLY": "0",
    }

    if dds == "cyclonedds":
        xml = write_cyclonedds_peers_xml(ip, domain_id=domain_id)
        out["CYCLONEDDS_URI"] = f"file://{xml}"
        out["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
    else:
        xml = write_fastdds_peers_xml(ip, domain_id=rid)
        out["FASTRTPS_DEFAULT_PROFILES_FILE"] = str(xml)
        out["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"

    return out


def apply_remote_ros_env(
    robot_ip: str,
    *,
    domain_id: Optional[int] = None,
    dds: DdsBackend = "fastrtps",
) -> dict[str, str]:
    """
    为连接远程 G1 设置进程环境变量。应在 ``import rclpy`` / ``rclpy.init`` 之前调用。

    Returns
    -------
    dict
        已设置的关键变量摘要（便于打印日志）。
    """
    env = remote_ros_env_dict(robot_ip, domain_id=domain_id, dds=dds)
    os.environ.pop("ROS_LOCALHOST_ONLY", None)
    for k, v in env.items():
        os.environ[k] = v
    if dds == "cyclonedds":
        os.environ.pop("FASTRTPS_DEFAULT_PROFILES_FILE", None)
    else:
        os.environ.pop("CYCLONEDDS_URI", None)
    return env


def summarize_connection(robot_ip: str, applied: dict[str, str]) -> str:
    return (
        f"G1 remote ROS2 link | robot_ip={robot_ip!r} | "
        + " ".join(f"{k}={v!r}" for k, v in sorted(applied.items()))
    )


def _cli() -> None:
    import argparse
    import shlex

    p = argparse.ArgumentParser(description="Print or apply G1 remote ROS 2 DDS env.")
    p.add_argument("--g1-ip", default=DEFAULT_G1_ROBOT_IP)
    p.add_argument("--ros-domain-id", type=int, default=None)
    p.add_argument("--dds", choices=("fastrtps", "cyclonedds"), default="fastrtps")
    p.add_argument(
        "--print-shell-exports",
        action="store_true",
        help="Print `export VAR=...` lines for bash/zsh eval (does not modify parent shell unless eval'd).",
    )
    args = p.parse_args()
    env = remote_ros_env_dict(args.g1_ip, domain_id=args.ros_domain_id, dds=args.dds)
    if args.print_shell_exports:
        for k, v in env.items():
            print(f"export {k}={shlex.quote(v)}")
        if args.dds == "cyclonedds":
            print("unset FASTRTPS_DEFAULT_PROFILES_FILE 2>/dev/null || true")
        else:
            print("unset CYCLONEDDS_URI 2>/dev/null || true")
        return
    apply_remote_ros_env(args.g1_ip, domain_id=args.ros_domain_id, dds=args.dds)
    print(summarize_connection(args.g1_ip, env))


if __name__ == "__main__":
    _cli()
