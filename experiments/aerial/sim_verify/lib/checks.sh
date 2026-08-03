# checks.sh — shared environment checks. SOURCE this (don't execute).
# Each chk_* prints one status line and returns 0 (present) / 1 (missing).
# Relies on env: OPENFLY_ROOT, ENV_NAME, AIRSIM_HOST, AIRSIM_PORT.

_ok(){ printf '  [OK]  %s\n' "$1"; }
_no(){ printf '  [--]  %s\n          fix: %s\n' "$1" "$2"; }

chk_os(){
  if grep -qi '22\.04' /etc/os-release 2>/dev/null; then _ok "Ubuntu 22.04"; return 0
  else _no "Ubuntu 22.04 (recommended)" "OpenFly targets Ubuntu 22.04"; return 1; fi
}
chk_nvidia(){
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then _ok "NVIDIA driver"; return 0
  else _no "NVIDIA driver (nvidia-smi)" "install the 4090 driver"; return 1; fi
}
chk_nvcc(){
  if command -v nvcc >/dev/null 2>&1; then _ok "CUDA toolkit (nvcc)"; return 0
  else _no "CUDA toolkit / nvcc" "only needed for flash-attn build; skip if not training"; return 1; fi
}
chk_ros2(){
  if [ -f /opt/ros/humble/setup.bash ]; then _ok "ROS2 Humble"; return 0
  else _no "ROS2 Humble" "install per docs.ros.org Humble Ubuntu debs"; return 1; fi
}
chk_conda(){
  if command -v conda >/dev/null 2>&1; then _ok "conda"; return 0
  else _no "conda / miniconda" "install miniconda"; return 1; fi
}
chk_conda_env(){
  if conda env list 2>/dev/null | grep -qE '(^|/)openfly[[:space:]]'; then _ok "conda env 'openfly'"; return 0
  else _no "conda env 'openfly'" "conda create -n openfly python=3.10 -y"; return 1; fi
}
chk_openfly_clone(){
  if [ -n "${OPENFLY_ROOT:-}" ] && [ -f "$OPENFLY_ROOT/scripts/sim/env_bridge.py" ]; then _ok "OpenFly clone ($OPENFLY_ROOT)"; return 0
  else _no "OpenFly-Platform clone" "git clone https://github.com/SHAILAB-IPEC/OpenFly-Platform.git \$OPENFLY_ROOT"; return 1; fi
}
chk_toolws(){
  if [ -n "${OPENFLY_ROOT:-}" ] && [ -f "$OPENFLY_ROOT/tool_ws/install/setup.bash" ]; then _ok "tool_ws built"; return 0
  else _no "tool_ws (colcon build)" "cd \$OPENFLY_ROOT/tool_ws && colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3"; return 1; fi
}
chk_scene(){
  local e="${ENV_NAME:-env_airsim_16}"
  if [ -n "${OPENFLY_ROOT:-}" ] && [ -d "$OPENFLY_ROOT/envs/airsim/$e" ]; then _ok "scene $e"; return 0
  else _no "AirSim scene $e" "download HF OpenFly_DataGen/airsim -> \$OPENFLY_ROOT/envs/airsim/$e"; return 1; fi
}
chk_apt(){
  local p="$1"
  if dpkg -s "$p" >/dev/null 2>&1; then _ok "apt: $p"; return 0
  else _no "apt: $p" "sudo apt install -y $p"; return 1; fi
}
chk_py(){
  local m="$1"
  if python3 -c "import $m" >/dev/null 2>&1; then _ok "py: $m"; return 0
  else _no "py: $m" "pip install (inside openfly env): $m"; return 1; fi
}
chk_port(){
  local h="${AIRSIM_HOST:-127.0.0.1}" p="${AIRSIM_PORT:-41451}"
  if command -v nc >/dev/null 2>&1 && nc -z -w3 "$h" "$p" >/dev/null 2>&1; then _ok "bridge reachable $h:$p"; return 0
  else _no "bridge $h:$p not reachable" "launch on renderer: python scripts/sim/env_bridge.py --env \$ENV_NAME (wait ~20s)"; return 1; fi
}
