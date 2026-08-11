#!/usr/bin/env bash
# Aerial WAM — 4090 renderer bridge (RESIDENT). Run this ON the 4090
# (10.229.20.125) in its own terminal and LEAVE IT RUNNING; the H100 ②/④ gate
# connects cross-net to 10.229.20.125:41451 while this holds the AirSim client.
#
# It does NOT live in this repo's code path — the bridge (env_bridge.py) ships
# with OpenFly-Platform. This wrapper just does the fragile, easy-to-forget
# launch sequence (ROS2 source + conda activate + cross-net reminder) so it is
# one command every time instead of four remembered-from-memory lines.
#
# Usage (on the 4090):
#   experiments/aerial/scripts/start_renderer_4090.sh
# Override the 4090-local paths if yours differ:
#   OPENFLY_ROOT=/data/OpenFly-Platform ENV_NAME=env_airsim_16 \
#   CONDA_ENV=openfly ROS_SETUP=/opt/ros/humble/setup.bash \
#   experiments/aerial/scripts/start_renderer_4090.sh
#
# Cross-net prerequisite (one-time, NOT done here): AirSim RPC must bind to
# 0.0.0.0 (settings.json LocalHostIp / OpenFly env_airsim_16.yaml), not
# 127.0.0.1, and 41451/tcp must be allowed — else H100 cannot reach it.
# Single consumer: only ONE client may hold 41451 at a time.
set -uo pipefail

OPENFLY_ROOT="${OPENFLY_ROOT:-/data/OpenFly-Platform}"
ENV_NAME="${ENV_NAME:-env_airsim_16}"
CONDA_ENV="${CONDA_ENV:-openfly}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
BRIDGE_REL="${BRIDGE_REL:-scripts/sim/env_bridge.py}"

say(){ echo "[renderer] $*"; }
die(){ echo "[renderer] ✗ $*" >&2; exit 1; }

[ -d "$OPENFLY_ROOT" ] || die "OPENFLY_ROOT not found: $OPENFLY_ROOT (set OPENFLY_ROOT=...)"
BRIDGE="$OPENFLY_ROOT/$BRIDGE_REL"
[ -f "$BRIDGE" ] || die "bridge not found: $BRIDGE (set BRIDGE_REL=...)"

# ROS2 (OpenFly's env_bridge is a ROS2 node — required every fresh shell).
if [ -f "$ROS_SETUP" ]; then
  say "source $ROS_SETUP"
  # shellcheck disable=SC1090
  source "$ROS_SETUP"
else
  say "WARNING: ROS setup not found at $ROS_SETUP — continuing (set ROS_SETUP=...)"
fi

# conda activate (best-effort; conda's shell hook is not always on PATH in scripts).
if command -v conda >/dev/null 2>&1; then
  say "conda activate $CONDA_ENV"
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV" \
    || say "WARNING: conda activate $CONDA_ENV failed — using current python."
else
  say "WARNING: conda not on PATH — using current python (activate $CONDA_ENV manually if needed)."
fi

say "python: $(command -v python) ($(python --version 2>&1))"
say "launching bridge: $BRIDGE --env $ENV_NAME"
say "wait ~20s for 'ready to be connected'; then run the H100 gate."
say "reachability check from H100:  python3 -c \"import socket;socket.create_connection(('10.229.20.125',41451),5)\""
cd "$OPENFLY_ROOT"
exec python "$BRIDGE" --env "$ENV_NAME"
