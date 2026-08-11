# Aerial WAM —— 同步代码 & 启动环境 手册

> **下次同步代码 / 在新盒子建环境前,先看这份。** 三台机、四个脚本、六个坑,都在这。
> 脚本都在 `experiments/aerial/scripts/`,自定位(`git rev-parse` 找 repo 根),
> 同一份在 Mac/H100/4090 都能跑。

## 三个工作区

| 机器 | 地址 / 路径 | 角色 |
|---|---|---|
| **Mac**(本机) | worktree `/Users/xudazhong/Projects/robomaster-tt-control/.claude/worktrees/aerial-rl-skeleton`,分支 `aerial-rl-skeleton` | 写代码。沙盒连不上任何 remote → 推送在联网终端手动跑 |
| **8×H100** | `a25689@10.239.121.22 -p 31126` | 训练 / 离线 gate(①③)/ ②④ rollout 客户端(跨网连 4090)。**唯一需要 pull 代码的机器** |
| **4090** | `10.229.20.125`(AirSim RPC `41451`) | AirSim/Unreal 渲染器 + bare `~/repos/...git`。只跑渲染,不 pull 代码 |

**remote 名字每台机不同**(脚本按「哪个存在就用哪个」处理,无需记):
- Mac:`origin` = 4090 bare repo(`yao@10.229.20.125:~/repos/...git`)、`github` = GitHub mirror。两个都推。
- H100:**只有 `origin` = github**。走 github pull。

## 四个脚本

### 1. Mac 推代码 —— `sync_push.sh`
```bash
bash experiments/aerial/scripts/sync_push.sh "提交说明"   # add -A + commit + 推所有 remote
bash experiments/aerial/scripts/sync_push.sh              # 只推已提交的
```
自己 `cd` 到 repo 根,不会再出现「在 ~/Projects 推错仓库」的 refspec 报错。

### 2. H100 拉代码 —— `sync_pull.sh`
```bash
bash experiments/aerial/scripts/sync_pull.sh              # fetch --all + 硬对齐到 origin/aerial-rl-skeleton
```
脏工作区会拒绝(除非 `FORCE=1`)。**首次**(脚本还没到这台机)用自举命令(见下方坑 ②)。

### 3. H100 建/激活环境 —— `env_h100.sh`（`source`,不是执行）
```bash
INSTALL=1 source experiments/aerial/scripts/env_h100.sh   # 首次:建 .venv + 装 torch cu128 + airsim 客户端
source experiments/aerial/scripts/env_h100.sh             # 以后每次新终端:激活 + 自检
```
最小依赖(**非**整套 FastWAM):`torch==2.7.1+cu128` · `torchvision==0.22.1` · `numpy==1.26.4` · `pyyaml` · `airsim`/`opencv-python-headless`/`msgpack-rpc-python`。
自检结尾要看到 `[env] READY` + `cuda_available=True`。导出 `$AERIAL_PY` 供后续命令用。

### 4. 4090 起渲染器 —— `start_renderer_4090.sh`（独立终端常驻）
```bash
bash experiments/aerial/scripts/start_renderer_4090.sh    # 等 "ready to be connected"（~20s）
```
路径不同用 env 覆盖:`OPENFLY_ROOT=... ENV_NAME=... bash .../start_renderer_4090.sh`。
跨网绑定(`0.0.0.0:41451`)是一次性 OpenFly 配置,脚本不碰。

## 标准流程(从改代码到跑 ②④ gate)

1. **Mac**:改代码 → `bash .../sync_push.sh "msg"`。
2. **H100**:`bash .../sync_pull.sh`(首次用坑②的自举)→ `source .../env_h100.sh`(首次加 `INSTALL=1`)。
3. **4090**:`bash .../start_renderer_4090.sh` → 等 `ready`。
4. **H100** 跑 gate(artifact 走旧 checkout 绝对路径,见坑⑥):
```bash
"$AERIAL_PY" -m experiments.aerial.rl._v0_gate --signals 2,4 --rollout-eval \
  --config configs/aerial_rl_rollout.yaml \
  --depth-ckpt /home/a25689/aerial-rl-skeleton/experiments/aerial/rl/artifacts/depth_ckpt_da3_20260810/depth_step_2000_da3_head.pt \
  --rollout-dataset /home/a25689/aerial-rl-skeleton/experiments/aerial/rl/artifacts/dataset_v1_rgb \
  --device cuda --emit experiments/aerial/rl/artifacts/v0_partial_24.json
```
   先盯 `[v0-gate] obstacle-facing scan: {...}` 看 `accepted`/16。

## 六个坑（今早都踩过，已修进脚本）

1. **推错仓库**:从 `~/Projects`(sibling repo)推 → `src refspec ... does not match any`。
   分支只在 `robomaster-tt-control` 的 worktree 里。`sync_push.sh` 自动 `cd` 修掉。
2. **多行命令被拆行**:粘贴多行块,shell 把中间行当独立命令跑(`-bash: remote: No such file...`),
   导致 checkout 没执行、卡在旧 commit。**首次自举一行一条**:
   ```bash
   git fetch --all
   ```
   ```bash
   git checkout -B aerial-rl-skeleton origin/aerial-rl-skeleton
   ```
   带尖括号 `>` 的行更要小心(会被当重定向 → `syntax error near unexpected token newline`)。
3. **H100 是临时容器**:重建后 conda/venv、torch、pyyaml 全没(`ModuleNotFoundError: yaml` / `torch`)。
   → `INSTALL=1 source env_h100.sh` 一键重建。Task 的训练环境**不随盒子存活**。
4. **系统 python 无 ensurepip**(`apt install python3.10-venv` 提示):脚本自动 `venv --without-pip` +
   `get-pip.py`,无需 sudo/apt。若 `get-pip.py` 连不上 `bootstrap.pypa.io` → 换内网 pip 镜像。
5. **`import cv2` 报 `libGL.so.1`**:airsim 拉了非 headless 的 `opencv-python`(需 libGL,GPU pod 没有)。
   脚本强制卸载换 `opencv-python-headless`。pip 会警告 airsim 要 `opencv-contrib-python` —— **无害**,cv2 能用即可。
6. **artifacts 不在 git、随盒子丢**:H100 有**两个 checkout** —— `~/robomaster-tt-control`(本次新 clone,
   代码新但 artifacts 空)和 `~/aerial-rl-skeleton`(旧的,`dataset_v1_rgb` / `depth_ckpt_da3_20260810` /
   `wm_ckpt_v2clean_20260810` 全在)。**别拷贝**:从新 clone 跑,`--depth-ckpt`/`--rollout-dataset` 用
   `~/aerial-rl-skeleton/...` 绝对路径。共享盘 `/home/a25689/aerial_cache_shared/` 存 runs/orchestration,
   重建通常不清 —— 找丢失产物先搜这里和 `~/aerial-rl-skeleton`、`~/rl_collect_run`。

## 治理红线（不随手册放宽）

- 四信号(①非塌缩含①d深度 / ②接近量↑ / ③D̂尺度一致 / ④近障避让)**全过前不翻 flags**
  (`depth_head.enable` / `safety.kind` / `corrector.enable_wm_update` 保持 off)。
- 阈值(①d AbsRel ≤0.30、③ ≤0.25 等)= §4.1 冻结,改阈值需 re-freeze。
- 代码走 git(禁 scp 热补丁);`step_hz` 实测不猜。
