# Aerial 仿真环境能力验证 Spec（Sim Capability GO / NO-GO）

**状态：** DRAFT v1.3
**日期：** 2026-08-03
**服务对象：** `2026-08-03-aerial-wam-pure-vision-design-v2.md` §0.5 的 GO/NO-GO 前置
**产出：** 一份能力矩阵 + Fork A / A⁻ / B 判定 + 机器可读报告 `sim_capability_report.json`
**落地实现：** `experiments/aerial/sim_verify/`（自包含独立项目：`preflight.sh` / `setup_env.sh` / `run_all.sh` / `verdict.py` / `probes/` / `SETUP.md`；分支 `aerial-wam`）

> **一句话：** 在投入 v2 任何模型实现之前，用可复现的探针确认仿真器**到底能提供哪些信号**——真实场景 RGB、IMU、气压/GPS、碰撞、深度 GT、物理步进、**连续稠密帧（L2f）**——并据此把项目锁定到 Fork A（v2 可行）/ A⁻（需切物理模式或修帧率）/ B（不可达）。API 可读不等于可用：探针含**数值健全性**。

> **v1 → v1.1 变更：**
> 1. 落地为自包含独立项目 `experiments/aerial/sim_verify/`；§6 的内联脚本降级为参考实现，**以项目文件为准**。
> 2. 新增**环境优先检查（preflight-first）**原则（§5.3）：先检查、缺什么才装。
> 3. 依据 OpenFly 上游 README 补正搭建事实：**Ubuntu 22.04 · CUDA · Python 3.10 · ROS2 Humble · colcon `tool_ws`**；场景为 HF 预打包 UE+AirSim 二进制、默认 headless(xvfb)。
> 4. **修正：** `run_closed_loop.py` **无 `--env`/`--env-name` 参数**（env 名按 episode 从标注推），原 T3 命令里的 `--env-name` 删除。
> 5. **诚实标注：** 上游**未文档化** `settings.json`/SimMode；「切 Multirotor」是标准 AirSim 行为的推断，需在机器上实测确认，不作为既定事实。

> **v1.1 → v1.2 变更：**
> 1. **L2f 升格为硬门禁：** 连续 Scene 抓取须报告实际 fps、时间戳单调性、运动提示后的帧间差分；达不到 `L2F_MIN_FPS`（默认 5）不得标 Fork A（光流 `[1d]` / 多帧深度 `[1b]` 依赖）。
> 2. **数值健全性：** IMU 近零加速度、深度非稠密/近常数、高度非有限值 → 该项 `pass=false`（避免「API 能调但信号是死的」误判为 A）。
> 3. Fork A 额外要求 **L2b 高度（气压或 GPS 其一）** 与 **L2f**；与 v2 §6.3 / `[1d]` 对齐。
> 4. 落地：`lib/sanity.py`、`probes/t2_capability.py`、`verdict.py`、`tests/test_sanity.py`；离线单测不连 AirSim。

> **v1.2 → v1.3 变更（与实现对齐修正）：**
> 1. **T0 修正：** 原 §6 T0 写成「离线 mock 接线自检」，但实际 `probes/t0_connectivity.py` 是 **RPC 连通性 + SimMode 猜测**（`verdict` 读 `t0_connectivity.connected`）。改正 T0，mock 自检降为可选旁注。
> 2. **§4.2 建环境命令修正：** `conda env create -f environment.yml` → `conda create -n openfly python=3.10` + `pip install -r requirements.txt`（对齐 `setup_env.sh` 与上游）。
> 3. **§6 T2 内联旧代码收敛：** 删除 `physics_step`/`scene_capture` 等旧探针名，改为与 `probes/t2_capability.py` 一致的简述（键名 `physics`/`continuous_frames`/`scene`）。
> 4. **§5.2 补全** `AIRSIM_CAMERA` / `L2F_*`；§7 判定表标注退出码 `0/2/3` 并把 Fork B 条件对齐 verdict（T0 连不上或 T1 无真图）；§9 树补 `lib/__init__.py`。

---

> **Phase-1 @ 10.229.20.125（2026-08-03）：** Multirotor 下 L1/L2a–e 通过。干净独占 L2f @ **1920×1080** ≈**3.12 fps**（&lt;5）曾记 A⁻。**工作分辨率复测**（训练默认 **224×224** CaptureSettings，含 `Binaries/Linux/settings.json`）：顺序 ≈**36.3 fps**、批量 ≈**90 fps**，IMU≈9.84 → **`L2F_MIN_FPS=5` 达标，正式 Fork A（工作分辨率）**。1080p ~3 fps 仅作高清可视化约束。详见 `experiments/aerial/sim_verify/artifacts/phase1_125_SUMMARY.md` / `l2f_clean_retest_224.json`。


## 1. 目的与范围

v2 的惯性 + 稠密感知地基**无法**由静态 OpenFly parquet 提供（无 IMU、稀疏宏原语关键帧）。唯一来源是**能渲染任意视角的物理仿真器**。本 spec 只回答一个问题并给出证据：

**这台仿真器能不能实时产出 v2 所需的全部信号？** 分三级能力核验，输出确定性判定。

**不在范围内：** 模型训练、策略设计、数据转换（见 v2 与各自 spec）。

---

## 2. 背景：现有基础设施（已知可用）

核实自 `experiments/aerial/eval/run_closed_loop.py`、`eval/README.md`、`scripts/deploy_orchestration.md`、`scripts/stage0_oracle_eval.sh`：

| 项 | 值 | 来源 |
|----|-----|------|
| OpenFly 平台 | `https://github.com/SHAILAB-IPEC/OpenFly-Platform.git` | eval/README.md:21 |
| `OPENFLY_ROOT` | `/home/a25689/aerial_eval_cache/OpenFly-Platform` | stage0_oracle_eval.sh:12 |
| Bridge 封装 | `<root>/scripts/sim/airsim_bridge.py` → `AirsimBridge(env_name)` | run_closed_loop.py:253 |
| Bridge 启动器 | `python scripts/sim/env_bridge.py --env env_airsim_16`（独立终端） | eval/README.md:36 |
| **训练/推理/eval-client** | `ssh a25689@10.239.121.21 -p 31126`（8×H100，单机跑训练+推理+闭环 client） | 用户 2026-08-03 |
| **AirSim 渲染主机** | `10.229.20.125`（1×4090，跑 Unreal+AirSim 场景），RPC 默认 `41451`，**单消费者** | 用户 2026-08-03 |
| 环境变量 | `AIRSIM_HOST=10.229.20.125` `AIRSIM_PORT=41451` | — |
| 场景命名 | `env_airsim_16` 等；场景放 `envs/airsim/env_airsim_xx/` | eval/README.md:29 |
| 标注 | `seen_airsim16_m1a20.json` | stage0_oracle_eval.sh:11 |

**运行前提（先确认，别只靠本 spec）：**
- **实现只在 `aerial-wam` worktree：** 主分支（robomaster-tt-control `main`）**没有** `experiments/aerial/`。跑验证前先 `ls experiments/aerial/sim_verify/` 确认脚本齐全，**以项目文件为准，不要照抄 §6 的内联示例**。
- **主机以本表为准：** 本表取代 v2 §0.5.3 里过时的 `:30905`。跑前先改 `config.env`，按当前主机实际路径/端口填，勿死抄旧端口。

**关键约束（硬）：**
- **单消费者：** 4090 渲染主机同一时刻只允许一个 AirSim 客户端。验证脚本从 8×H100 客户端发起时，必须确保没有其它 eval/训练进程同时连 `41451`，否则会抢占。
- **跨机联通：** 客户端在 `10.239.121.21`（H100），渲染器在 `10.229.20.125`（4090）；验证前先 `nc -vz 10.229.20.125 41451` 确认 RPC 端口可达。
- 现有 `OpenFlyBridge` 仅调用 `get_camera_data("color")` + `set_drone_pos(x,y,z,pitch,yaw°,roll)` **瞬移**（run_closed_loop.py:277/291）——**没有物理、没有 IMU、没有碰撞、没有深度**。这强烈暗示 AirSim 跑在 **ComputerVision 模式**（可移动相机，无飞行动力学）。此点是 L2 的核心待验项。

---

## 3. 能力分级（验证目标）

| 级别 | 能力 | 现状 | AirSim 依赖 | v2 用途 |
|------|------|------|-------------|---------|
| **L0** | mock 运动学 + 伪 RGB | ✅ 已有（`--bridge mock`） | 无 | 仅离线接线自检 |
| **L1** | 连接渲染器 + **真实场景 RGB** + 瞬移到任意位姿 | ✅ 已用（stage0 `--bridge openfly`） | Scene 相机 + `simSetVehiclePose`/`set_drone_pos` | grounding、拓扑视觉记忆、WM 像素预训练 |
| **L2a** | **IMU**（角速度/加速度/姿态） | ❓ 未接线 | `getImuData` + Multirotor 模式 | VIO `[1c]`、尺度锚定 |
| **L2b** | **气压/GPS 高度** | ❓ 未接线 | `getBarometerData`/`getGpsData` | 3D 净空、Z 尺度 |
| **L2c** | **碰撞检测** | ❓ 未接线 | `simGetCollisionInfo` | 避障监督、`p_coll`、RL 代价 |
| **L2d** | **深度 GT** | ❓ 未接线 | `simGetImages(DepthPlanar)` | 深度头 `[1b]` 监督 |
| **L2e** | **物理步进**（真飞行，非瞬移） | ❓ 疑为 CV 模式 | Multirotor `moveByVelocity*` | 连续控制、闭环 RL |
| **L2f** | **连续帧率**（关键帧之间的稠密帧） | ❓ | 自由触发相机 + 时序探针 | 光流 `[1d]`、多帧深度、稠密视频 WM |

**L2 决定 Fork：** AirSim 原生支持全部 L2，但**前提是运行在 Multirotor 模式**。若 OpenFly 用 ComputerVision 模式，则 L2a/c/e 缺失，需切模式（Fork A⁻）。**L2f** 即使在有相机的模式下也可能因 RPC/渲染节流而帧率不足——须实测，不得用「单次 Scene 抓取成功」代替。

**健全性（与「API 能调用」正交）：**
| 项 | 拒绝条件（示例） |
|----|------------------|
| IMU | `lin_acc` 模长 < 0.5（近零死信号）或非有限值 |
| 气压/GPS | 高度非有限 |
| 深度 | 非稠密、有限像素 < 50%、动态范围近常数 |
| 连续帧 | 时间戳非单调、`fps < L2F_MIN_FPS`（默认 5）、运动提示后帧间无差异 |

---

## 4. 环境需求与搭建

### 4.1 硬件 / OS
| 项 | 要求 |
|----|------|
| 渲染主机 | Linux + NVIDIA GPU（跑 Unreal + AirSim 场景二进制） |
| 客户端主机 | Linux（能 `import airsim` 与 OpenFly deps），可与渲染主机同机或经 RPC `41451` 联网 |
| macOS / 无 GPU | **仅能跑 L0 mock**；L1/L2 一律不可（eval/README.md:14） |

### 4.2 搭建步骤（渲染主机）
```bash
# 1) 克隆 OpenFly 平台
git clone https://github.com/SHAILAB-IPEC/OpenFly-Platform.git /data/OpenFly-Platform
cd /data/OpenFly-Platform

# 2) 按上游 README 建 conda 环境 + 装依赖（与 setup_env.sh 一致；上游用 requirements.txt，非 environment.yml）
conda create -n openfly python=3.10 -y
conda activate openfly
pip install -r requirements.txt && pip install packaging ninja
pip install airsim opencv-python numpy   # sim_verify 客户端依赖（见 requirements.txt）

# 3) 下载至少一个 seen.json 用到的 AirSim 场景
#    from https://huggingface.co/datasets/IPEC-COMMUNITY/OpenFly_DataGen/tree/main/airsim
#    into envs/airsim/env_airsim_16/   (与标注里的 env 名一致)

# 4) 配置 AirSim settings.json —— 决定 L2 能力的关键！
#    默认位置：~/Documents/AirSim/settings.json
#    见 §5.1；ComputerVision 无 IMU/碰撞/物理，Multirotor 全有。

# 5) 启动场景 bridge（独立终端，常驻）
python scripts/sim/env_bridge.py --env env_airsim_16
```

### 4.3 网络与并发
- 场景 bridge 与 AirSim 二进制跑在 **4090 (`10.229.20.125`)**；客户端脚本从 **8×H100 (`10.239.121.21:31126`)** 连 `AIRSIM_HOST:AIRSIM_PORT`（`10.229.20.125:41451`）。
- **运行任何验证前**：`nc -vz 10.229.20.125 41451` 确认可达，并确认无其它进程占用该 AirSim（单消费者约束）。

---

## 5. 配置

### 5.1 AirSim `settings.json`（L2 成败的开关）
```jsonc
// Multirotor 模式 —— v2 需要（IMU/碰撞/物理全有）
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "Vehicles": {
    "drone1": {
      "VehicleType": "SimpleFlight",
      "Sensors": {
        "Imu":        { "SensorType": 2, "Enabled": true },
        "Barometer":  { "SensorType": 1, "Enabled": true },
        "Gps":        { "SensorType": 3, "Enabled": true }
      }
    }
  },
  "CameraDefaults": {
    "CaptureSettings": [
      { "ImageType": 0, "Width": 640, "Height": 480 },   // Scene(RGB)
      { "ImageType": 1, "Width": 640, "Height": 480 }    // DepthPlanar
    ]
  }
}
```
> 若现网是 `"SimMode": "ComputerVision"`：只有相机，无 IMU/碰撞/物理 → 落 Fork A⁻，需切 Multirotor 后重验。切模式可能影响 OpenFly 现有瞬移式 eval，须回归。

### 5.2 验证脚本环境变量
> 在 8×H100 客户端 (`a25689@10.239.121.21:31126`) 上导出。`OPENFLY_ROOT` / `ANN`
> 为旧机路径示例，**需按本机实际落位调整**（用户在新机上仍是 `a25689`，家目录约定可能沿用，但先 `ls` 确认）。
```bash
export OPENFLY_ROOT=$HOME/aerial_eval_cache/OpenFly-Platform   # 确认实际路径
export AIRSIM_HOST=10.229.20.125                               # 4090 渲染器
export AIRSIM_PORT=41451
export AIRSIM_CAMERA=0                                         # Scene/Depth 抓取用的相机 id（0 失败可试 front_center）
export ENV_NAME=env_airsim_16
export ANN=$HOME/aerial_cache_shared/orchestration/heldout/seen_airsim16_m1a20.json  # 仅 T3 需要，确认实际路径
export OUT=./artifacts/sim_capability_report.json
# L2f 连续帧探针（见 config.env.example）
export L2F_N=10                 # 采帧数
export L2F_INTERVAL_S=0.05      # 目标帧间隔（~20 Hz）
export L2F_MIN_FPS=5.0          # 实际 fps 低于此则 continuous_frames 判 fail
```

### 5.3 环境优先检查（preflight-first）

**核心原则：先检查、缺什么才装、可反复跑。** 不盲装。项目把它拆成两个幂等脚本 + 一套共用检查库：

| 脚本 | 角色 |
|------|------|
| `lib/checks.sh` | 共用检查函数（系统前置 / OpenFly 环境 / apt / python 客户端 / bridge 端口），各返回 OK/缺失 |
| `preflight.sh` | **只读**优先检查：逐项 `[OK]/[--]` + 修复命令；关键项缺则 `exit 1` |
| `setup_env.sh` | **检查→缺则装**（幂等）：系统前置缺则停并提示；OpenFly 层自动补；默认干跑，`CONFIRM=1` 才落地 |

标准流程：

```bash
cp config.env.example config.env    # 填 OPENFLY_ROOT / ENV_NAME / 端口
./preflight.sh                      # ① 看缺什么
CONFIRM=1 ./setup_env.sh            # ② 只补不通过的 OpenFly 层项
./preflight.sh                      # ③ 复检直到关键项全 [OK]
./run_all.sh                        # ④ run_all 内置 preflight 门禁 → 探针 → verdict
```

**边界（有意为之）：**
- **系统前置**（NVIDIA 驱动 / CUDA / ROS2 Humble / conda 本体）**不自动装**——太重、需人工决策；缺则停并指向 §4.2 / SETUP.md §1.1。
- **场景不自动下载**——HF 压缩包文件名多变，`setup_env.sh` 只打印下载命令让人确认（§4.2）。
- `setup_env.sh` 默认 **DRY-RUN**（只打印计划），显式 `CONFIRM=1` 才执行，避免在机器上意外跑 `sudo apt` / `colcon`。

---

## 6. 测试方法与脚本

> **以项目文件为准：** 下列脚本已落地为 `experiments/aerial/sim_verify/probes/` + `verdict.py`；本节保留其逻辑与判据作为设计说明。运行方式见 §5.3（`./run_all.sh` 串起 preflight → T0/T1/T2 → verdict）。

`run_all.sh` 串起的三个探针（T0→T1→T2）从便宜到决定性，全部把结果并入 `sim_capability_report.json`，`verdict.py` 读取。**在能连 `41451` 的客户端上运行，且确保无其它 AirSim 客户端。** 报告键名 = 探针名：`t0_connectivity` / `t1_render` / `t2_capability` / `verdict`。

> **可选离线自检（不在 run_all 内、不碰 AirSim）：** 在任意机器（含 macOS）上跑
> `run_closed_loop.py --bridge mock --policy replay`（配 `tests/fixtures/mini_openfly`）确认 runner 本身没坏。项目未把它做成探针，仅作接线级 smoke。

### T0 — RPC 连通性 + 模式猜测（`probes/t0_connectivity.py`）
最便宜的真实检查：(1) TCP 端口可达（等价 `nc -vz`），(2) `import airsim`，(3) 连接并粗猜 SimMode（`MultirotorClient` 成功且 `getMultirotorState()` 可读 → Multirotor；回退 `VehicleClient` → ComputerVision(likely)）。**不需要 OpenFly clone。** 并入 `t0_connectivity`，`connected` 为 `verdict` 判 Fork B 的依据。
**通过判据：** `connected=true`。TCP 不可达或 `import airsim` 失败即 fail（先修网络/依赖）。

### T1 — 真渲染（现有 L1 路径，`probes/t1_render.py`）
用 OpenFly `AirsimBridge($ENV_NAME)` 瞬移到位姿、取 `get_camera_data("color")`，验证是**真实渲染帧**而非 MockBridge 常值填充。
**通过判据：** 返回 `(H,W,3)` 且 `img.std() > 3.0`（像素方差显著 → 真场景）。需 `$OPENFLY_ROOT` clone + 匹配 `$ENV_NAME` 的场景；缺则 skip/fail。

### T2 — 直连 AirSim 能力探针（决定性，L2，`probes/t2_capability.py`）
**绕过瘦封装**，直接用 `airsim` 客户端逐项探测，判 Fork 的核心。连接优先 `MultirotorClient`，失败回退 `VehicleClient`（CV 模式）。每项 `try/except` 报告实际可用性，**API 可读还要过 `lib/sanity.py` 数值健全性才算 `pass`**。探测项（报告键）：`imu` / `barometer` / `gps` / `collision` / `depth` / `physics` / `continuous_frames`（L2f，多帧）+ 回退 `scene`（单帧，仅证明 Scene API 存在，**不充当 L2f 证据**）。
**逐项通过判据（实现以 `probes/t2_capability.py` + `lib/sanity.py` 为准）：**
- `imu`：可调用 **且** `lin_acc` 有限、模长 ≥ 0.5 → L2a ✅（全零/死信号 → fail）
- `barometer` / `gps`：高度有限 → L2b ✅（Fork A 要求二者之一）
- `collision.has_collided`：字段可读 → L2c ✅
- `depth`：稠密 + 有限像素占比 ≥ 50% + 动态范围非平凡 → L2d ✅
- `physics.moved`：`|dz| > 0.2` → L2e ✅（真物理，非瞬移）
- `continuous_frames`：固定间隔采 `L2F_N` 帧；时间戳单调；实际 `fps ≥ L2F_MIN_FPS`；运动提示后帧间 mean-abs-diff 显著 → L2f ✅
- 单次 `scene` 抓取**不能**单独充当 L2f 通过证据
- `mode_guess == ComputerVision(likely)` 或 IMU/物理/L2f 失败 → 落 **Fork A⁻**

> **API 兼容提醒：** OpenFly 可能用 Microsoft AirSim / Cosys-AirSim / Colosseum 之一，个别方法签名有差异。脚本用逐项 try/except **报告实际可用性**，不假设。若某项报错，先查该 fork 的 API 名再判「真缺失」还是「名不同」。

> **环境变量（L2f）：** `L2F_N`（默认 10）、`L2F_INTERVAL_S`（默认 0.05）、`L2F_MIN_FPS`（默认 5.0），见 `config.env.example`。

### T3 — 端到端渲染回归（可选，确认切模式没弄坏现有 eval）
仅当 §5.1 从 CV 切到 Multirotor 后跑，确认现有瞬移式 eval 仍出图：
```bash
# scripts/verify_sim/t3_openfly_eval_smoke.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"; export PYTHONPATH=.
# 注意：run_closed_loop.py 无 --env/--env-name；env 名按 episode 从标注推。
python3 -m experiments.aerial.eval.run_closed_loop \
  --bridge openfly --policy replay \
  --openfly-root "$OPENFLY_ROOT" \
  --ann "$ANN" --max-episodes 1 --max-steps 20 \
  --dump-frames /tmp/t3_frames --out /tmp/t3_openfly.json
echo "[T3] PASS if metrics.json written and frames non-placeholder"
```

---

## 7. 判定规则（能力矩阵 → Fork）

汇总 `sim_capability_report.json`：

`verdict.py` 退出码即 Fork：`0=A`、`2=A⁻`、`3=B`（`run_all.sh` 透传）。

| 结果 | Fork（退出码） | 含义与下一步 |
|------|------|--------------|
| T0 `connected` ✅ 且 T1 ✅ 且 T2：**IMU + (气压\|GPS) + 碰撞 + 深度(健全) + 物理 + 连续帧(L2f)** 全 ✅ | **A（0）** | v2 可行。进 v2 §7 V0：先固化「sim 出 IMU+稠密帧+碰撞+连续 RGB」的 rollout 采集接口，再训 `[1b]/[1c]/[1d]`。 |
| T1 ✅、深度/相机 ✅，但 IMU/物理/L2f 任一 ❌（CV 模式或帧率不足） | **A⁻（2）** | 尝试切 `SimMode: Multirotor` + 加 Sensors（§5.1，⚠️ 上游未文档化此开关，需实测确认场景二进制读 `~/Documents/AirSim/settings.json`）；或提高渲染/RPC 帧率后重跑 L2f。跑 T3 回归，再重跑 T2；成功则升级到 A。 |
| T0 连不上 **或** T1 ❌（只有占位图） | **B（3）** | 仿真渲染不可用 → v2 不可达。要么修渲染主机/场景，要么承认「离散原语 VLN，SR≈10%」并补数据采集。 |

**任何情况下，在报告落到 A 之前，不投入 v2 模型实现（对齐 v2 §0.5.4）。**

---

## 8. 风险与注意

| 风险 | 缓解 |
|------|------|
| 单消费者被抢占 | 验证期停 eval worker；确认渲染主机无第二客户端（deploy_orchestration.md:77） |
| 切 Multirotor 破坏现网瞬移 eval | 先备份 `settings.json`；T3 回归；切换在维护窗口做 |
| AirSim fork API 差异 | T2 逐项 try/except 报告，不假设；差异项查对应 fork 文档 |
| 场景未下载全 | 每个 `seen.json` 用到的 `env_*` 都要下；缺场景 → T1 失败 |
| CV 模式下深度/相机可用但 IMU/碰撞不可用 | 明确落 A⁻，不误判为 A |
| 单次 Scene 成功但稠密时序不够 | L2f 硬门禁；不得用单帧冒充连续视频能力 |
| API 返回全零 IMU / 常数深度 | 健全性检查标 fail，避免假 Fork A |
| sim-to-real（若最终真机部署） | 本 spec 只验 sim 能力；真机 IMU 来自飞控，另行验证（见 v2 §1.2） |

---

## 9. 交付物

1. `artifacts/sim_capability_report.json` —— preflight + T0–T2 逐项结果（机器可读，含 `verdict` 键）。
2. 一句话判定：**Fork A / A⁻ / B**，写回 v2 §0.5.4 的决策位。
3. 若 A⁻：一份 `settings.json` diff（CV→Multirotor）+ T3 回归结论。

**已落地项目结构** `experiments/aerial/sim_verify/`：

```
sim_verify/
├── SETUP.md                 # 从零搭建运行手册（系统前置→OpenFly→跑通）
├── README.md                # 项目说明 + 快速开始
├── requirements.txt         # 客户端依赖 airsim/numpy/opencv/msgpack-rpc
├── config.env.example       # 主机/端口/路径/L2F_* 模板
├── preflight.sh             # 优先环境检查（只读）
├── setup_env.sh             # 检查→缺则装（幂等，CONFIRM=1 落地）
├── run_all.sh               # preflight 门禁 → T0/T1/T2 → verdict
├── verdict.py               # 能力矩阵 → Fork A/A⁻/B（退出码 0/2/3；含 L2b/L2f）
├── lib/{__init__.py, checks.sh, report.py, sanity.py}
├── probes/{t0_connectivity.py, t1_render.py, t2_capability.py, t3_repo_eval_smoke.sh}
└── tests/test_sanity.py     # 离线健全性单测（不连 AirSim）
```

---

## 10. 关联文档

| 文档 | 角色 |
|------|------|
| `experiments/aerial/sim_verify/` | **本 spec 的落地实现**（脚本 + SETUP + README） |
| `experiments/aerial/sim_verify/SETUP.md` | 从零搭建运行手册（含 OpenFly 上游步骤） |
| `2026-08-03-aerial-wam-pure-vision-design-v2.md` §0.5 | 本验证服务的 GO/NO-GO 前置 |
| `experiments/aerial/eval/run_closed_loop.py` | Bridge 接口事实源（Mock / OpenFly；无 `--env` 参数） |
| `experiments/aerial/eval/README.md` | OpenFly Linux 搭建步骤 |
| OpenFly-Platform 上游 README | 权威搭建步骤（Ubuntu22.04/ROS2/conda/colcon/场景） |
