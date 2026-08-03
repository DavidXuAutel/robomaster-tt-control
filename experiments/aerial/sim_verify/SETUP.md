# SETUP — 从零搭建可验证的 Aerial 仿真环境

假设两台机器都是**裸机**。目标：把 OpenFly + AirSim 场景在 4090 上跑起来，然后用
`sim_verify` 探针拿到 **Fork A / A⁻ / B** 判定。

> **来源标注：**
> - 🟢 = OpenFly 上游 README 原文命令（照抄）
> - 🔵 = 本项目补充 / 架构建议
> - 🟡 = 上游未文档化、需你在机器上实测确认的点
>
> 上游要求：**Ubuntu 22.04 · CUDA · Python 3.10 · ROS2 Humble**。

---

## 0. 策略：先单机，再联网

| 阶段 | 做什么 | 为什么 |
|------|--------|--------|
| **Phase 1（先做）** | 全部在 **4090** 上跑：OpenFly + AirSim + `sim_verify` 都在本机，探针连 `127.0.0.1:41451` | 最快拿到 Fork 判定，绕开跨机 python 环境复制 + 防火墙 |
| **Phase 2（之后 RL 才需要）** | 把 **8×H100** 作为远程 client 连 4090 的 `41451` | 验证能力不需要 H100；RL rollout 采样才需要算力机做 client |

先把 Phase 1 跑通。下面 §1–§2 是 Phase 1，§3 是 Phase 2。

---

## 0.5 工作方式：先检查，缺什么才装 🔵

**不要盲装。** 本项目自带优先环境检查——先跑检查，只对「不通过」的项下载/安装：

```bash
cd sim_verify
cp config.env.example config.env    # 先填 OPENFLY_ROOT / ENV_NAME / 端口

./preflight.sh          # ① 只读检查：逐项 [OK]/[--]，列出缺什么 + 修复命令
CONFIRM=1 ./setup_env.sh   # ② 自动补 OpenFly 层缺失项（干跑去掉 CONFIRM=1）
./preflight.sh          # ③ 复检，直到关键项全 [OK]
./run_all.sh            # ④ run_all 自身也会先跑 preflight 当门禁
```

- `preflight.sh`：**只检查、不安装**。系统前置 + OpenFly 环境 + apt + python 客户端 + bridge 端口。
- `setup_env.sh`：**检查→缺则装**（幂等，可反复跑）。系统前置（驱动/CUDA/ROS2/conda）不自动装，
  缺则停并提示；OpenFly 层（env/clone/pip/apt/colcon/客户端依赖）自动补；场景因 HF 文件名多变仅提示。
- 下面 §1 各步是 `setup_env.sh` 背后做的事的**逐条展开**——手动排障或看它到底装了什么时对照用。

---

## 1. 4090 渲染主机 —— 从零安装

### 1.1 系统前置 🔵

```bash
# Ubuntu 22.04。确认 NVIDIA 驱动 + CUDA：
nvidia-smi                      # 能看到 4090 + 驱动版本
nvcc --version || echo "需装 CUDA toolkit（flash-attn 编译需要 nvcc）"
```

安装 **ROS2 Humble**（OpenFly 硬依赖，`env_bridge.py` 走 ROS2）——按官方 apt 步骤：
`https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html`
装完每个新终端都要：
```bash
source /opt/ros/humble/setup.bash   # 🟡 后续 colcon / env_bridge 都需要先 source
```

安装 miniconda（若无）：`https://docs.conda.io/en/latest/miniconda.html`

### 1.2 克隆 + conda 环境 🟢

```bash
git clone https://github.com/SHAILAB-IPEC/OpenFly-Platform.git /data/OpenFly-Platform
cd /data/OpenFly-Platform

conda create -n openfly python=3.10 -y
conda activate openfly
```

### 1.3 pip 依赖 🟢

```bash
pip install -r requirements.txt
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation   # 🟡 需 nvcc+torch；若只验证仿真可先跳过，报错再补
git clone https://github.com/kvablack/dlimp
cd dlimp && pip install -e . && cd ..
```

### 1.4 apt 依赖 🟢

```bash
sudo apt install -y xvfb libgoogle-glog-dev ros-humble-pcl-ros nlohmann-json3-dev
# xvfb = headless 渲染
```

### 1.5 构建 ROS2 工作区 🟢

```bash
cd /data/OpenFly-Platform/tool_ws
colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
# 构建后按上游惯例 source 该 ws（若 env_bridge 需要）：
source install/setup.bash          # 🟡 确认路径；ROS2 ws 通常这样 source
cd /data/OpenFly-Platform
```

### 1.6 下载 AirSim 场景 🟢

从 Hugging Face `IPEC-COMMUNITY/OpenFly_DataGen` 的 `airsim/` 目录下载并解压，把
`env_airsim_xxx/` 移到 `envs/airsim/`：

```bash
# 需要 huggingface_hub 或 git-lfs；示例用 hf CLI：
pip install -U huggingface_hub
huggingface-cli download IPEC-COMMUNITY/OpenFly_DataGen \
  --repo-type dataset --include "airsim/env_airsim_16*" \
  --local-dir /data/hf_openfly
# 解压后：
mkdir -p /data/OpenFly-Platform/envs/airsim
mv /data/hf_openfly/airsim/env_airsim_16* /data/OpenFly-Platform/envs/airsim/
chmod +x /data/OpenFly-Platform/envs/airsim/env_airsim_16*/*.sh 2>/dev/null || true
ls /data/OpenFly-Platform/envs/airsim/          # 应看到 env_airsim_16 目录
```
> 🟡 HF 上的确切文件名/压缩包名以页面为准；至少下 `seen.json` 用到的那个 env。

### 1.7 AirSim 能力配置（决定 Fork 的关键）🟡

上游 README **不文档化** AirSim `settings.json`/SimMode——能力由场景二进制 + 每场景
YAML 决定。两处要看：

1. **每场景 YAML** `configs/env_airsim_16.yaml`：定义 IP / 端口 / 编排参数。Phase 1 单机
   保持默认（localhost）即可；Phase 2 联网时改这里（见 §3）。
2. **AirSim `settings.json`**（标准 AirSim 会读 `~/Documents/AirSim/settings.json`）：
   决定 `SimMode` = `Multirotor`（有 IMU/碰撞/物理）还是 `ComputerVision`（只有相机）。
   OpenFly 的瞬移式 `set_drone_pos` 用法**疑似 ComputerVision 模式**。
   - **先不要改**：直接进 §2 跑探针，让 `verdict` 告诉你现网是哪种模式。
   - 若判定为 **Fork A⁻**（相机+深度有、IMU/物理无）→ 再考虑在 `settings.json` 里设
     `"SimMode": "Multirotor"` + 加 `Imu/Barometer/Gps` 传感器（见验证 spec §5.1），
     然后重启 bridge 重验。🟡 改后务必回归 OpenFly 现有瞬移 eval 是否还正常。

### 1.8 启动场景 bridge 🟢（独立终端，常驻）

```bash
cd /data/OpenFly-Platform
conda activate openfly
source /opt/ros/humble/setup.bash          # 每个新终端都要
python scripts/sim/env_bridge.py --env env_airsim_16
# 等约 20s，出现 "ready to be connected" 才算好
```

---

## 2. 在 4090 上跑验证（Phase 1，localhost）

**新开一个终端**（bridge 那个别关）：

### 2.1 拿到 sim_verify 🔵
把本项目目录拷到 4090（整目录可 scp / tar），例如 `/data/sim_verify`。或直接用本仓库
里的 `experiments/aerial/sim_verify/`。

### 2.2 客户端依赖 🔵
`airsim` 客户端已可能随 OpenFly 装好；没有就在 openfly 环境里补：
```bash
conda activate openfly
pip install -r /data/sim_verify/requirements.txt   # airsim numpy opencv msgpack-rpc
```

### 2.3 配置并跑 🔵
```bash
cd /data/sim_verify
cp config.env.example config.env
# 编辑 config.env：
#   AIRSIM_HOST=127.0.0.1          # 单机 localhost
#   AIRSIM_PORT=41451              # 与 bridge/YAML 一致
#   OPENFLY_ROOT=/data/OpenFly-Platform
#   ENV_NAME=env_airsim_16
nc -vz 127.0.0.1 41451             # 确认端口活着
./run_all.sh                       # 退出码 0=Fork A, 2=Fork A⁻, 3=Fork B
```

看终端能力矩阵 + `artifacts/sim_capability_report.json`，据 `verdict` 决定下一步。

---

## 3.（可选，Phase 2）8×H100 作为远程 client

**仅当 Phase 1 判定 Fork A、要进 RL 采样时才做。** 验证能力本身不需要这步。

### 3.1 4090 侧：让 41451 对外可达 🟡
- 改 `configs/env_airsim_16.yaml` 里的 IP/端口，让 AirSim RPC 绑到 `0.0.0.0` 或 4090 的
  内网 IP（不是 127.0.0.1），否则 H100 连不上。
- 放行防火墙：`sudo ufw allow 41451/tcp`（按实际网络策略）。
- **单消费者**：同一时刻只允许一个 client 连 `41451`。

### 3.2 H100 侧 🔵
```bash
ssh a25689@10.239.121.21 -p 31126
# 建轻量环境（只需 airsim 客户端 + numpy）：
python3 -m venv ~/sim_verify_venv && source ~/sim_verify_venv/bin/activate
pip install -r sim_verify/requirements.txt
cd sim_verify && cp config.env.example config.env
#   AIRSIM_HOST=10.229.20.125      # 4090 内网 IP
#   AIRSIM_PORT=41451
nc -vz 10.229.20.125 41451
python probes/t0_connectivity.py   # 先只测连通
python probes/t2_capability.py     # 纯 airsim 客户端，H100 上就能跑
./verdict.py
```
> 🟡 **T1（真渲染）在 H100 上跑需要 OpenFly 的 `scripts/sim/airsim_bridge.py` 及其
> python 依赖也在 H100 上可导入。** 若 `airsim_bridge.py` 只依赖 `airsim` 包，把
> `OPENFLY_ROOT` 指向一份 clone 即可；若它拉 ROS2 等重依赖，**T1 就留在 4090 上跑，
> T0/T2 放 H100**——两机结果都 merge 进同一份 report 即可判定。

---

## 4. 校验点 / 常见坑

| 现象 | 处理 |
|------|------|
| `env_bridge.py` 卡住不出 "ready" | 等满 20s；确认场景已解压到 `envs/airsim/`、`.sh` 有可执行权限、xvfb 已装 |
| `import airsim` 失败 | `pip install airsim`；注意是纯 python RPC 客户端 |
| ROS2 命令找不到 / colcon 报错 | 每个终端先 `source /opt/ros/humble/setup.bash` |
| flash-attn 编译失败 | 若只验证仿真可先跳过 §1.3 的 flash-attn；确需时装匹配的 torch+CUDA 再编 |
| T2 的 `imu`/`physics` FAIL | 大概率 ComputerVision 模式 → Fork A⁻，见 §1.7 切 Multirotor |
| H100 连不上 41451 | 4090 的 YAML 把 RPC 绑到了 127.0.0.1；改绑内网 IP + 放行防火墙（§3.1） |

---

## 5. 从零到判定 —— 最短清单（先检查后装）

```
[ ] 4090: 手动装系统前置 Ubuntu22.04+NVIDIA驱动/CUDA+ROS2 Humble+conda  (§1.1)
[ ] 4090: cp config.env.example config.env（填 OPENFLY_ROOT/ENV_NAME）  (§0.5)
[ ] 4090: ./preflight.sh                     # 看缺什么                  (§0.5)
[ ] 4090: CONFIRM=1 ./setup_env.sh           # 只补 OpenFly 层缺失项     (§0.5)
[ ] 4090: 手动下载 env_airsim_16 场景到 envs/airsim/                     (§1.6)
[ ] 4090: 启动 env_bridge.py，等到 "ready to be connected"              (§1.8)
[ ] 4090: ./preflight.sh 复检全 [OK] → ./run_all.sh                     (§2)
[ ] 读 verdict → Fork A / A⁻ / B，贴回结果                               (§2.3)
[ ] (A⁻ 才做) 调 settings.json 切 Multirotor 重验                        (§1.7)
[ ] (RL 才做) Phase 2 联网 H100 client                                  (§3)
```
