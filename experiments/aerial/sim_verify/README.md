# sim_verify — Aerial 仿真能力验证（独立项目）

在投入 v2 任何模型实现前，用可复现探针确认仿真器（4090 上的 AirSim）到底能产出哪些
v2 需要的信号，并输出 **Fork A / A⁻ / B** 判定 + 机器可读报告。

- **从零搭建环境：先看 [`SETUP.md`](SETUP.md)**（裸机 → OpenFly + AirSim → 跑通探针）。
- 设计依据：`docs/superpowers/specs/2026-08-03-aerial-sim-capability-verification-spec-v1.md`
- 自包含：关键测试（T0/T1/T2）只依赖 `airsim` + `numpy` + OpenFly clone，**不依赖本仓库
  其余代码**，可整目录 `tar` 后 scp 到任意主机。

> 下面的「快速开始」假设 `SETUP.md` 已把 OpenFly + AirSim 装好、bridge 已启动。

## 设备拓扑（2026-08-03）

| 角色 | 主机 |
|------|------|
| 客户端（跑本项目脚本） | 8×H100 `a25689@10.239.121.21 -p 31126` |
| AirSim / Unreal 渲染器 | 1×4090 `10.229.20.125`，RPC `41451`，**单消费者** |

`airsim` 是纯 python RPC 客户端，从 H100 直接连 `10.229.20.125:41451` 即可，无需在 H100 上跑 Unreal。

## 快速开始

```bash
# 0) 在 4090 上：确认 AirSim 场景 bridge 已启动（独立终端常驻）
#    cd <OPENFLY_ROOT> && python scripts/sim/env_bridge.py --env env_airsim_16

# 1) 客户端上：
cd experiments/aerial/sim_verify
cp config.env.example config.env      # 编辑：确认 OPENFLY_ROOT / 端口 / ENV_NAME

# 2) 优先环境检查 —— 先看缺什么，缺 OpenFly 层才自动装
./preflight.sh                        # 只读检查，列出 [OK]/[--] + 修复命令
CONFIRM=1 ./setup_env.sh              # 只补「不通过」的 OpenFly 层项（干跑：去掉 CONFIRM=1）
./preflight.sh                        # 复检直到关键项全 [OK]

# 3) 一键跑全部（run_all 内部也会先跑 preflight 当门禁；强制跳过用 FORCE=1）
./run_all.sh
```

`run_all.sh` 退出码即判定：`0=Fork A`，`2=Fork A⁻`，`3=Fork B`。

## 测试项

| 探针 | 检什么 | 依赖 |
|------|--------|------|
| `probes/t0_connectivity.py` | TCP 可达 + `import airsim` + 连接 + SimMode 猜测 | airsim |
| `probes/t1_render.py` | OpenFly `AirsimBridge` 取**真实场景 RGB**（`std>3` 排除 mock 常值填充） | airsim + OpenFly clone + 场景 |
| `probes/t2_capability.py` | 直连 AirSim 探 **IMU / 气压 / GPS / 碰撞 / 深度GT / 物理 / 连续帧(L2f)**；API 可读 + **数值健全性** 才算 pass | airsim（建议 numpy/opencv 解码帧差） |
| `verdict.py` | 汇总能力矩阵 → Fork A/A⁻/B（含 L2b 高度、L2f） | — |
| `probes/t3_repo_eval_smoke.sh`（可选） | 切 Multirotor 后跑本仓库闭环 runner 回归 | 本仓库 + OpenFly |
| `tests/test_sanity.py` | 离线单测健全性 helper（不连 AirSim） | 标准库 |

## 判定含义

| Fork | 条件 | 下一步 |
|------|------|--------|
| **A** | 真 RGB + IMU + (气压\|GPS) + 碰撞 + 深度(健全) + 物理 + **连续帧 L2f** 全通过 | v2 惯性+稠密+光流地基可行 → 进 v2 §7 V0 |
| **A⁻** | 相机+深度通过，但 IMU/物理/L2f 等缺失（多半 CV 模式或帧率不够） | 改 `SimMode:Multirotor` / 修相机帧率，T3 回归后重验 |
| **B** | 连不上 / 只有占位图 | 渲染器不可用，v2 不可达；先修渲染主机/场景 |

**健全性（节选）：** IMU `lin_acc` 近零 → fail；深度非稠密/几乎常数 → fail；L2f 要求时间戳单调、`fps ≥ L2F_MIN_FPS`、运动提示后帧间有差异。

## 注意

- **单消费者：** 运行前确认 4090 上没有其它 AirSim 客户端在连（会抢占）。
- `t2_capability.py` 的 `physics` 项会短暂 arm 无人机并上升 1s，仅在 Multirotor 模式生效；
  跑前确保场景内该动作安全。
- 各 fork（Microsoft AirSim / Cosys-AirSim / Colosseum）API 名可能有差异；探针不假设，
  失败项会打印具体异常，据此判断是「真缺失」还是「方法名不同」。
- 报告默认写 `./artifacts/sim_capability_report.json`。
