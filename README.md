# Aerial WAM 训练 runtime（`aerial-wam` 分支）

这是 **aerial WAM 无人机导航训练**的运行时代码分支，属于 `robomaster-tt-control` 仓库的一条
**孤儿分支**（`aerial-wam`），与主线的 RoboMaster TT 视觉避障代码互不相关、各自独立。

## 这是什么

整个 **FastWAM** 训练框架已 **vendoring**（整包拷入）到本分支，来源：

> FastWAM `feat/aerial-b0-b1-orchestration` @ `46a1138`

之所以整包搬入，是因为 aerial 训练跑在整个 `fastwam` 包之上（`fastwam.trainer` /
`fastwam.datasets.lerobot.*` / `fastwam.runtime`），不是独立可运行的模块。整包 vendoring 让
**checkout 即可跑**，符合设计文档「单一事实源 = git checkout」原则。

**代价**：本分支已与 FastWAM 上游脱钩，上游核心修复需手动同步回来。

## 目录结构

| 路径 | 内容 |
|------|------|
| `src/fastwam/` | FastWAM 核心包（含 v2 两处改动：`trainer.py` resume guard、`datasets/lerobot/weighted_source_dataset.py` 确定性采样） |
| `experiments/aerial/` | aerial 专属代码：数据转换、评测、编排、专家、tests、启动脚本 |
| `configs/` | model / data / task 配置 |
| `scripts/` | `train.py` 训练入口 + accelerate / deepspeed 配置 + 环境脚本 |
| `third_party/` | 运行时依赖（Wan 等） |
| `docs/design/` `docs/handover/` | v2 重设计方案 + v1 交接文档 |
| `README_FastWAM.md` / `README_zh.md` | FastWAM 上游原始 README（框架细节） |

## 怎么跑（训练主机 `:31126`，1 机 2×H100）

```bash
export RUNTIME=/home/a25689/aerial_wam_runtime/robomaster-tt-control
# 首次：git clone -b aerial-wam <robomaster-remote> "$RUNTIME"
git -C "$RUNTIME" checkout aerial-wam && git -C "$RUNTIME" pull
pip install -e "$RUNTIME"          # 装 fastwam 包
cd "$RUNTIME"

# 三段式启动（详见 docs/design/2026-07-29-aerial-nav-wam-redesign.md §8）
experiments/aerial/scripts/run_b0_v2_from_scratch.sh preflight   # 只做检查
experiments/aerial/scripts/run_b0_v2_from_scratch.sh smoke       # 10 步试跑估预算
MAX_STEPS=<按 smoke 估算> experiments/aerial/scripts/run_b0_v2_from_scratch.sh train
```

评测在 `:30682`（1×H100 + AirSim），协议见设计文档 §8.5；**只有 SR>0 且可复现才锁定 baseline**。

## 完整方案

- 设计（v2，从头训 B0）：[`docs/design/2026-07-29-aerial-nav-wam-redesign.md`](docs/design/2026-07-29-aerial-nav-wam-redesign.md)
- v1 交接：[`docs/handover/2026-07-29-aerial-b0-b1-orchestration-handover.md`](docs/handover/2026-07-29-aerial-b0-b1-orchestration-handover.md)
