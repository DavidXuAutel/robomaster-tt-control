# 手势控制模块交接说明（2026-07-20）

> 面向：大众。
> 目的：说明"纯视觉手势控制"这块我们做了什么、代码怎么组织、你怎么接手运行与二次开发。

## 0. 一句话结论

手势控制的**权威实现在云端 git 分支 `ouxuedong/auto-fly`**（在另一台电脑上开发，2026-07-19 提交 4 个 commit）。本次已把这块代码**合并进 `main`**，并与本地在做的**深度避障 / 离线仿真**工作整合到同一份代码里，回归测试 **44 项全绿**。

## 1. 这次动了什么（合并整理）

- 从 `origin/ouxuedong/auto-fly` **快进合并**了手势控制到 `main`（4 个提交，见下）。
- 之前本地曾有过**另一套**手势原型（`tt_control/gesture.py`，MediaPipe Hands 关键点+规则判断）。按"手势以云端 git 为准"，**已删除该旧实现**及其演示脚本（`gesture_dryrun.py`、`sim_gesture_flight.py`）和旧模型 `hand_landmarker.task`，统一走 git 版。
- 手势（git 版）与本地的**深度避障（`V` 键）/ 离线仿真（`--sim`）** 现共存于同一 `app.py`，互不冲突：
  - 手势 → 通过"推理事件流"触发起降；
  - 避障 → `V` 键 OFF→ARMED→ON 的自动杆量；
  - 仿真 → `--sim` 用虚拟机代替真机离线跑。
- 回归：`pytest tests/ -q` → **44 passed**（含手势 4 个测试 + 避障/仿真/控制/轨迹等）。

合并进来的 4 个手势提交：

| commit | 说明 |
|---|---|
| `94f06a6` | Add personalized Tello gesture control baseline（个性化手势基线） |
| `aa5cafa` | Add armed real-flight gesture test workflow（需 ARM 的真机测试流程） |
| `b6e7477` | Fix flight test to hover at native takeoff height（悬停在原生起飞高度） |
| `0d4542a` | Preserve gesture motion across brief hand dropouts（短暂丢手时保持动作连续） |

## 2. 手势方案怎么工作

**手势语义**（纯视觉，不用麦克风/音频）：

- **起飞**：保持**张开手掌**，然后在约 0.5–1.2 秒内**明显向上抬**。
- **降落**：拇指与中指先接触、再快速分离（**响指动作**），同时中指回落掌心。

**识别管线**：`Tello 前视相机帧 → MediaPipe Gesture Recognizer（21 点手部关键点）→ 动态轨迹特征 + DTW 模板匹配 / 阈值判断 → InferenceEvent(takeoff|land)`。App 在**单独一层**做飞行安全门控后才真正发指令。

**三种使用模式**：

1. **正常手势飞行**：识别到手势即执行起降（有安全门控）。
2. **dry-run 验收**（`--gesture-dry-run`）：只在界面显示识别结果、**不发飞行指令**，用于现场标定。起飞/降落各作为一项验收，两项都识别到显示 `GESTURE TEST PASSED`。
3. **真机 flight-test**（`--gesture-flight-test`）：需手动 `TEST ARM` 的严格真机流程，逐条写 JSONL 日志到 `logs/gesture_flight_tests/`。

**个人手势录制**：连接后在右侧 Control 面板 `TRAIN TAKEOFF / TRAIN LAND / TRAIN NONE` 各录 10 次，`SAVE PROFILE` 用归一化 21 点轨迹 + DTW 自动算阈值并立即启用；profile 存 `gesture_profiles/profile_*.json`（该目录不入库）。

## 3. 关键文件

```
tt_control/
  gesture_control.py     # 手势后端主体（★核心）
    - MediaPipeGestureBackend(InferenceBackend)  # 推理后端：infer / drain_events / status_text
    - GestureSequenceDetector + GestureThresholds # 起飞/响指的动态判断 + 可调阈值
    - GuidedTrainingSession                       # 面板引导式录制个人手势
  gesture_profile.py     # 少样本动态手势模板：关键点归一化 / DTW / 自动阈值 / 本地 profile 读写
  flight_test.py         # FlightTestRecorder：真机测试逐事件 JSONL 记录
  inference.py           # InferenceBackend 协议 + InferenceEvent；create_backend("gestures") 注册点
  app.py                 # 统一界面：_handle_inference_event 做安全门控 + ARM/测试状态机 + 训练按钮
  assets/gesture_recognizer.task   # MediaPipe 手势识别模型（8MB，随仓库分发）
tests/
  test_gesture_control.py / test_gesture_profile.py / test_app_gesture_events.py / test_flight_test.py
```

## 4. 怎么运行（接手速查）

```bash
# 环境
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # 含 mediapipe/opencv 等

# 本机摄像头离线自测手势识别（无需飞机）
python -m tt_control.gesture_control

# 一键组网 + 默认手势（macOS，见 README）
python auto_fly.py

# 手动启动 + 手势（指定飞机/本机 IP）
python main.py --tello-ip 192.168.0.100 --local-ip 192.168.0.102 --inference gestures -v

# 现场标定：dry-run（只显示不飞）
python auto_fly.py --gesture-dry-run

# 真机严格测试：需手动 TEST ARM
python auto_fly.py --gesture-flight-test

# 回归测试
python -m pytest tests/ -q
```

## 5. 二次开发 / 接入点

- **调手势灵敏度**：改 `gesture_control.py` 里的 `GestureThresholds`（张掌分数、上抬时长/位移、响指捏合/分离比例、冷却时间等）。
- **接你自己的模型**：实现 `InferenceBackend`（`tt_control/inference.py`），在 `infer(frame)` 里推理，并通过 `drain_events()` 吐出 `InferenceEvent(kind="takeoff"|"land", confidence, detail)`；在 `create_backend()` 注册一个名字即可用 `--inference 你的名字` 启用。**所有飞行安全判断都集中在 `App._handle_inference_event`**（电量、是否已离地、冷却、测试状态机），后端只负责"识别 + 报事件"，不碰飞控。
- **安全门控**：起飞要求已连接 + 电量≥30% + 在地面；降落要求检测到已离地；单次触发后有冷却；键盘 `L` / 界面 `LAND` / `Esc` 始终是人工备份。

## 6. 已知事项

- **依赖**：手势需要 `mediapipe`。若与其他环境（如避障/仿真的 numpy 版本）冲突，手势可放在独立 venv 跑；缺依赖时会给可读报错而非崩溃。
- **模型文件**：`tt_control/assets/gesture_recognizer.task`（8MB）随仓库分发，缺失时按提示重新下载。
- **个人 profile**：`gesture_profiles/` 默认 `.gitignore`，不入库、不覆盖旧记录。

## 7. 当前 git 状态（重要）

- `main` 已包含手势控制（HEAD = `0d4542a`），**这部分可直接用/可 push**。
- 本地工作区另有**深度避障 / 离线仿真**那批改动，目前仍为**未提交状态**（按约定暂不 commit），与手势代码已整合共存、测试通过。后续如需，可单独整理成提交。
