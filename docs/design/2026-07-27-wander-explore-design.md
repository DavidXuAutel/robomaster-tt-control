# 随机漫游探索（Wander）模块设计方案

> **⚠️ 文件所有权：本文件只允许 Claude (Claude Code) 修改。**
> DeepSeek / Grok / Cursor / 其他 Agent：**只读**。本文件是你们的实现规格书，
> 照做，不要改。发现规格有问题 → 在 `docs/dev-notes/` 新建笔记提出，由主人裁决。
> 本文件设为 chmod 444，有意为之，禁止 chmod 回去。

- 日期：2026-07-27
- 状态：设计定稿，待实现（实现方：DeepSeek + Grok，协作规则见 §8）
- 对应 WAM 场景规划：`docs/design/2026-07-25-wam-training-scenarios.md` 的 **G1 随机探索**
- 前置必读：`docs/design/2026-07-27-orbit-control-principles.md`（尤其「历史坑」9 条，
  全部适用于本模块）

---

## 0. 一句话定义

无人机在封闭室内**只向前飞**、高度受限地随机漫游：前方出现障碍 → 随机（但偏向开阔侧）
转头 → 确认前方开阔 → 继续前飞。**最终产品不是飞行本身，是 episode 数据**
（RGB 视频 + 深度 + 动作 + 遥测 + 决策状态，逐帧对齐），供大众 WAM 世界模型训练。

因此所有设计取舍的第一优先级排序是：**不炸机 > 数据质量（对齐/完整/可复现）>
数据多样性 > 单位电量的采集时长 > 轨迹美观**。飞得"聪明"不加分，数据坏了全部白飞。

---

## 1. 复用边界：哪些已经有了，禁止重写

| 已有组件 | 位置 | 本模块如何用 |
|---|---|---|
| 深度管线（~5Hz nearness 96×128，值越大越近） | `depth_backend.py` / `async_infer.py` | 原样使用，不改 |
| 三分区近度 `zone_nearness()` → (left, mid, right) | `tt_control/avoidance.py` AvoidanceController | 原样调用 |
| FSM 全局安全（电量/高度/深度过期/AUTO 超时/人工接管） | `tt_control/avoidance_fsm.py` `_check_safety` + step() 前置检查 | 原样继承，wander 作为 FSM 的一种模式挂进去 |
| DANGER 去抖模式（先零杆 hold 再 abort） | `_finish_orbit_decision` 的 `orbit_danger_hold_s` 模式 | **抄这个模式**，不要发明新的 |
| Episode 录制（video.mp4 + frames.csv 行号=视频帧号 + depth npy + meta.json） | `tt_control/episode_recorder.py` | 原样使用；只允许**追加**字段（§6），禁止改动同步核心 |
| 仿真 | `tt_control/sim_drone.py` | 单测/仿真验证阶梯用（§9） |
| 配置体系（configs/default.json > dataclass 默认值） | `tt_control/config.py` / `flight_config.py` | 新增 `wander` 节，**所有阈值只允许出现在这里**（orbit 坑 #9） |

**明确禁止改动**：OrbitController 及 orbit 相关 FSM 路径、降落重试路径
（`app.py` `_run_flight_cmd`，orbit 原则文档守则 #7）、EpisodeRecorder 的
帧↔行同步逻辑、任何 Claude 所有权文档。

---

## 2. 行为设计：WanderPolicy 状态机

新增文件 `tt_control/wander.py`（`WanderPolicy` + `WanderParams` + `WanderDecision`），
接入方式仿照 orbit：`FsmParams` 加 `wander_mode: bool`，AvoidanceFSM 在
`wander_mode=True` 时把 APPROACH 分支交给 WanderPolicy。**wander_mode 与
orbit_mode 互斥**，两者同时为 True 时启动即报错退出（不要静默选一个）。

### 2.1 状态与转移

```
WANDER_CRUISE   前飞（pitch 分段随机），每帧检查前方
   │ mid > turn_thresh（连续 turn_confirm_frames 个**深度帧**）
   ▼
WANDER_TURN     零 pitch，原地 yaw 转向（方向/角度见 2.2），用遥测 yaw 差判转够
   │ 转到目标角度
   ▼
WANDER_VERIFY   零杆悬停，等 mid < clear_thresh 连续 verify_frames 个**深度帧**
   │ 通过 → 回 WANDER_CRUISE（开新的随机前飞段）
   │ 不通过（verify_timeout_s 内 mid 始终高）→ 回 WANDER_TURN 追加转角（同向）
   ▼
（任意状态）max(zones) > danger_thresh
   → DANGER_HOLD：零杆 + 去抖（danger_hold_s，抄 orbit 模式）
   → 仍危险 → WANDER_RETREAT：pitch=-retreat_pitch 后退 retreat_s 秒 → WANDER_TURN
   → 后退后仍危险 → abort "wander_danger"（悬停解除 AUTO，episode 标注 outcome）
```

要点：

- **转向必须原地转（yaw），禁止盲目横移**。深度只有前向：横移/后退方向没有感知。
  现有 AVOID_TURN 的横移是针对"绕过单个障碍"的短程受控行为，经过真机验证，
  但漫游是长时行为，盲移风险累积，不允许。唯一允许的盲动作是
  WANDER_RETREAT 的**短后退**（≤ retreat_s，默认 0.7s）：刚飞过来的正后方
  大概率是自由空间，这是保命动作，对应 orbit 安全不变量 #2 的精神。
- **VERIFY 是必须的独立状态**。转完不能立刻前冲：深度 ~5Hz + 端到端 ~0.5s 延迟，
  转身瞬间拿到的 nearness 还是转身前的画面。必须悬停等**新鲜深度帧**确认。
  帧数按深度帧计（orbit 坑 #8）：verify_frames=3 ≈ 0.6s，不是 3 个控制帧。
  实现上用「nearness 对象的 ts 变化次数」计帧，不要数控制循环迭代。
- **段式随机而非逐帧随机**。WAM 场景文档 G1 的伪代码（逐帧 random.choice）
  **不要照抄**——逐帧随机产生的是抖动噪声，不是轨迹多样性，且深度 5Hz 跟不上
  逐帧变向。正确做法：CRUISE 进入时抽一次「段参数」（pitch 大小、段时长、
  可选高度微调），整段保持；段结束（时长到 or 遇障）再抽下一段。

### 2.2 转向策略（随机性的核心）

- **方向**：加权随机，偏向开阔侧。`p(左) = softmax([-left, -right] / temp)` 或
  简化为：开阔侧概率 turn_open_bias（默认 0.8），另一侧 0.2。
  不能纯贪心（永远选最开阔 → 沿墙打转，数据单一）；也不能纯随机
  （50% 概率转向更堵的一侧 → VERIFY 反复失败浪费电）。
- **角度**：uniform(turn_min_deg, turn_max_deg)，默认 [50°, 130°]。
  下限 50°：太小的转角在墙前会反复触发（还是那面墙）；上限 130°：
  不默认掉头，掉头容易在两面墙之间打乒乓。
- **转角测量用遥测 yaw 差**（处理 ±180 环绕，照抄 `_step_recover_heading` 的
  `(target - current + 180) % 360 - 180`）。Tello yaw 会漂移，但**测相对转角**
  在几秒尺度内足够准；禁止为此引入视觉里程计等新依赖。
  yaw 遥测不可用（读不到）→ 退化为定时转（yaw_speed × 时间估角度），并在
  sub_state 里标注 `yaw_dead_reckon`，episode meta 记一笔。
- **防打转（anti-corner）**：滑动窗口记录最近 corner_window_s（默认 20s）内的
  遇障转向次数。≥ corner_max_turns（默认 4）次 → 判定困在角落/家具间，
  执行一次「全景选向」：原地慢转 360°（yaw_speed=30，约 12s），期间每个深度帧
  记录 (当前yaw, mid)，转完选 mid 最低的朝向转过去，清空窗口。
  全景选向后仍在 corner_window_s 内再触发 → abort "wander_cornered"。
- **无障碍也要随机转**：每个 CRUISE 段结束时（没遇障、时长自然到期），以
  free_turn_prob（默认 0.5）概率插入一次小角度转向 uniform(20°, 60°)。
  否则轨迹退化为「直飞到墙才转」，动作分布里 yaw 样本几乎只在障碍前出现，
  世界模型学到的是伪相关（"转弯=有障碍"）。

### 2.3 高度控制（回答主人要点 1）

- **硬顶**：复用 FSM 已有的 `max_height_cm` 全局安全检查，wander 模式下配
  350（3.5m，主人指定：上方无感知）。这是 abort 线，不是控制目标。
- **巡航带**：h_min_cm=80，h_max_cm=200。理由：<80cm 进入地效区且容易撞
  低矮家具（桌面高度正是深度视野的下边缘外）；>200cm 室内天花板/灯具风险
  且深度模型训练分布偏地面视角。**留意 3.5m 是红线不是工作区**。
- **高度微调也是数据**：CRUISE 段参数里以 alt_change_prob（默认 0.3）概率
  带一个 throttle=±alt_throttle（默认 25）的爬升/下降子段（1~2s），
  但只在 mid < clear_thresh（前方开阔）时执行——爬升时前视野上移，
  正下前方是盲区。高度用遥测 h 闭环钳在巡航带内：超带即把 throttle 归零。
  h 读数异常（连续 5 帧读不到数值）→ throttle 永远 0，只做平面漫游。
- **注意 `tof` 与 `h` 的区别**：h 是气压/融合高度（相对起飞点），tof 是下视
  测距（相对正下方表面）。飞过桌子上方时 tof 突降但 h 不变——**巡航带判断
  用 h**，tof 只作记录字段不参与控制（否则过桌沿会误判骤降而猛拉油门）。

### 2.4 只向前飞（回答主人要点 2）

正确，且要写成硬约束：**除 WANDER_RETREAT 外，pitch ≥ 0、roll ≡ 0**。
单测必须覆盖：遍历 WanderPolicy 所有状态输出，断言 roll 恒 0、
pitch < 0 仅出现在 RETREAT 且时长受限。这是本模块的安全不变量 #1。

### 2.5 随机性与可复现（数据模块的独有要求）

- **单一 RNG**：WanderPolicy 构造时接收 `seed: int`，内部只用
  `random.Random(seed)`，**全模块禁止裸用 `random.xxx` 全局函数**。
- seed 默认取 episode 启动时间戳，**必须写进 episode meta.json**
  （`recorder.note(wander_seed=..., wander_params={...})`），同时写入当时
  生效的全部 WanderParams。目的：任何一段数据都能回答"当时策略是什么"，
  且仿真里能复现同一决策序列做回归。
- 每次抽样决策（段参数、转向、全景选向结果）作为**决策事件**记录（§6）。

---

## 3. 与全局安全的对接（继承 + 新增）

全部继承 FSM 现有检查：低电量、高度异常、深度过期（depth_stale_s）、
AUTO 全局超时（max_auto_engaged_s，**注意 orbit 坑 #9：app.py 的 AutoWatchdog
读同一配置，不要再造一个**）、人工接管即刻复位。

wander 特有：

1. **danger 处理三段式**（hold → retreat → abort）如 §2.1。danger_hold_s 默认
   0.4s，与 orbit 相同——单深度帧证据不足以行动（orbit 坑 #5）。
2. **电量策略**：漫游是长时任务，min_battery_pct 建议 15（比 orbit 的 10 高）：
   abort 后主人还要手动飞回来降落，留余量。**不做自动返航**（无定位，做不了，
   WAM 文档 F2 是独立课题，不要顺手实现）。
3. **场地是安全设计的一部分（运维要求，写进 checklist）**：Tello 无 GPS、
   本项目无定位，软件**无法**做地理围栏。漫游只允许在物理封闭的室内进行
   （门窗关闭；镜子/大面积玻璃/纯白无纹理墙用胶带贴条或遮挡——单目相对深度
   对这三类表面会给出错误的"开阔"）。WiFi 断联依赖 Tello 固件自动降落，
   但**不许把它当保护机制用**：断联=丢一段数据+机身不受控，场地必须小到
   不可能飞出 WiFi 范围。
4. **abort 不是失败**：所有 abort episode 照常关闭落盘、outcome 如实标注
   （`set_outcome("aborted", reason)` 已支持）。边缘场景数据对世界模型同样
   有价值，禁止实现"失败就删数据"。

---

## 4. WanderParams 参数表（configs/default.json 新增 `wander` 节）

| 参数 | 默认值 | 说明 |
|---|---|---|
| seed | 0（0=用时间戳） | RNG 种子，写入 meta |
| cruise_pitch_min / max | 12 / 25 | 每段前飞 pitch 抽样区间。上限 25 起步，长飞验证后再考虑加 |
| segment_s_min / max | 3.0 / 8.0 | 前飞段时长抽样区间 |
| turn_thresh | 0.58 | mid 超此值触发转向。必须 < danger_thresh−0.15，给转向留反应距离 |
| turn_confirm_frames | 2 | 触发转向需连续**深度帧**数（按 ts 变化计） |
| clear_thresh | 0.40 | VERIFY 通过阈值 |
| verify_frames | 3 | VERIFY 需连续开阔**深度帧**数 |
| verify_timeout_s | 3.0 | VERIFY 超时 → 追加转角 |
| turn_min_deg / max_deg | 50 / 130 | 遇障转角抽样区间 |
| free_turn_prob | 0.5 | 无障碍段末随机转向概率（转角 20~60°） |
| turn_open_bias | 0.8 | 转向开阔侧的概率 |
| yaw_speed | 40 | 转向 yaw 杆量。太小转不动（<15 静摩擦区），太大过冲 |
| danger_thresh | 0.78 | 与 orbit 同值同义 |
| danger_hold_s | 0.4 | danger 去抖（抄 orbit 模式） |
| retreat_pitch / retreat_s | 15 / 0.7 | 保命后退杆量/时长（唯一允许的盲动作） |
| corner_window_s / corner_max_turns | 20.0 / 4 | 防打转窗口 |
| h_min_cm / h_max_cm | 80 / 200 | 高度巡航带（红线 350 在 fsm.max_height_cm） |
| alt_change_prob / alt_throttle | 0.3 / 25 | 高度微调子段概率/杆量 |

调参守则同 orbit 文档 §5：先看日志再改、一次一个、真机验证。

---

## 5. FSM 接入点（实现指引）

- `FsmParams` 增加 `wander_mode: bool = False` 及上表中属于 FSM 层的项
  （或整个 WanderParams 独立成节由 config.py 装配，仿照 OrbitParams 的装配路径）。
- `AvoidanceFSM.__init__` 增加可选 `wander: Optional[WanderPolicy]`。
- `step()` 的 APPROACH 分支开头：`wander_mode=True` 时直接
  `return self._step_wander(...)`，**不进入** orbit/AVOID_TURN 逻辑。
  死胡同检测（danger rising）在 wander 下跳过——漫游靠近障碍是常态，
  wander 自己的三段式 danger 处理接管（同 orbit 跳过该检测的理由）。
- WanderPolicy 是**纯函数式内核**：`decide(nearness, telemetry, now) -> WanderDecision`，
  不碰 socket、不碰线程、不 sleep。所有对时间的依赖通过 now 参数注入
  （orbit 同款约定，否则没法单测/仿真）。`now=0.0` 必须合法
  （orbit 曾因 0.0 哨兵炸过，见 `test_fsm_now_zero_does_not_clear_orbit_latch`）。
- 与 `--record` 的关系：wander 起飞即录（现有 EpisodeRecorder 生命周期不变），
  wander 模式下 meta_base 加 `scenario: "wander_explore"`。

---

## 6. 数据记录增强（只允许追加，不许改同步核心）

EpisodeRecorder 现状已满足主体需求（视频帧↔CSV 行严格 1:1、深度按 ts 去重、
depth_age_ms 诚实核算）。wander 需要的增量：

1. **frames.csv 追加两列**（追加到 FRAME_FIELDS 尾部，不许插中间——下游可能
   按列序读）：
   - `wander_state`：WANDER_CRUISE / WANDER_TURN / ... （空串=非 wander 模式）
   - `wander_event`：本帧发生的决策事件短码，无事件为空。事件码：
     `SEG(pitch,dur)` 新前飞段、`TURN(dir,deg,reason)` reason∈{obstacle,free,retry}、
     `PANO(chosen_yaw)` 全景选向、`RETREAT`、`DANGER_HOLD`
2. **meta.json notes**：wander_seed、生效 WanderParams 全量、
   turns_total、panos_total、retreats_total、corner_aborts。
3. **世界模型关心动作分布**：数据 QA 工具（§8 分工 B）必须输出每个 episode 的
   动作直方图（pitch/yaw/throttle 各自的取值分布）与 wander_event 统计。
   一批数据里若 yaw 样本 90% 都伴随高 mid（只在障碍前转弯），说明
   free_turn_prob 太低，这是数据质量问题，不是飞行问题。
4. **录制不许阻塞控制环**：现状 capture() 在控制线程同步执行（10Hz 限流 +
   flush），真机已验证可接受，**保持现状**；如果实现方发现耗时超预算想改成
   异步队列——不许顺手改，先提 dev-note（改错线程边界会破坏帧↔行 1:1）。

---

## 7. 坑清单（实现前逐条读，验收时逐条对照）

**继承 orbit 原则文档的 9 条历史坑**，其中直接适用的映射：

| orbit 坑 | 在 wander 里的形态 |
|---|---|
| #2 bang-bang 振荡 | 转向-验证-前飞若无 VERIFY 状态和确认帧数，会在墙前抽搐 |
| #4 高增益+延迟必振荡 | yaw_speed 过大 + 用控制帧判"转够了" → 过冲来回找角度 |
| #5 单帧证据不足 | turn/danger/verify 全部要求连续**深度帧**确认 |
| #8 帧率认知 | 所有 frames 参数按 5Hz 深度帧计，用 ts 变化计数，不数控制循环 |
| #9 阈值单一来源 | wander 全部阈值只在 configs/default.json；发现代码里出现字面量阈值=评审直接打回 |

**wander 新增坑（设计阶段预判，实现方注意）**：

1. **单目相对深度是"相对"的**：nearness 每帧独立归一化。大平面（白墙）占满
   视野时整图变平，mid 可能不升反降——纯软件无法根治，靠场地布置（§3.3）+
   turn_thresh 留反应距离 + danger 三段式兜底。**禁止**为此给深度管线打补丁。
2. **转身瞬间的旧深度帧**：转向刚结束时拿到的 nearness 是 ~0.5s 前的画面
   （转身前的方向）。VERIFY 必须等 ts **新于**转向结束时刻的帧才开始计数，
   否则会用"背后那面墙已开阔"的旧帧放行前冲。**实现时给 WanderDecision 记
   turn_end_ts，VERIFY 只认 depth_ts > turn_end_ts 的帧。**
3. **yaw 遥测漂移与环绕**：±180 环绕必须用模运算求差；长时间累计漂移没关系
   （只用相对角），但**不要**实现"回到初始朝向"之类依赖绝对 yaw 的功能。
4. **段随机 ≠ 帧随机**：照抄 WAM 文档 G1 伪代码（逐帧 random.choice）= 直接打回。
5. **throttle 与前视盲区**：爬升时下前方看不见。高度子段只许在前方开阔时做，
   且 throttle 与大 pitch 不同段（子段内 pitch 减半）。
6. **室内气流**：Tello 在离墙 <50cm 时会被自身涡流推向墙（贴墙效应）。
   turn_thresh=0.58 的经验距离（约 1~1.5m 触发）已含此余量，不要为"更贴近
   障碍拍摄"调高它。
7. **电量尾段行为劣化**：<30% 时 Tello 机动性下降、h 漂移变大。QA 工具按
   bat 分段统计数据质量，训练侧可按需过滤，采集侧不用特殊处理（如实记录即可）。
8. **测试环境**：跑测试必须 `.venv/bin/python`（3.11）；系统 python3 是 3.9，
   `int | None` 语法直接炸。已知无关预存失败：
   `test_main_depth_guard.py::test_depth_inference_without_service_returns_2`。

---

## 8. DeepSeek / Grok 协作与相互监督规则

### 8.1 分工（按模块边界切，不按文件行数切）

- **实现方 A（DeepSeek）——控制内核**：`tt_control/wander.py`（WanderPolicy /
  WanderParams / WanderDecision）+ FSM 接入（`avoidance_fsm.py` 的
  wander 分支）+ config 装配 + `configs/default.json` wander 节。
- **实现方 B（Grok）——测试与数据 QA**：`tests/test_wander.py`（§9.1 用例清单）+
  EpisodeRecorder 追加字段（§6.1，改动面小且独立）+ 新工具
  `episode_check.py`（数据 QA：帧数一致性、depth_age 分布、动作直方图、
  wander_event 统计、§9.3 验收指标自动核算）。

理由：A 与 B 几乎无文件交集（只在 WanderDecision 的字段定义上有接口耦合，
先由 A 提交 dataclass 定义，B 依赖之），可并行，且 B 的测试天然是对 A 的监督。

### 8.2 相互监督（硬性流程）

1. **接口先行**：A 先只提交 `WanderDecision` / `WanderParams` dataclass +
   docstring（不含实现），B 据此写测试骨架。接口一旦双方开工，改字段需
   双方确认并在 commit message 注明。
2. **交叉评审**：A 的每次提交由 B 对照本文档 §2/§3/§7 逐条审（重点：roll≡0
   不变量、深度帧计数、阈值无字面量）；B 的测试由 A 审"测的是不是规格说的
   行为"（防止测试写成实现的镜像——断言抄实现代码 = 无效测试）。
   评审结论写在 `docs/dev-notes/` 下，格式：逐条 checklist + 结论。
3. **分歧处理**：两方对规格理解不一致时，**以本文档字面为准**；文档没写到的，
   停下来提 dev-note 等主人/Claude 裁决，**禁止**任何一方"顺手定了"。
4. **共同红线**（违反即整个提交打回）：
   - 不改本文档、orbit 原则文档、orbit 控制代码、降落路径、录制器同步核心
   - 不新增任何硬编码安全阈值（orbit 坑 #9）
   - 不引入新第三方依赖（现有 requirements 已够）
   - 每次提交前 `.venv/bin/python -m pytest tests/ -q` 全绿
     （除已知预存失败）；真机暴露的 bug 修复必须带回归测试

### 8.3 完成定义（DoD）

代码完成 ≠ 模块完成。模块完成 = §9 阶梯全部通过 + episode_check.py 对一次
≥3 分钟真机漫游数据的 QA 报告全部指标达标 + 本文档由 Claude 补记「已验证基线」节。

---

## 9. 测试与验收阶梯（依次通过，不许跳级）

### 9.1 单元测试（合成近度图，仿 test_orbit.py 的 _grid/_chair 手法）

必须覆盖（B 实现，允许增补，不允许删减）：

1. 前方开阔 → CRUISE，pitch 在 [cruise_pitch_min, max] 内，roll==0
2. **全状态遍历 roll≡0；pitch<0 仅在 RETREAT 且 ≤ retreat_s**（安全不变量）
3. mid 单帧尖刺不触发转向（turn_confirm_frames 按深度帧 ts 计数）
4. 触发转向后 pitch==0，yaw 符号与开阔侧一致（bias 用固定 seed 断言）
5. VERIFY 拒绝 turn_end_ts 之前的旧深度帧（坑 #2 的直接回归测试）
6. VERIFY 超时 → 同向追加转角
7. danger 三段式：单帧尖刺零杆不 abort；持续 → RETREAT → 仍危险 → abort
8. 防打转：窗口内第 4 次转向触发全景选向；选向后再触发 → abort "wander_cornered"
9. 同一 seed 两次运行，决策序列完全一致（喂相同帧序列，比对 WanderDecision 流）
10. h 超巡航带 → throttle 归零；h 读数缺失 → throttle 恒 0
11. now=0.0 起步不误触发任何超时/清零（orbit 同款回归）
12. wander_mode 与 orbit_mode 同时 True → 构造即抛错

### 9.2 仿真（sim_drone.py）

固定 seed 跑 10 分钟仿真漫游：无 abort（或仅合理 abort）、转向次数 > 0、
无「连续 30s 杆量全零且非 DANGER/VERIFY」（卡死检测）。作为 CI 可跑的冒烟。

### 9.3 真机阶梯与验收指标

1. **地面预演**：不起飞，手持无人机对着障碍走位，观察 HUD 的 wander 状态流转
   与日志（AUTO dbg 行需包含 wander_state 与事件码）。
2. **首飞（保守参数）**：cruise_pitch 上限压到 15，2 分钟，人手悬在 L 键。
   目标：状态流转正确、无撞击、abort 原因可解释。
3. **验收飞行**：默认参数，单块电池连续漫游 **≥5 分钟**，随后 episode_check.py
   核算，指标全部达标：
   - 撞击次数 = 0（人工观察）
   - 遇障转向 ≥ 8 次，free turn ≥ 3 次，且 TURN 方向不全同侧
   - 帧完整：frames.csv 行数 == video 帧数；depth_age_ms P95 < 600ms
   - 动作多样性：pitch/yaw 直方图各 ≥ 4 个非空 bin；yaw 非零样本中
     伴随低 mid（<clear_thresh）的占比 ≥ 20%（证明 free turn 生效）
   - abort ≤ 1 次且原因合理（电量/主动接管不算失败）
4. 验收通过后：由 Claude 在本文档补「已验证基线」节（同 orbit 文档 §6 格式），
   之后参数变更按 orbit 守则流程走。

---

## 10. 明确不做（划清边界，防实现方发散）

- 不做定位/建图/返航/轨迹规划（无传感器基础，全是坑）
- 不做多机、不做动态目标交互（独立课题）
- 不做深度模型侧的任何改动
- 不做"更聪明的探索策略"（覆盖率驱动、好奇心驱动等）——先把朴素随机漫游
  的数据管线跑通验收，策略升级是下一个迭代的事
