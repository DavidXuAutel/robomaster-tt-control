# RoboMaster TT 单机避障"做扎实" + 数据交付 — 设计方案

| | |
|---|---|
| **日期** | 2026-07-21 |
| **状态** | **待确认**(先议后定,确认后再进入实现) |
| **范围** | 单机避障能力做扎实 + 面向大众 WAM 的数据链路与交付物定义 |
| **关联** | [`./2026-07-17-tt-visual-avoidance-design.md`](./2026-07-17-tt-visual-avoidance-design.md) · [`../references/2026-07-21-avoidance-model-capability.html`](../references/2026-07-21-avoidance-model-capability.html) · [`../references/2026-07-20-scoutxwam-world-model-analysis.md`](../references/2026-07-20-scoutxwam-world-model-analysis.md) |

> 本文是 2026-07-21 会话的结论沉淀。上游判断:无人机是大众"遥操作数据飞轮 + 世界模型(WAM)"方法论里**第二个、更廉价更安全的具身载体**;避障是**兜底/安全网**,让无人机能不炸机地自主飞、把训练级数据采回来;WAM 才是要培养的"大脑"。

---

## ⭐ 本轮测试目标(实测验收锚点)

> 一句话:**一次测试,同时达成"验证一件事、产出两类数据"。** 可概括为三个目标 =「1 个验证目标 + 2 类数据产出」。

### 目标一 · 验证:摸清避障能力边界

验证「**单目相机 + Depth Anything V2 Small + 反应式控制律**」在真机上的**能力与失效边界**,把文档里"纸面写的边界"变成"实飞证过的边界"。

- 三种行为真机成立:**接近减速 / 朝空旷侧拐弯(TURN)/ 被围悬停(BLOCKED)**;
- **失败降级链**逐条触发:看门狗(挂载超时、感知失联)、指令非 ok、键盘瞬间夺权;
- **盲区确认**:玻璃 / 白墙 / 反光 / 细线 确实躲不了,写进 SOP;
- **参数整定 + 重复性**:真实延迟(~9fps)下调稳杆量阈值,同场景 ≥5 次行为一致。
- **成功标准**:每种行为真机可复现、降级链可触发、边界如实回填能力报告与 SOP。

### 目标二 · 产出:两类数据

| | 2a 仿真轨迹校正数据 | 2b 自由避障飞行视频数据 |
|---|---|---|
| **来自哪种飞行** | **校正飞行**(同向 pad、直线、**不转向**) | **避障飞行**(自主避障、含转向,pad 可选) |
| **产出** | 真机位姿 + 动作(`frames.csv`) | `video.mp4` + `frames.csv`(动作+本体状态) |
| **给谁用** | `sim_match_report.py` 出可信度报告(MAE/RMSE/终点误差/叠图) | 喂世界模型做**具身特定后训练** |
| **作用** | 证明"**仿真是真机的可信替身**"(对准/校正) | 让 WAM 学**无人机的因果动力学** |
| **成功标准** | 轨迹匹配误差量化、报告成文 | episode 完整、同步、可仿真复现 |

### 三者关系(现场执行须知)

- **目标一 与 2b 在同一批避障飞行里同时完成**——飞的时候既验证避障、又把视频数据采回来(一开 `--record` 即可);
- **2a 是单独的几次受控校正飞行**——因为避障要转向、会破坏 Mission Pad 轨迹拼接,校正必须用"不转向 + 同向 pad"单独飞。

---

## 0. 一句话目标

把单架 Tello 做成一个**可信赖的具身数据节点**:能安全自主飞(避障兜底)、每次飞行同步产出**训练级的成对数据(观测+动作+状态)**、能证明"仿真是真机的可信替身"、并预留好把 WAM 换进来当大脑的接口。

---

## 1. 目标定义(可检验的成功标准)

"做扎实"= 从"demo 能跑"升级到"可信赖",四条硬标准:

| # | 标准 | 验收方式 |
|---|---|---|
| G1 | **可靠** | 同一避障场景真机飞 ≥5 次,行为一致(减速/拐开/悬停),无失控 |
| G2 | **可复现** | 每次飞行的观测+动作+状态全量落盘,能在仿真里回放,轨迹误差有量化指标 |
| G3 | **边界清楚** | 避障能力/失效场景写成文档(已产出 HTML),SOP 里写死"什么能飞什么不能飞" |
| G4 | **可交付** | 产出的数据是训练级(同步、干净、有 episode 结构与元数据),大众能直接接入 WAM 管线 |

---

## 2. 交付给大众的产出物(Deliverables)

### D1 · 避障能力与场景边界说明 ✅(已产出)

`docs/references/2026-07-21-avoidance-model-capability.html`。核心边界:

| 能做 | 做不到 / 会失效 |
|---|---|
| 躲有纹理的大块不透明障碍;接近减速;朝空旷侧锁向拐开;被围悬停 | 测绝对距离(相对深度无尺度);玻璃/白墙/反光/电线;飞向目标点/认路;穿窄缝;预测动态障碍;识别物体语义 |

**一句话交接给大众:** 这套避障是"反应式安全网,只保证别撞上有纹理的大块障碍",导航/去目标/精细穿越是 WAM 的职责,不在避障范围内。

### D2 · 训练级数据集(格式规范见 §4)

**它对大众的价值(按用途分,别混):**

| 数据用途 | "随便飞/避障自动飞"的数据 | 说明 |
|---|---|---|
| **训世界模型**(学空间几何/动力学/深度,= ScoutXWAM 的 DROID-1K Spatial 那层) | ✅ **有效** | 世界模型要的是"给动作,世界怎么变"的多样转移样本,动作谁产的不重要,多样覆盖反而好 |
| **仿真校正/可信度锚定**(见 D3) | ✅ **有效** | 真机轨迹是校准仿真的"真值锚" |
| **训动作策略/VLA/模仿学习**(学怎么完成任务) | ❌ **价值低** | 需要有目的的人类遥操作演示,避障自动输出学不到"意图"。若要这类数据,须改为遥操作采集 |

> **对大众 WAM 的直接价值点:** ①提供真机 RGB(-D)+动作+位姿的成对数据,喂世界模型学空间智能;②提供"真实 o"作为校准仿真想象 ô 的锚(避免飞轮自我欺骗)。

### D3 · 仿真可信度证据(Tier 1 匹配验证)

真机飞一条 → 仿真回放同一条 → **量化对比轨迹误差**(逐轴 MAE/RMSE、终点误差、路径长度比),出一份"仿真可信度报告"。这是让大众敢用"仿真增广/想象 rollout"的地基证据。

> 关键澄清:**"忠实回放"本身价值有限**(不产新数据、非闭环、无环境重建),它值钱只在"回放+对比=证明仿真可信"这一步。不加对比指标的纯播放,只有可视化价值。

### D4 · WAM 适配接口说明

`ExternalModelPolicy` 的输入/输出约定:输入 RGB(-D),输出速度场/动作 → 映射到 `RcAxes` 下发。RC 仲裁优先级:**键盘 > WAM > 避障 > 悬停**。让避障与 WAM 能干净切换。

---

## 3. 避障"做扎实"——开发待补清单

现状:真机已验证"起飞→前飞→遇墙停"(只覆盖了控制律三行为里的接近刹车/被围一种)。以下是补齐清单,标注**[代码]**(需开发)/ **[实测]**(现场执行)/ **[文档]**。

### 3.1 行为完整性(真机)
- **[实测]** 验证 **TURN 拐弯绕障**:挡板放正前方偏一侧留空档,验证"减速→朝更空侧锁向偏航拐开、方向不横跳"。**这是区别于"只会停"的核心能力,尚未验证。**
- **[实测]** 验证 **BLOCKED 被围悬停**:左中右都堵住,验证原地悬停不硬撞。
- **[代码]** 确保逐帧记录 `ctrl_state`(CRUISE/TURN_L/TURN_R/BLOCKED/MANUAL)进数据(见 §4),便于复盘。

### 3.2 失败降级链(逐条实测)
- **[实测]** 看门狗:单次挂载 >30s 自动悬停解除 AUTO(掐表)。
- **[实测]** 感知失联 >1.5s 自动悬停(故意断服务/挡镜头)。
- **[实测]** 位移/RC 指令非 ok → 悬停重发。
- **[实测]** 键盘 WASD/SPACE/ESC/L 在 AUTO ON 时瞬间夺权(逐键按)。
- **[代码]** 看门狗触发原因写入 episode 元数据(`meta.json` 的 `abort_reason`)。

### 3.3 边界与 SOP
- **[文档]** 感知盲区(玻璃/白墙/低纹理/逆光)写进飞行 SOP;挡板要求"大、不透明、有纹理"。
- **[代码]** HUD/日志暴露"低置信"信号(如深度帧过期、近度方差异常)。

### 3.4 参数整定与重复性
- **[实测]** 真实延迟(~9fps)下整定 `cruise_speed/yaw_speed/stop_thresh/clear_thresh`,宁慢勿快。
- **[实测]** 同场景 ≥5 次一致性;换距离/角度复验。
- **[代码]** 真机避障参数集中到配置(可 CLI 覆盖),飞行现场好调、可回填。

### 3.5 数据同步录制(把"安全飞"和"采数据"合并进同一次飞行)
- **[代码]** **核心开发项**:新增 episode 录制器,飞行中同步落盘 RGB + 深度网格 + 动作 + 状态 + 时间戳(见 §4)。当前只记位姿 CSV,缺 RGB/深度/动作的逐帧同步。

### 3.6 验收归档
- **[实测/文档]** 每次飞行记录:感知 RTT、生效阈值、最小净空、结论,回填 checklist。

**开发项汇总(需要写代码的):**
1. `tt_control/episode_recorder.py`(新)—— 逐帧同步录制器,挂进 `app.py` 更新循环。
2. `app.py` —— 接录制器;补 `ctrl_state`、看门狗原因、低置信信号的记录。
3. `sim_match_report.py`(新)—— Tier 1 仿真可信度匹配报告(D3)。
4. (P1,可延后)`tt_control/data_qa.py` —— 帧级/回合级数据质检。

---

## 4. 数据格式详规(训练级 episode 结构)—— 2026-07-21 更新为视频+索引

**设计原则:** 一次飞行 = 一个 episode 目录;**视频(MP4)是主观测**,`frames.csv` 是"帧↔动作↔状态"的逐帧索引兼轨迹;深度为辅助(96×128 近度)。对齐 ScoutXWAM 的"RGB(-D) + 动作 + 本体感知 + episode"约定,也对齐 DROID/LeRobot"MP4 + 表格"的通行做法。

> **为何 MP4 而非逐帧 JPEG(2026-07-21 定案):** 微调的是**世界模型**,视频经 VAE 编码后本就以序列消费;MP4 同等时长下体量远小于逐帧 JPEG(10min@10Hz:JPEG ~600MB vs H.264 ~100–200MB)、文件数少、跨机传输(DLP)友好。训练所需的"帧↔动作严格对齐"由 **视频第 N 帧 = `frames.csv` 第 N 行 + `t_mono_ms` 真实时刻** 保证,不靠 MP4 的名义帧率。编码 H.264/CRF18,尽量少损失训练信号。

```
logs/episodes/ep_<YYYYMMDD_HHMMSS>/
├── meta.json            # 回合级元数据(含 video 信息)
├── video.mp4            # H.264/CRF18;第 N 帧 <-> frames.csv 第 N 行(严格 1:1)
├── frames.csv           # 逐帧索引+轨迹(核心)
└── depth/000001.npy …   # 96×128 float16 近度网格(按 ts 去重,不重复落盘)
```

### 4.1 `frames.csv`(逐帧,`t_mono_ms` 为对齐基准)

| 字段 | 含义 |
|---|---|
| `frame_id` | 行号(=视频帧号+1) |
| `t_mono_ms` | 单调时钟毫秒(**对齐基准**,真实时刻;MP4 名义帧率仅供播放) |
| `video_frame` | 该行对应 video.mp4 的 0 基帧号 |
| `rgb_path` | 仅回退逐帧 JPEG 模式时有值 |
| `depth_path` / `has_depth` | 对齐的深度网格路径 / 是否有效深度 |
| `depth_rtt_ms` | 深度感知推理耗时 |
| `depth_age_ms` | **配对深度相对本帧的滞后**(同步透明度,筛查不同步坏帧用) |
| `pad_id` | 当前 Mission Pad 编号(丢垫为空,此行位姿不可比) |
| `pos_x/y/z_cm` | 相对 pad(或拼接全局系)位姿,**尺度锚** |
| `yaw_deg` / `height_cm` | 航向 / 下视高度 |
| `vgx/vgy/vgz` | 机体速度(SDK 上报) |
| `pitch_deg/roll_deg` | 姿态角 |
| `bat_pct` | 电量 |
| `act_roll/pitch/throttle/yaw` | **动作**:下发的 RC 杆量(**归一化 -100~100,非米/速度**,换算见下) |
| `ctrl_state` | 决策状态:CRUISE/TURN_L/TURN_R/BLOCKED/MANUAL/HOVER |
| `near_left/mid/right` | 左中右近度(复盘决策) |

**动作语义(训练必读):** `act_*` 是 Tello RC 归一化杆量。转真实速度用 SimDrone 同一套常数:
`v_forward = (act_pitch/100)·VMAX`、`v_right = (act_roll/100)·VMAX`(VMAX=0.6 m/s)、
`v_up = (act_throttle/100)·VZMAX`(VZMAX=0.5 m/s)、`yaw_rate = (act_yaw/100)·YAWRATE`(YAWRATE=70°/s)。

### 4.2 `meta.json`(回合级,实现产出)

```json
{
  "episode_id": "ep_20260721_153000",
  "drone": "192.168.10.1", "sim": false,
  "action_source": "avoidance|manual|mixed",   // 按逐帧决策自动推断,可被 meta_base 覆盖
  "camera": {"model": "tello-front-mono", "res": "960x720"},
  "depth": {"model": "DepthAnythingV2-Small", "semantic": "nearness(relative)"},
  "scale_anchor": "mission_pad",                 // 单目深度无米制,尺度靠 pad
  "video": {"file": "video.mp4", "codec": "h264", "crf": 18,
            "nominal_fps": 10, "frames": 0, "width": 960, "height": 720},
  "n_frames": 0, "n_depth_frames": 0, "depth_grid": [96,128],
  "frames_manual": 0, "frames_auto": 0,
  "battery_start_pct": 92, "battery_end_pct": 76,
  "outcome": "completed|aborted|land_failed", "abort_reason": null,
  "t_start_ms": 0, "t_end_ms": 0, "notes": {}
}
```

### 4.3 仿真校正与轨迹复现的关键输出(Q4)

要给大众的 **两类**产出,数据结构如下:

1. **可复现的原始 episode**(上面 §4.1/4.2)—— 复现所需最小充分集 =
   `t_mono_ms`(时序)+ `act_*`(动作)+ `pos_*/yaw`(真机真值)+ `video.mp4`(观测)。
   仅凭这几列即可在仿真里重放并叠图对比(见 `sim_replay.py` / `sim_match_report.py`)。

2. **可信度报告 `match_report.json` + `match_report.png`**(`sim_match_report.py` 产出):
   ```json
   {
     "episode": "ep_...", "frames_total": 0,
     "method": "action-driven sim replay vs recorded real pose",
     "sim_kinematics": {"VMAX":0.6,"VZMAX":0.5,"YAWRATE":70.0},
     "metrics": {
       "comparable_frames": 0,
       "mae_m": {"x":0,"y":0,"z":0}, "rmse_m": {"x":0,"y":0,"z":0},
       "yaw_mae_deg": 0, "endpoint_error_m": 0,
       "path_len_real_m": 0, "path_len_sim_m": 0, "path_len_ratio": 0
     }
   }
   ```
   PNG = 真机实线 vs 仿真(动作重放)虚线俯视叠图。**这就是"仿真是真机可信替身"的量化证据。**

### 4.4 同步保证与不同步筛查
- **保证:** 单调时钟 `t_mono_ms` 唯一基准;视频帧与 CSV 行严格 1:1;深度按 `ts` 去重。
- **筛查坏帧(P1 质检项):** `t_mono_ms` 间隔异常→丢帧/卡顿;`depth_age_ms` 过大→深度太旧不该配这帧;`pad_id<0`→位姿不可比(校正时剔除);`sim_match_report` 残差异常大→动作/位姿时序错位。

### 4.5 待与大众确认的格式点
- 深度:96×128 相对近度(现成,对校正够用)是否够,还是要**原始/高分辨率逆深度**喂世界模型(需感知服务加 raw 模式)?**世界模型若自带深度分支、只吃 RGB,则现状即可。**
- 是否严格对齐大众侧已有的 **DROID-1K Spatial / Scout-GELLO** 字段命名与目录(若有,直接对齐最省返工)。

---

## 5. 建议与优先级

**推荐主攻:P0 两根柱子在同一次飞行里合并完成** —— 补齐避障行为+失败链(§3.1/3.2 实测)的同时,上线 episode 录制器(§3.5 代码),一次飞行既验避障、又采训练数据。**投入产出比最高,直接兑现大众 (a)(b) 两条要求。**

**次优:D3 仿真可信度(Tier 1)** —— 投入小(一个 `sim_match_report.py` + 复用现有回放),却是大众世界模型最需要的"真机锚"地基。**你已表示这个建议可行,纳入方案。**

**明确不做(避免过度工程):**
- 不把反应式避障改造成带目标/地图的规划器(那是 WAM 的活)。
- 环境几何重建(real2sim)= 单目相对深度下难做准,**列为后续预研,不在本轮**,除非大众明确要。

---

## 6. 待大众对齐的问题(动手前建议先问)

1. **"仿真里建模"指哪层?** 轨迹回放+匹配验证(本方案 Tier 1,能做),还是环境几何重建(需单独立项)?
2. **数据 schema:** 有没有现成的 WAM/DROID 格式要我对齐?给定义我直接按它录。
3. **数据用途:** 这批数据训世界模型(→ 避障随便飞即可)还是训动作策略(→ 需改人遥操作)?
4. **WAM 接口:** 期望的输入(RGB-D?多视角?)输出(速度场?动作序列?)是什么?

---

## 7. 执行顺序与进度

```
第1步 [代码] episode_recorder.py + app.py 接入 + 单测   ✅ 已完成(--sim 端到端验证通过)
第2步 [实测] 真机:TURN 拐弯 + BLOCKED + 失败链逐条      ⏳ 待现场(需飞机+场地)
第3步 [实测] 参数整定 + ≥5 次重复性                       ⏳ 待现场
第4步 [代码] sim_match_report.py(Tier 1 可信度)          ✅ 已完成(误差指标 + 叠图 PNG)
第5步 [文档] 数据集说明 + SOP + 可信度报告交接大众         ⏳ 待第2/3步实飞数据回填
(P1,可并行/延后) data_qa.py 帧级+回合级质检              ⏳ 未开始
```

### 已落地(2026-07-21)

| 交付 | 文件 | 说明 |
|---|---|---|
| 录制器 | `tt_control/episode_recorder.py` | 同步落盘 **视频 MP4(H.264/CRF18) + frames.csv 索引 + 深度网格**;帧↔行严格 1:1;深度按 ts 去重 + `depth_age_ms` 同步核算;限流;产出 `logs/episodes/ep_*/`(meta.json/video.mp4/frames.csv/depth) |
| 接入 | `tt_control/app.py`(+`config.py`/`main.py`) | `--record [--record-hz]`；起飞开录、降落/急停/断开收尾;逐帧记录动作与决策状态(MANUAL/CRUISE/TURN/BLOCKED) |
| 可信度报告 | `sim_match_report.py` | 用真机记录动作重放 SimDrone 运动学，量化逐轴 MAE/RMSE、航向误差、终点误差、路径长度比 + 真机vs仿真叠图 PNG |
| 深度 RTT | `tt_control/depth_backend.py` | 新增 `infer_ms` 属性供录制器记 `depth_rtt_ms` |
| **AUTO 看门狗** | `tt_control/auto_safety.py` + `app.py` | 感知失联(深度>1.5s)/挂载超时(>30s)→ 自动悬停解除 AUTO + HUD显因 + 录制器记因;控制回路已接入并 headless 验证(此前 docs 声称"已接入"实为缺失,本轮补齐) |
| 测试 | `tests/test_episode_recorder.py` · `tests/test_sim_match_report.py` · `tests/test_auto_safety.py` | 结构/字段/深度去重/限流/SimDrone 端到端 + 匹配误差方向 + 看门狗逻辑；全量 **55 项 pytest 绿** |

用法：
```bash
# 采集(真机或 --sim)：飞行全程同步录数据
python main.py --inference depth-anything --depth-service http://127.0.0.1:8899/depth --record
python main.py --sim --record                      # 离线验证采集管线

# 出仿真可信度报告(真机 episode 录完后)
python sim_match_report.py logs/episodes/ep_<stamp>
```

> 第 2/3 步为真机现场项(需飞机+场地),代码侧已就绪:一开 `--record`,下次实飞就能把训练级数据采回来。第 5 步待实飞数据回填后成文交接。
