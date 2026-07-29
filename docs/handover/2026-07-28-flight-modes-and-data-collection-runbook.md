# 飞行模块入口与采数运行手册

面向**后续接手的 Agent 与操作员**。目标：任何人不看历史会话，照本手册即可
选对模块、用对配置、按规范采数与交付，且不会踩到前人用真机炸出来的坑。

最后更新：2026-07-29（增补人工示教采数入口）

---

## 0. 先读什么

| 顺序 | 文件 | 为什么 |
|---|---|---|
| 1 | `CLAUDE.md` | 项目总入口、进度、真机测试铁律 |
| 2 | **本文件** | 模块怎么选、怎么启动、怎么采数交付 |
| 3 | `docs/handover/2026-07-29-teleop-intent-data-collection.md` | **人工示教 + 意图标注**（动作头 SFT）；与规则式 AUTO 分库 |
| 4 | 对应模块守则（见 §5） | 动那个模块的代码/参数前**必读**，都是真机事故沉淀 |
| 5 | `docs/handover/2026-07-25-capability-inventory.md` | 11 个模块的详细能力与键盘映射 |

**动手前的硬性前提**：修改 orbit 或 wander 的代码与参数，必须先读完 §5 列出的
守则文件。这两个模块的每一个参数都对应过一次真机事故，凭现象猜参数是所有
历史坑的共同起点。

---

## 1. 模块 × 配置档对照表（最重要的一张表）

飞行模块由 `fsm` 节的两个互斥开关决定，**通过 `--config` 选档切换，不要手改
`configs/default.json` 去切模式**。

| 飞行模块 | 配置档 | `orbit_mode` | `wander_mode` | 用途 |
|---|---|---|---|---|
| POI 环绕（椅子绕飞） | `configs/orbit-chair.json` | true | false | 锁定一个目标持续绕圈，采「视觉伺服」数据 |
| 随机漫游（铁丝笼） | `configs/wander-cage.json` | false | true | 自主探索 + 遇障转向，采「探索避障」数据 |
| 随机漫游（通用） | `configs/wander.json` | false | true | 非笼内场地的漫游基线 |
| 半自动避障 / 默认 | `configs/default.json` | true | false | 参数基线快照，也是 orbit 的出厂默认 |

### 各档的独立性约定（2026-07-28 主人裁定）

1. **绕飞与漫游是彼此独立的模块**，参数、规则互不耦合。
2. 每个模块的调参**只准写进自己的档**，不回灌 `configs/default.json`。
   default.json 的 orbit/fsm 节视为 2026-07-27 已验证基线快照。
3. `configs/orbit-chair.json` 把 orbit 的 22 个参数**显式写全**，不依赖
   default.json 回退——这样他人改 default.json 也波及不到绕飞。
4. 已知残留耦合：`avoid` 节的 `band_top` / `band_bottom` 被两模块共享
   （`WanderPolicy` 借用 `AvoidanceController.zone_nearness()` 取三区近度）。
   **禁止为调漫游而动这两项。** 详见
   `docs/dev-notes/2026-07-28-orbit-wander-decoupling.md`。

---

## 2. 启动命令

### 2.1 飞前必做：找飞机 IP

飞机重启后 IP 会变，**每次开飞前重新扫描**，不要沿用上次的 IP：

```bash
python3 station_mode.py find
```

若提示未找到，等几秒重试（飞机组网需要时间）。仍找不到则确认飞机指示灯为
组网状态、与 Mac 同一路由器。首次组网用 `python3 station_mode.py setup`。

注意：`station_mode.py find` 需要绑定 8889 端口，**主程序运行时会冲突**，
先关掉主程序再扫。

### 2.2 椅子绕飞（Orbit）

```bash
.venv/bin/python main.py \
  --config configs/orbit-chair.json \
  --tello-ip 192.168.0.101 \
  --local-ip 192.168.0.103 \
  --inference depth-anything \
  --start-depth-service \
  --record --record-hz 10 \
  -v 2>&1 | tee logs/orbit-chair-$(date +%H%M).log
```

### 2.3 随机漫游（Wander）

```bash
.venv/bin/python main.py \
  --config configs/wander-cage.json \
  --tello-ip 192.168.0.101 \
  --local-ip 192.168.0.103 \
  --inference depth-anything \
  --start-depth-service \
  --record --record-hz 10 \
  -v 2>&1 | tee logs/wander-cage-$(date +%H%M).log
```

### 2.4 仿真（无飞机，验证逻辑用）

```bash
.venv/bin/python main.py --sim --config configs/orbit-chair.json --record -v
```

### 2.5 参数说明

| 参数 | 说明 |
|---|---|
| `--config` | 选模块，见 §1 对照表 |
| `--start-depth-service` | **必须带**，受管启动本地深度服务，退出时 atexit 自动停 |
| `--record --record-hz 10` | 采数时带上；不采数可省略 |
| `--tello-ip` | 每次由 `station_mode.py find` 现扫 |
| `--local-ip` | Mac 在路由器下的 IP（当前环境 192.168.0.103） |

### 2.6 深度服务铁律（违反会残留进程占 GPU）

- **必须**用 `--start-depth-service`。
- **禁止** `server/da_v2_service.py &` 再挂 `--depth-service http://127.0.0.1:...`
  做本地真机测试——外挂进程不随 main 退出。
- 仅当主人明确要求连远端常驻服务时才用 `--depth-service URL`。
- 收工自检，应无输出：

```bash
lsof -nP -iTCP:8899,8890 -sTCP:LISTEN
```

---

## 3. 采数标准流程

> **用途分流（2026-07-29）**  
> - 规则式 AUTO（orbit / wander）→ 见本节；更适合世界模型头 / 探索数据。  
> - **人工示教 + 意图标注（动作头 SFT）→ 另循**  
>   `docs/handover/2026-07-29-teleop-intent-data-collection.md`  
>   （不按 `V`、一段一意图、填 `templates/teleop_manifest.csv`）。  
> 两套数据分目录交付，勿把 AUTO 回合改标成 teleop。

### 3.1 飞前

1. 确认电量 ≥ 60%（一个 4~5 分钟回合约耗 55%）。
2. `python3 station_mode.py find` 拿到 IP。
3. 按 §1 选档，按 §2 启动，等日志出现 `深度推理服务就绪 ✓`。
4. 界面按 `C` 连接，确认日志 `drone connected`。

### 3.2 飞行

| 按键 | 作用 |
|---|---|
| `T` | 起飞（**录制在此刻自动开始**，每次起飞新建一个 episode） |
| `V` | AUTO 切换 OFF → ARMED → ON → ARMED |
| `WASD` / 方向键 | 人工接管（自动模式下打杆即接管，记为 `ctrl_state=MANUAL`） |
| `SPACE` | 悬停并解除 AUTO |
| `L` | 降落（录制在此刻收尾保存） |
| `ESC` | 急停 |
| `X` | 退出程序 |

要点：

- **一次起飞 = 一个 episode**，中途不必重启程序，落地再起飞会新建目录。
- 单回合建议飞满 2 分钟以上；低于 30 秒的片段数据量太小，不值得交付。
- 遇险随时人工接管，**人工接管的数据同样有价值**（是「视觉→操控」的人类示范），
  不需要为了数据纯净而硬扛。

### 3.3 飞后

按 `L` 降落，日志出现 `episode saved: .../ep_YYYYMMDD_HHMMSS (N frames)` 即落盘完成。

---

## 4. 飞后校验与打包

### 4.1 通用校验（与模块无关，必做）

```bash
D=logs/episodes/ep_20260727_205833   # 换成实际目录

.venv/bin/python - <<PY
import csv, collections, os
D="$D"
rows=list(csv.DictReader(open(f'{D}/frames.csv')))
nz=sum(1 for r in rows if any(float(r[k] or 0)!=0
        for k in ('act_roll','act_pitch','act_throttle','act_yaw')))
print("帧数:", len(rows), "| 有动作帧:", f"{nz}({nz/len(rows)*100:.0f}%)")
print("控制状态:", dict(collections.Counter(
      r['ctrl_state'].split()[0] if r['ctrl_state'] else 'EMPTY' for r in rows)))
print("时长: %.1fs" % (float(rows[-1]['t_mono_ms'])/1000))
PY
```

三条底线，任一不满足就不要交付：

- `video.mp4` 帧数 == `frames.csv` 行数（对齐是训练的前提）
- 有动作输出的帧占比 > 90%
- 飞行日志中 `abort` / `watchdog disengage` 计数符合预期

### 4.2 模块专用验收

**`episode_check.py` 只适用于 wander，不要拿它验收 orbit 数据。**
它的验收项是转向次数与方向多样性，orbit 回合天然没有 turn 事件，跑出来
必定 `OVERALL: FAIL`，那是工具适用边界问题，不是数据有问题。

```bash
# 仅限 wander episode
.venv/bin/python episode_check.py logs/episodes/ep_YYYYMMDD_HHMMSS
```

orbit 目前无专用脚本，按 `docs/design/2026-07-27-orbit-control-principles.md`
§6 的健康表人工对照：`tn` 是否收敛到 target、`|pos|` 波动、有无 abort。

### 4.3 写 README 并打包

每份交付数据**必须附 `README.md`**，说明对齐保证、动作列语义、`ctrl_state`
取值、以及任何会让对方误判的地方。可直接参考已交付的两份：

- `logs/episodes/ep_20260727_204010/README.md`（漫游 + 人工接管）
- `logs/episodes/ep_20260727_205833/README.md`（绕飞 · 纯自动）

```bash
mkdir -p logs/exports
cd logs/episodes && zip -rq ../exports/ep_YYYYMMDD_HHMMSS.zip ep_YYYYMMDD_HHMMSS && cd ../..
unzip -tq logs/exports/ep_YYYYMMDD_HHMMSS.zip && echo "✅ 压缩包完整"
```

### 4.4 交付数据规范

```
ep_YYYYMMDD_HHMMSS/
├── meta.json     回合元数据
├── video.mp4     H.264 / CRF18 / 960×720
├── frames.csv    逐帧数据，行数 == 视频帧数
├── depth/        深度 .npy（附带产物）
└── README.md     必须有
```

- **对齐保证**：`video.mp4` 第 N 帧 ↔ `frames.csv` 第 N 行，严格 1:1。
- **动作列**：`act_roll` / `act_pitch` / `act_throttle` / `act_yaw`，
  范围 −100~+100，语义同 Tello SDK `rc`，正方向依次为右 / 前 / 上 / 顺时针。
- **`ctrl_state`**：动作来源。wander 为干净枚举（`WANDER_CRUISE` 等）；
  **orbit 为带实时遥测的长字符串**，筛选须用 `startswith('ORBIT')`。
- **真实时序**：容器标称 10 fps，精确时刻以 `t_mono_ms` 为准。
- `meta.json` 的 `outcome: aborted / session_cleanup` 只是断连收尾的流程标记，
  **不代表飞行异常**，交付说明里要写清，免得对方误判。

大众训练所需的是**视频 + 操控动作**，深度非必需（`depth/` 是附带产物）。

---

## 5. 模块边界与红线

| 模块 | 守则文件 | 谁能改守则 |
|---|---|---|
| POI 环绕 | `docs/design/2026-07-27-orbit-control-principles.md` | **仅 Claude**，其他 Agent 只读 |
| 随机漫游 | `docs/design/2026-07-27-wander-explore-design.md` | **仅 Claude**，其他 Agent 只读 |

通用红线：

1. 上述两份守则文件**只允许 Claude 修改**，其他 Agent 一律只读，且禁止 chmod。
   有异议就在 `docs/dev-notes/` 新建笔记提出，等主人裁决，禁止自行偏离实现。
2. 不改 orbit 控制律结构、不改降落路径、不改 EpisodeRecorder 同步核心。
3. 不新增硬编码安全阈值——全项目安全阈值只允许一个来源（配置文件），
   新增看门狗必须接配置（这是历史坑 #9 的教训）。
4. 漫游安全不变量：除 `WANDER_RETREAT` 外 `pitch ≥ 0` 且 `roll ≡ 0`。
5. 环绕安全不变量：DANGER / LOST / ACQUIRE 期间杆量必须全零；
   目标偏离中央时禁止正 pitch 前冲。
6. 改参数**一次只改一个**，真机验证后再改下一个。多参数同调无法归因。

### 一个真实的反例（2026-07-28）

有人想用**漫游**模块实现「绕着椅子飞」，反复下调 `turn_thresh` 都不成功，
反而制造了误触发。根因是需求与模块能力错配，不是参数问题：

- 漫游遇障是 50°~130° 的偏航转向，语义是「转开去别处」，不是「绕着目标转」；
- 漫游安全不变量要求 `roll ≡ 0`，而 2026-07-24 真机结论是
  「yaw 绕障几何上无法绕过椅子，必须用 roll 横移」（见 `tt_control/avoidance.py:99`）；
- 三区近度取中带**中位数**，两米外的椅子占不到 mid 区一半像素，抬不动读数，
  降门槛只会先捞到远墙与地面的背景近度。

**教训：先确认需求该由哪个模块承载，再谈调参。** 绕目标飞的正确载体是
OrbitController。

---

## 6. 故障速查

| 现象 | 原因 | 处置 |
|---|---|---|
| `station_mode.py find` 报 `Address already in use` | 主程序占着 8889 | 先关主程序再扫 |
| 连接失败 / 找不到飞机 | 飞机重启后 IP 变了 | 重新 `find`，不要沿用旧 IP |
| `land` 首次超时、第二次 ok | Tello UDP 正常现象 | **不是 bug，不要「优化」重试逻辑**（历史坑 #7） |
| `AUTO watchdog disengage: depth stale` | 深度帧断流（撞击、图传中断） | 检查飞机状态；该保护是有意设计 |
| 环绕 `tn` 停在 0.5 不收敛 | pitch 距离增益问题 | 查历史坑 #6，别只调 target |
| 环绕左右摇头 | 增益过大 / 滤波退化 | 查历史坑 #4、#2，按「检测跳变→滤波→增益→结构」顺序排查 |
| 漫游原地反复 retry 打转 | VERIFY 灰区死锁 | 已修，见 `docs/dev-notes/2026-07-28-wander-verify-graylock.md` |
| 单目深度看不见铁丝网 | 稀疏栅格的物理局限，越近判越远 | 无软件解，靠降速 + 人工接管 |

---

## 7. 提交前必跑

```bash
.venv/bin/python -m pytest tests/ -q
```

改动 orbit / wander 时至少跑：

```bash
.venv/bin/python -m pytest tests/test_orbit.py tests/test_avoidance.py tests/test_wander.py -q
```

已知无关预存失败：`test_main_depth_guard.py::test_depth_inference_without_service_returns_2`。

必须用 `.venv/bin/python`（3.11），系统 `python3` 是 3.9 跑不了。
真机暴露的 bug，修复必须附带回归测试。
