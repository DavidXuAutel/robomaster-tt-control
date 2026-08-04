# 人工示教（含意图标注）采数方案

面向 **Kairos / ScoutWAM 类世界-动作模型（WAM）的动作头 SFT**。
操作员与后续 Agent 照本文即可采、标、验收、打包。

最后更新：2026-07-29

相关文档：
- 飞行模块总手册：`docs/handover/2026-07-28-flight-modes-and-data-collection-runbook.md`
- 标注总表模板：`docs/handover/templates/teleop_manifest.csv`
- 背景判断：规则式 AUTO 数据更适合训世界模型头；动作头应以本方案人工示教为主
  （待大众确认微调用法后，可再调整配比）

---

## 0. 为什么单独做这一套

WAM 微调通常同时涉及：

| 头 | 学什么 | 对本数据的要求 |
|---|---|---|
| 世界模型头 | 画面 + 动作 → 下一帧画面 | 动作真实、对齐准即可；策略可次优 |
| 动作头（SFT） | 画面（+语言）→ 杆量 | **会直接模仿示范动作**；次优规则策略会把模型练偏 |

因此：**动作头 SFT 用人工示教 + 一句意图；规则式 wander/orbit 另库交付，勿混作模仿示范。**

---

## 1. 目标与成功标准

目标：交付「RGB 视频 + 逐帧四轴杆量 + 一句意图」的人工示教轨迹。

成功标准：
1. 全程人工（几乎全部帧 `ctrl_state=MANUAL`），**不挂 AUTO（不要按 `V`）**
2. 一次起飞只完成一条意图，意图句与画面一致
3. 视频帧数 = `frames.csv` 行数；有动作帧占比 > 85%
4. 每段在 `teleop_manifest.csv` 有一行，且 `quality=ok` 才进交付包

---

## 2. 任务清单（先采 6 类）

每类 **8～12 段**，段长 **20～60 秒**（10 Hz ≈ 200～600 帧；足够 Kairos 类
模型按 clip 均匀采样 81 帧）。同一意图建议连飞 2～3 遍，略变起点/朝向。

| task | 意图句示例（intent） | 备注 |
|---|---|---|
| `approach` | 向前接近椅子，停在约 1 米处 | 终点悬停 1～2 s 再降落 |
| `orbit_chair` | 顺时针绕椅子飞一圈，保持椅子在画面中 | 可另采逆时针，intent 写清方向 |
| `avoid_left` / `avoid_right` | 前方有椅子，向左绕开后继续前飞 | 左右分开标，勿写「绕开」含糊句 |
| `corridor` | 沿通道向前飞，保持居中 | 笼内用两侧障碍物夹出通道即可 |
| `yaw_explore` | 原地左转约 90°，再向前飞 | 转角写进意图 |
| `altitude` | 升高到约 1 米再下降到约 0.6 米 | 补 throttle 覆盖（现有自动数据几乎无升降） |

可选加采（有余力）：急停悬停、近障后撤离、换障碍材质/光照复采同类任务。

意图句写法：
- ✅ 「向左绕开椅子后继续前飞」
- ❌ 「飞一下」「避障」「随便转转」

---

## 3. 启动命令（示教专用）

示教**不依赖** orbit/wander 策略，配置档只用深度与录制管线。推荐 `default.json`
（或任意档均可），**关键是飞行中不按 `V`**。

```bash
python3 station_mode.py find

.venv/bin/python main.py \
  --config configs/default.json \
  --tello-ip <扫到的 IP> \
  --local-ip 192.168.0.103 \
  --inference depth-anything \
  --start-depth-service \
  --record --record-hz 10 \
  -v 2>&1 | tee logs/teleop-$(date +%H%M).log
```

深度服务铁律同总手册：必须 `--start-depth-service`；收工自检
`lsof -nP -iTCP:8899,8890 -sTCP:LISTEN` 应无输出。

---

## 4. 单次采数 SOP

### 4.1 飞前（先写意图，再起飞）

1. 电量 ≥ 60%。
2. 在纸笔或 `teleop_manifest.csv` 预填：`intent` / `task` / `scene` / `lighting`。
3. 启动程序，`C` 连接，确认 `drone connected`。
4. **确认 AUTO 为 OFF**（不要按 `V`）。

### 4.2 飞行

1. `T` 起飞（录制自动开始，新建一个 episode）。
2. 悬停稳定后，按预填意图一口气飞完。
3. 中途失误 → `L` 降落，本段标 `quality=discard`，重飞；**不要边飞边改口**。
4. 成功 → `L` 降落，立刻在 manifest 填 `episode_id` 与 `quality=ok`。

要点：
- 一次起飞 = 一个 episode = 一条意图。
- 示教段 20～60 s 即可，不必强求 2 分钟（那是自动探索数据的建议）。
- 全程用 WASD / 方向键；`SPACE` 可短悬停，但意图若要求连续运动则少用。

### 4.3 飞后当场登记

打开 `docs/handover/templates/teleop_manifest.csv`（可复制到
`logs/exports/teleop/teleop_manifest.csv` 作为当次交付总表），补全该行。
`episode_id` 以 `logs/episodes/` 下新建目录名为准。

---

## 5. 标注字段（manifest）

| 字段 | 必填 | 说明 |
|---|---|---|
| `episode_id` | ✓ | 如 `ep_20260729_103012` |
| `mode` | ✓ | 固定填 `teleop` |
| `task` | ✓ | 见 §2 的 task 名 |
| `intent` | ✓ | 一句中文意图，建议 ≤ 30 字 |
| `scene` | ✓ | 如「笼内，中央一把椅子」 |
| `lighting` | ✓ | `daylight` / `lamps_on` / `dim` |
| `quality` | ✓ | `ok` / `discard` |
| `duration_s` | 建议 | 从 meta 或目测填写 |
| `notes` | 可选 | 异常、可保留的瑕疵说明 |

每个 `quality=ok` 的 episode 目录内建议另附短 `README.md`（可从 manifest 抄
intent/scene），方便对方单包阅读。

---

## 6. 验收门槛

交付前对每个 `ok` 段检查：

```bash
# 视频帧数 == CSV 行数（示例）
.venv/bin/python - <<'PY'
import csv, cv2, sys
from pathlib import Path
ep = Path(sys.argv[1])
n_csv = sum(1 for _ in open(ep/"frames.csv")) - 1
cap = cv2.VideoCapture(str(ep/"video.mp4"))
n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
print(ep.name, "csv", n_csv, "video", n_vid, "OK" if n_csv==n_vid else "FAIL")
PY
logs/episodes/ep_XXXXXXXX_XXXXXX
```

另查：
- `meta.json` 中 `frames_manual` 应接近 `n_frames`（示教不应出现大段 AUTO）
- 有动作帧占比 > 85%（`act_roll/pitch/throttle/yaw` 至少一个非零）
- intent 与画面抽查一致（建议抽 10%）
- `quality=discard` **不打进交付 zip**

说明：`episode_check.py` 面向 wander 事件验收，**不要用它卡示教数据**。

---

## 7. 打包与目录约定

```text
logs/exports/teleop/
├── teleop_manifest.csv          # 当次交付总表（仅含 quality=ok）
├── ep_YYYYMMDD_HHMMSS.zip       # 各示教 episode
└── README.md                    # 给对方的总说明（见下）
```

打包单个 episode（与既有流程一致）：

```bash
mkdir -p logs/exports/teleop
cd logs/episodes
zip -r ../exports/teleop/ep_YYYYMMDD_HHMMSS.zip ep_YYYYMMDD_HHMMSS \
  -x "*.DS_Store"
```

交付总说明 `logs/exports/teleop/README.md` 建议写清：
1. 数据用途：**动作头 SFT（人工示教）**；含 `intent` 语言标注
2. 对齐保证：视频第 N 帧 ↔ `frames.csv` 第 N 行
3. 动作列：`act_roll/pitch/throttle/yaw`，范围 −100～+100
4. 与 `exports/` 下规则式 AUTO 包的区别：**请勿把 AUTO 包当动作模仿数据混训**

规则式 wander/orbit 包继续放在 `logs/exports/`（或 `logs/exports/auto_policy/`），
manifest 中 `mode` 标 `auto_wander` / `auto_orbit`，与 teleop 分目录。

---

## 8. 第一周落地量（建议）

| 日 | 任务 | 目标段数 |
|---|---|---|
| Day1 | `approach` + `orbit_chair` | 各 10 |
| Day2 | `avoid_*` + `corridor` | 各 10 |
| Day3 | `yaw_explore` + `altitude` | 各 8，并换光照复采半天 |

合计约 **50～60 段 / 30～50 分钟有效飞行**，足够对方先试微调。
场景尽量换布局与光照；动作覆盖上注意补 **横移（roll）与升降（throttle）**。

---

## 9. 与自动策略数据的配比（默认建议）

在大众确认微调用法之前，默认按以下理解执行：

| 数据集 | 业务用途建议 |
|---|---|
| 本方案 teleop | 动作头 SFT 主力；世界模型头也可复用 |
| 规则式 wander / orbit | 仅建议给世界模型头 / 失败纠错；**默认不喂动作头** |

若对方书面确认「动作头不用你们的杆量、只训世界模型头」，则示教占比可下调，
改为补激励飞行（四轴覆盖）即可。

---

## 10. Agent 执行红线

1. 示教采数禁止开启 AUTO（`V`）。
2. 禁止把 wander/orbit 自动回合改标成 `mode=teleop`。
3. 禁止无 `intent` 的示教段进入交付包。
4. 调参/改模块代码仍走总手册与 orbit/wander 守则；本方案**不改控制律**。
