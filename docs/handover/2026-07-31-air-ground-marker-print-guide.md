# 标记物打印 / 张贴指南（软件侧约定）

日期：2026-07-31  
**物理边长与飞行高度待真机标定后再冻结**；本稿只固定「用什么、怎么编号」。

## 单点室内（当前默认软件路径）

| 用途 | 形态 | ID / label |
|---|---|---|
| 区域锚点 | 高饱和**蓝色**色板 | `AX-01`（写入 shared_map） |
| 物体 A | 高饱和**红色**色板 | `object_a` |

二者必须不同色，禁止红锚+红物体。

## 多区域

| 用途 | 形态 | ID |
|---|---|---|
| 各区入口锚点 | AprilTag **tag36h11** | `TAG-0`, `TAG-1`, … 与地图 `anchor_ids` 一致 |
| 物体 A | 仍建议红色板 | `object_a` |

打印图源（官方）：[AprilRobotics/apriltag-imgs](https://github.com/AprilRobotics/apriltag-imgs)  
仓库内缩小样例：`tests/fixtures/mission/apriltags/`（测试用，打印请用高分辨率官方图）。

## 与语义地图

`configs/mission/shared_map.*.json` 里 `anchor_ids` 必须与现场贴纸 ID 一致；`dog_goal_id` 与狗地图路点同名。
