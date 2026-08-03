# 机器狗型号探测清单

日期：2026-07-31  
用途：连上设备后填写，供 `DogSdkAdapter(mode=backend)` 挂真导航前确认。  
状态：待现场填写（宇树，具体型号未确认）

## 识别字段

| 字段 | 值 | 备注 |
|---|---|---|
| 品牌 | 宇树 Unitree | |
| 型号（铭牌/App） | | B1 / B2 / Go2 等 |
| 序列号 / SN | | |
| 固件版本 | | |
| 机载算力（工控机/板卡） | | 如 Legion / Jetson |
| 操作系统 | | Ubuntu 版本 |
| 导航栈 | | 厂商 App / ROS1 / ROS2 |
| LiDAR 型号 | | |
| 地图格式 / 路点 API | | `goto(goal_id)` 形参名 |
| 取消导航 API | | 对应 `NavBackend.cancel` |
| 到点判定 | | 回调 / 轮询 |
| 相机接口 | | 本地重找物体 A |
| 气检 / RS485 | | 有则填协议 |
| 网络 | | IP / 端口 / 是否需 VPN |

## 与语义地图对齐检查

- [ ] 狗地图内存在每个 `dog_goal_id`（与 `configs/mission/shared_map.*.json` 同名）
- [ ] 连续导航同一 staging 路点可重复到点
- [ ] `cancel` 后不再前进
- [ ] 断网/急停行为符合安全预期

## 禁止

- 未确认型号前不固化 `Unitree*` 目录名进生产路径  
- 不把无人机估的 XYZ 写进狗目标  
