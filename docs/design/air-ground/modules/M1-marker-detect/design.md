# M1 · 标记物检测（蓝锚 / 红物体 / AprilTag fail-fast）

状态：开工中  
依据：G1–G2 §2；闸门：不装新依赖除非主人批准  

## 目标

1. 锚点与物体 A **解耦**（默认蓝锚 + 红物体）
2. `MarkerSpec` + `detect_marker`
3. `mode=apriltag` 缺库 → **抛错**，禁止静默回退 color
4. `need_anchor_frames` 生效
5. 无 `mode=auto`

## 非目标

装 `pupil-apriltags`；冻结物理尺寸；真机录制。

## 接口

```text
MarkerSpec(label, kind=color|apriltag, hsv_ranges?, tag_ids?, min_area_ratio, min_confidence)
PRESET_OBJECT_A_RED / PRESET_ANCHOR_BLUE
detect_marker(frame, spec) -> Detection | None
RegionConfirmer(..., mode=color|apriltag, need_frames, color_spec=PRESET_ANCHOR_BLUE)
  apriltag 缺库 → RuntimeError
```

可选 `detector` 注入便于无库测 Tag 路径。

## 适配器

Tello/Autel：`anchor_mode` 默认 `color`；物体用红 preset；`need_anchor_frames` 传入 Confirmer。

## 测试

- 仅红 / 仅蓝 → 不确认区域+物体同报
- 蓝+红 → found
- apriltag 无库构造 → RuntimeError
- need_frames=N 生效
