# M2 · G2 合成契约 runner（synthetic_contract）

状态：开工  
闸门：不装 pupil-apriltags；Tag 场景用注入 detector；只报 synthetic，不冒充真机。

## 交付

- `mission_brain/g2_runner.py`：读场景 manifest → 喂帧 → 断言事件
- `tests/test_g2_scripted_scenes.py`：≥20 参数化场景，确定性 oracle
- evidence 可解码（imwrite 校验已在 Tello）

## 场景（代码生成，非 20 套大 PNG）

S01–S05 happy 蓝+红；S06–S08 仅蓝；S09–S10 仅红；S11–S12 streak不足；  
S13–S14 低置信（小红点）；S15 错 Tag（注入 det）；S16 重复；S17 abort；  
S18 两区串行；S19 暗光（预先写死 expect=no_dispatch）；S20 噪声红斑。
