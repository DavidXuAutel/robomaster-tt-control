"""半自动避障的 AUTO 看门狗：纯逻辑、无 I/O，便于单测。

在 AUTO ON 期间每次下发杆量前判定，命中即让上层悬停并解除 AUTO（人工接管仍是第一保障）：
1. **单次挂载超时**：AUTO ON 持续超过 `max_engaged_s`，防止无人值守久飞。
2. **感知失联**：距最近一帧有效深度超过 `depth_stale_s`（含挂载后迟迟无深度），
   避免在 depth 服务挂掉后继续用「复用的旧深度」盲飞。

时钟约定：`now / engaged_since / last_depth_ts` 均用同一墙钟（`time.time()`），
与 depth_backend.DepthFrame.ts 一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AutoWatchdog:
    max_engaged_s: float = 30.0
    depth_stale_s: float = 1.5

    def check(
        self,
        now: float,
        engaged_since: Optional[float],
        last_depth_ts: Optional[float],
    ) -> Optional[str]:
        """返回需要解除 AUTO 的原因字符串；无需解除则返回 None。"""
        if engaged_since is None:
            return None  # 未挂载
        if now - engaged_since > self.max_engaged_s:
            return f"engaged>{self.max_engaged_s:.0f}s"
        if last_depth_ts is None:
            # 挂载后宽限期内允许还没有深度帧；超过阈值仍无深度 → 判失联
            if now - engaged_since > self.depth_stale_s:
                return f"no depth>{self.depth_stale_s:.1f}s"
            return None
        if now - last_depth_ts > self.depth_stale_s:
            return f"depth stale>{self.depth_stale_s:.1f}s"
        return None
