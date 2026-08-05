"""Runtime wrapper: DepthHead → ``obs.info['depth_min_pred']`` (frozen §4 ④ wiring).

Kept separate from ``dynamics_torch`` so the collector can depend on a tiny
protocol without importing torch at module import time on GPU-less hosts.
The real ``_DepthHead`` is loaded lazily inside ``from_checkpoint``.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Deque, Optional

import numpy as np

from experiments.aerial.rl.env.obs import Observation


class DepthMinPredictor:
    """Maintain a short RGB history and emit scalar ``depth_min_pred``.

    Spec: collector must set ``obs.info['depth_min_pred']`` **before**
    ``safety.should_override``. When no checkpoint is loaded this is a no-op
    (returns None) so V0 default collection stays shield-inert.
    """

    def __init__(self, *, n_frames: int = 4, device: str = "cpu") -> None:
        self.n_frames = int(n_frames)
        self.device = device
        self._hist: Deque[np.ndarray] = deque(maxlen=self.n_frames)
        self._model: Any = None

    @classmethod
    def from_checkpoint(
        cls,
        path: Path | str,
        *,
        device: str = "cpu",
    ) -> "DepthMinPredictor":
        import torch
        from experiments.aerial.rl.dynamics_torch import _DepthHead

        payload = torch.load(str(path), map_location="cpu")
        n_frames = int(payload.get("n_frames", 4))
        model = _DepthHead.from_payload(payload)
        model.load_state_dict(payload["model"], strict=True)
        model.to(device)
        model.eval()
        pred = cls(n_frames=n_frames, device=device)
        pred._model = model
        return pred

    def reset(self) -> None:
        self._hist.clear()

    def predict_min(self, obs: Observation) -> Optional[float]:
        """Push ``obs.rgb`` into history; return min ``D̂`` or None if unloaded."""
        if self._model is None:
            return None
        import torch

        rgb = np.asarray(obs.rgb, dtype=np.uint8)
        self._hist.append(rgb)
        # Pad left with the oldest frame if history is still warming up.
        frames = list(self._hist)
        while len(frames) < self.n_frames:
            frames.insert(0, frames[0])
        stack = np.stack(frames[-self.n_frames :], axis=0)  # [L,H,W,3]
        tensor = torch.from_numpy(stack).unsqueeze(0)  # [1,L,H,W,3]
        with torch.no_grad():
            depth, _ = self._model.predict_from_window(tensor.to(self.device))
        d = depth.squeeze(0).detach().float().cpu().numpy()
        finite = d[np.isfinite(d) & (d > 0)]
        if finite.size == 0:
            return None
        return float(np.min(finite))
