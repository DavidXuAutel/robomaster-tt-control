from __future__ import annotations

import inspect
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

IMAGE_SIZE_WH = (224, 224)


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def actions_chunk_to_primitive(action_chunk: np.ndarray) -> int:
    """Use first executed step (or mean of first replan_steps) → nearest primitive."""
    from experiments.aerial.openfly_actions import delta_to_nearest_primitive

    step = np.asarray(action_chunk[0], dtype=np.float64)
    return delta_to_nearest_primitive(step)


class FastWAMAerialPolicy:
    """FastWAM infer_action → OpenFly discrete primitive id."""

    def __init__(
        self,
        model: Any,
        processor: Optional[Any] = None,
        action_horizon: int = 16,
        replan_steps: int = 1,
        num_inference_steps: int = 10,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        text_cfg_scale: float = 1.0,
        negative_prompt: str = "",
        rand_device: str = "cpu",
        tiled: bool = False,
        num_video_frames: Optional[int] = None,
        dump_video: bool = False,
    ) -> None:
        self.model = model
        self.processor = processor
        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, self.action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self._num_video_frames = num_video_frames
        # When True, infer_action also returns the decoded world-model video; the
        # most recent clip (list[PIL.Image]) is stashed here for the eval runner to
        # dump. The runner toggles dump_video per step to bound VAE-decode cost.
        self.dump_video = bool(dump_video)
        self.last_generated_frames: Optional[list[Any]] = None

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        if self.processor is None:
            return torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if self.processor is None:
            return np.asarray(action.detach().cpu().numpy(), dtype=np.float64)

        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _build_image_tensor(self, obs_rgb: np.ndarray) -> torch.Tensor:
        image = _resize_rgb(obs_rgb, IMAGE_SIZE_WH)
        device = getattr(self.model, "device", "cpu")
        dtype = getattr(self.model, "torch_dtype", torch.float32)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=device,
            dtype=dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _infer_action_chunk(
        self,
        obs_rgb: np.ndarray,
        state: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        image_tensor = self._build_image_tensor(obs_rgb)
        proprio = self._normalize_state(np.asarray(state, dtype=np.float32).reshape(4))

        prompt = DEFAULT_PROMPT.format(task=instruction)
        infer_kwargs = {
            "prompt": prompt,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        infer_params = inspect.signature(self.model.infer_action).parameters
        if self._num_video_frames is not None and "num_video_frames" in infer_params:
            infer_kwargs["num_video_frames"] = int(self._num_video_frames)

        want_video = self.dump_video and "return_video" in infer_params
        if want_video:
            infer_kwargs["return_video"] = True

        with torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)

        self.last_generated_frames = pred.get("video") if want_video else None

        action_tensor = pred["action"]
        action_chunk = self._denormalize_action(action_tensor)
        if action_chunk.ndim == 3:
            action_chunk = action_chunk[0]
        return np.asarray(action_chunk, dtype=np.float64)

    def predict_primitive(
        self,
        obs_rgb: np.ndarray,
        state: np.ndarray,
        instruction: str,
    ) -> int:
        """Run FastWAM and map the first replan step to an OpenFly primitive id."""
        action_chunk = self._infer_action_chunk(obs_rgb, state, instruction)
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        exec_chunk = action_chunk[:n_exec]
        return actions_chunk_to_primitive(exec_chunk)
