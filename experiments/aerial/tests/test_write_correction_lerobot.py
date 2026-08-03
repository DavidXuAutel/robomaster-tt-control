from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from experiments.aerial.write_correction_lerobot import load_correction_episode


def _write_episode(tmp_path, *, action=None):
    image_path = tmp_path / "ep000_frames" / "frame000000.png"
    image_path.parent.mkdir()
    Image.fromarray(np.full((8, 9, 3), 127, dtype=np.uint8)).save(image_path)
    record = {
        "observation.images.ego": str(image_path.relative_to(tmp_path)),
        "observation.state": [1.0, 2.0, 3.0, 0.25],
        "action": [3.0, 0.0, 0.0, 0.0] if action is None else action,
        "task": "fly forward",
        "intervention": True,
    }
    episode_path = tmp_path / "ep000.jsonl"
    episode_path.write_text(json.dumps(record) + "\n")
    return episode_path


def test_load_correction_episode_emits_lerobot_schema(tmp_path):
    frames = load_correction_episode(_write_episode(tmp_path))

    assert len(frames) == 1
    frame = frames[0]
    assert frame["observation.images.ego"].shape == (8, 9, 3)
    assert frame["observation.images.ego"].dtype == np.uint8
    assert frame["observation.state"].shape == (4,)
    assert frame["action"].shape == (4,)
    assert frame["task"] == "fly forward"
    assert frame["meta.action_source"] == "pos_delta_v1"


@pytest.mark.parametrize("action", ([1.0, 2.0, 3.0], [1.0, np.nan, 0.0, 0.0]))
def test_load_correction_episode_rejects_invalid_actions(tmp_path, action):
    episode_path = _write_episode(tmp_path, action=action)

    with pytest.raises(ValueError, match="action"):
        load_correction_episode(episode_path)
