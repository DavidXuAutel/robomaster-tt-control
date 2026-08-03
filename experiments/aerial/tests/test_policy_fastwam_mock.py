import numpy as np

from experiments.aerial.eval.policy_fastwam import actions_chunk_to_primitive


def test_chunk_mean_maps_to_forward():
    chunk = np.tile(np.array([3.0, 0.0, 0.0, 0.0]), (8, 1))
    assert actions_chunk_to_primitive(chunk) == 1
