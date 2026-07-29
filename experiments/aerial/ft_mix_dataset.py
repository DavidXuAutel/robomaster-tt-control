from collections.abc import Sequence

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from fastwam.datasets.lerobot.weighted_source_dataset import WeightedSourceDataset


def build_ft_mix_dataset(
    original: DictConfig,
    correction: DictConfig,
    source_probs: Sequence[float] = (0.75, 0.25),
    source_names: Sequence[str] = ("original", "correction"),
    generator_seed: int = 0,
    deterministic: bool = True,
    processor: DictConfig | None = None,
) -> WeightedSourceDataset:
    """Build the two RobotVideoDataset sources and apply source mixing.

    ``deterministic`` defaults to True so the realized correction rate holds the
    target share in every window and does not trip the FT correction-rate gate
    by sampling noise; the sample drawn within each source stays random.
    """

    del processor  # Exposed at data.train.processor for model config interpolation.
    datasets = [instantiate(original), instantiate(correction)]
    generator = torch.Generator().manual_seed(int(generator_seed))
    return WeightedSourceDataset(
        datasets=datasets,
        probs=source_probs,
        names=source_names,
        generator=generator,
        deterministic=deterministic,
    )
