from collections.abc import Mapping, Sequence

import torch
from torch.utils.data import Dataset


class WeightedSourceDataset(Dataset):
    """Sample one of several datasets according to fixed source probabilities.

    Two mixing modes:

    - ``deterministic=False`` (default): each draw picks a source with
      ``torch.multinomial(probs)``. Correct in expectation, but the realized
      per-window rate is a binomial random variable, so a short window can drift
      far from the target (this is what tripped the FT correction-rate gate).

    - ``deterministic=True``: use a streaming largest-remainder scheduler so the
      cumulative source counts track ``probs`` as closely as integer counts
      allow. Any contiguous window of ``W`` draws then contains ``round(W*p)``
      samples of each source to within one, regardless of window alignment or a
      mid-run resume. The source mix is exact; the *sample within* the chosen
      source is still drawn from ``generator`` so data variety is preserved.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        probs: Sequence[float],
        names: Sequence[str],
        generator: torch.Generator | None = None,
        deterministic: bool = False,
    ):
        if not (len(datasets) == len(probs) == len(names)):
            raise ValueError("datasets, probs, and names must have equal lengths")
        if not datasets:
            raise ValueError("at least one source dataset is required")
        if any(len(dataset) == 0 for dataset in datasets):
            raise ValueError("source datasets must not be empty")
        if any(probability < 0 for probability in probs):
            raise ValueError("source probabilities must be non-negative")
        if abs(sum(probs) - 1.0) >= 1e-6:
            raise ValueError("source probabilities must sum to 1")
        if len(set(names)) != len(names):
            raise ValueError("source names must be unique")

        self.datasets = list(datasets)
        self.probs = torch.tensor(probs, dtype=torch.float64)
        self.names = list(names)
        self.generator = generator if generator is not None else torch.Generator().manual_seed(0)
        self.deterministic = bool(deterministic)
        self.counts = {name: 0 for name in self.names}

        # Streaming apportionment state for deterministic mode.
        self._probs_list = [float(probability) for probability in probs]
        self._draw_ordinal = 0
        self._assigned = [0 for _ in self.names]

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def _next_source_deterministic(self) -> int:
        """Pick the source that is furthest behind its target share so far.

        For draw ``n`` (0-based) the target cumulative count for source ``s`` is
        ``(n + 1) * p_s``; we assign the draw to whichever source has the largest
        gap between its target and what it has actually received. Ties break to
        the lowest index for reproducibility.
        """

        drawn = self._draw_ordinal + 1
        best_index = 0
        best_debt = None
        for index, probability in enumerate(self._probs_list):
            debt = probability * drawn - self._assigned[index]
            if best_debt is None or debt > best_debt:
                best_debt = debt
                best_index = index
        self._assigned[best_index] += 1
        self._draw_ordinal += 1
        return best_index

    def __getitem__(self, _index):
        if self.deterministic:
            source_index = self._next_source_deterministic()
        else:
            source_index = int(torch.multinomial(self.probs, 1, generator=self.generator).item())
        self.counts[self.names[source_index]] += 1
        sample_index = int(
            torch.randint(
                len(self.datasets[source_index]),
                (1,),
                generator=self.generator,
            ).item()
        )
        sample = self.datasets[source_index][sample_index]
        if not isinstance(sample, Mapping):
            raise TypeError("source datasets must return mapping samples")
        result = dict(sample)
        result["data_source"] = self.names[source_index]
        return result

    def pop_source_counts(self) -> dict[str, int]:
        counts = self.counts.copy()
        self.counts = {name: 0 for name in self.names}
        return counts
