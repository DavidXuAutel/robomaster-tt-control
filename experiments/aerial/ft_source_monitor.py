import logging
from collections import Counter
from collections.abc import Sequence


logger = logging.getLogger(__name__)


class FTSourceMonitor:
    """Log sampled sources and enforce correction rates over fixed windows.

    A single window that drifts out of ``[min_correction_rate,
    max_correction_rate]`` only logs a warning: with stochastic mixing the
    per-window rate is noisy and an occasional excursion is expected, so failing
    the run on one window would abort long runs by chance. The run is failed only
    after ``max_consecutive_violations`` consecutive out-of-band windows, which
    indicates a sustained mix problem rather than sampling noise.
    """

    def __init__(
        self,
        correction_name: str = "correction",
        log_every: int = 50,
        window_steps: int = 200,
        min_correction_rate: float = 0.20,
        max_correction_rate: float = 0.30,
        max_consecutive_violations: int = 3,
    ):
        if log_every <= 0 or window_steps <= 0:
            raise ValueError("log_every and window_steps must be positive")
        if not 0.0 <= min_correction_rate <= max_correction_rate <= 1.0:
            raise ValueError("correction rate bounds must satisfy 0 <= min <= max <= 1")
        if max_consecutive_violations <= 0:
            raise ValueError("max_consecutive_violations must be positive")

        self.correction_name = correction_name
        self.log_every = int(log_every)
        self.window_steps = int(window_steps)
        self.min_correction_rate = float(min_correction_rate)
        self.max_correction_rate = float(max_correction_rate)
        self.max_consecutive_violations = int(max_consecutive_violations)
        self._log_counts: Counter[str] = Counter()
        self._window_counts: Counter[str] = Counter()
        self.reset(start_step=0)

    def reset(self, *, start_step: int) -> None:
        """Reset process-local counters after startup or checkpoint resume."""

        self._log_counts.clear()
        self._window_counts.clear()
        self._log_start_step = int(start_step) + 1
        self._window_start_step = int(start_step) + 1
        self._skip_partial_window = int(start_step) % self.window_steps != 0
        self._consecutive_violations = 0

    def record(self, sources: str | Sequence[str], *, step: int) -> None:
        if isinstance(sources, str):
            sources = [sources]
        if not sources:
            raise ValueError("at least one data source is required per training step")

        self._log_counts.update(str(source) for source in sources)
        self._window_counts.update(str(source) for source in sources)

        if step - self._log_start_step + 1 >= self.log_every:
            total = sum(self._log_counts.values())
            correction_rate = self._log_counts[self.correction_name] / total
            logger.info(
                "[ft-data] source counts steps=%d-%d counts=%s correction_rate=%.4f",
                self._log_start_step,
                step,
                dict(sorted(self._log_counts.items())),
                correction_rate,
            )
            self._log_counts.clear()
            self._log_start_step = step + 1

        if step % self.window_steps == 0:
            if self._skip_partial_window:
                logger.info(
                    "[ft-data] skipping partial resumed source window steps=%d-%d",
                    self._window_start_step,
                    step,
                )
                self._window_counts.clear()
                self._window_start_step = step + 1
                self._skip_partial_window = False
                return
            total = sum(self._window_counts.values())
            correction_rate = self._window_counts[self.correction_name] / total
            if self.min_correction_rate <= correction_rate <= self.max_correction_rate:
                self._consecutive_violations = 0
            else:
                self._consecutive_violations += 1
                logger.warning(
                    "[ft-data] correction rate %.4f outside [%.4f, %.4f] for steps "
                    "%d-%d (consecutive %d/%d)",
                    correction_rate,
                    self.min_correction_rate,
                    self.max_correction_rate,
                    self._window_start_step,
                    step,
                    self._consecutive_violations,
                    self.max_consecutive_violations,
                )
                if self._consecutive_violations >= self.max_consecutive_violations:
                    raise RuntimeError(
                        "FT correction rate out of [%.4f, %.4f] for %d consecutive "
                        "windows; last window steps %d-%d rate %.4f"
                        % (
                            self.min_correction_rate,
                            self.max_correction_rate,
                            self._consecutive_violations,
                            self._window_start_step,
                            step,
                            correction_rate,
                        )
                    )
            self._window_counts.clear()
            self._window_start_step = step + 1
