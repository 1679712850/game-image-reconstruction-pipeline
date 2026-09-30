"""Shared, auditable inference budget for the existing detection pipeline."""
from dataclasses import dataclass, field

from app.detection_config import DetectionBudget


@dataclass
class BudgetTracker:
    config: DetectionBudget
    calls: int = 0
    global_passes: int = 0
    tile_passes: int = 0
    retry_calls: int = 0
    by_source: dict[str, int] = field(default_factory=dict)

    def begin_pass(self, source: str) -> bool:
        if source == "global":
            if self.global_passes >= self.config.max_global_passes:
                return False
            self.global_passes += 1
            return True
        if source == "tile":
            if self.tile_passes >= self.config.max_tile_passes:
                return False
            self.tile_passes += 1
            return True
        return True

    def consume(self, source: str) -> bool:
        """Reserve one model call; return false when the relevant budget is exhausted."""
        if self.calls >= self.config.max_total_inference_calls:
            return False
        if source in {"tile", "redetection", "gap_fill"}:
            limit = self.config.max_retry_calls if source in {"redetection", "gap_fill"} else self.config.max_total_inference_calls
            if source in {"redetection", "gap_fill"}:
                if self.retry_calls >= limit:
                    return False
                self.retry_calls += 1
        self.calls += 1
        self.by_source[source] = self.by_source.get(source, 0) + 1
        return True

    def report(self) -> dict:
        return {"calls": self.calls, "global_passes": self.global_passes,
                "tile_passes": self.tile_passes, "retry_calls": self.retry_calls,
                "by_source": dict(self.by_source),
                "max_total_inference_calls": self.config.max_total_inference_calls,
                "max_retry_calls": self.config.max_retry_calls}


def restore_budget(config: DetectionBudget, report: dict) -> BudgetTracker:
    return BudgetTracker(config, **{k:v for k,v in report.items()
                                  if k in {'calls','global_passes','tile_passes','retry_calls','by_source'}})
