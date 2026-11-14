from .base import Request, Scheduler
from .fifo import FIFOScheduler
from .adaptive import AdaptiveScheduler

SCHEDULERS: dict[str, type[Scheduler]] = {
    FIFOScheduler.name: FIFOScheduler,
    AdaptiveScheduler.name: AdaptiveScheduler,
}


def make_scheduler(name: str, **params) -> Scheduler:
    """Build a scheduler by name, e.g. make_scheduler("fifo", max_concurrency=4)."""
    try:
        cls = SCHEDULERS[name]
    except KeyError:
        raise ValueError(f"unknown scheduler {name!r}; choices: {sorted(SCHEDULERS)}") from None
    return cls(**params)


__all__ = ["Request", "Scheduler", "FIFOScheduler", "AdaptiveScheduler", "SCHEDULERS", "make_scheduler"]
