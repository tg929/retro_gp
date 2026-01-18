from __future__ import annotations

"""
Unified budgeting / accounting utilities for "planner-only" comparisons.

Primary budget unit: the number of *actual* one-step model inference calls.
If the underlying one-step model has an internal cache, cache hits should not
consume budget.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable


class BudgetExceeded(RuntimeError):
    """Raised when a planner exceeds its allocated calls budget."""


@dataclass
class CallCounter:
    """Lightweight counter for planner budget accounting."""

    one_step_calls: int = 0
    template_apply_calls: int = 0
    route_evals: int = 0
    start_time: float = field(default_factory=time.perf_counter)
    wall_time: Optional[float] = None

    def stop(self) -> None:
        """Freeze `wall_time` (seconds) if not already set."""
        if self.wall_time is None:
            self.wall_time = float(time.perf_counter() - self.start_time)

    @property
    def elapsed(self) -> float:
        """Seconds since start (or frozen wall_time)."""
        return float(self.wall_time) if self.wall_time is not None else float(time.perf_counter() - self.start_time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "one_step_calls": int(self.one_step_calls),
            "template_apply_calls": int(self.template_apply_calls),
            "route_evals": int(self.route_evals),
            "wall_time": float(self.elapsed),
        }


@runtime_checkable
class OneStepModelLike(Protocol):
    """Minimal protocol compatible with gp_retro_nn.OneStepRetrosynthesisModel."""

    name: str

    def predict(self, product_smiles: str, topk: int):  # -> List[OneStepPrediction]
        raise NotImplementedError


class BudgetedOneStepModel:
    """
    Wrap a one-step model with a hard calls budget.

    Notes on counting:
    - If `base_model` exposes a dict-like `_cache` keyed by (product_smiles, topk),
      cache hits do NOT consume budget.
    - Otherwise, we conservatively count every wrapper-level `predict()` call.
    """

    def __init__(self, base_model: OneStepModelLike, *, counter: CallCounter, max_calls: int):
        if max_calls < 0:
            raise ValueError(f"max_calls must be >= 0, got {max_calls}")
        self.base_model = base_model
        self.counter = counter
        self.max_calls = int(max_calls)
        self.name = getattr(base_model, "name", "one_step")

    def _is_cache_hit(self, product_smiles: str, topk: int) -> bool:
        cache = getattr(self.base_model, "_cache", None)
        if isinstance(cache, dict):
            return (str(product_smiles), int(topk)) in cache
        return False

    def predict(self, product_smiles: str, topk: int):
        topk_i = int(topk)

        # Best-effort: do not charge budget for cache hits.
        cache_hit = self._is_cache_hit(product_smiles, topk_i)
        if not cache_hit:
            if int(self.counter.one_step_calls) >= int(self.max_calls):
                raise BudgetExceeded(
                    f"one-step calls budget exceeded: used={self.counter.one_step_calls} max={self.max_calls}"
                )
            self.counter.one_step_calls += 1

        return self.base_model.predict(str(product_smiles), topk=topk_i)

