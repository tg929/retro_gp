from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Callable, List, Tuple, Optional, Any

@dataclass
class ObjectiveSpec:
    """Specification for one objective.

    Attributes
    ----------
    name: str
        Objective name (unique key).
    sense: str
        'max' or 'min' (direction to optimize).
    weight: float
        Weight for scalarization (if used).
    target: Optional[float]
        Optional target value. Some scalarizations (Tchebycheff) can use a reference/aspiration point.
    lower_is_better: Optional[bool]
        Convenience flag mirroring 'sense'. If provided, overrides 'sense' for backward-compatibility.
    """
    name: str
    sense: str
    weight: float = 1.0
    target: Optional[float] = None
    lower_is_better: Optional[bool] = None

    def direction(self) -> int:
        """Return +1 for maximize, -1 for minimize."""
        if self.lower_is_better is not None:
            return -1 if self.lower_is_better else +1
        s = (self.sense or '').lower()
        return +1 if s in ('max', 'maximize', 'asc') else -1

Objective = Dict[str, float]
ObjectiveVector = Dict[str, float]
