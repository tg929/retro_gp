from dataclasses import dataclass, field
from typing import Any, Dict

from gp_retro_repr import Route


@dataclass
class PlannerResult:
    solved: bool
    route: Route
    steps: int
    calls_used: int
    wall_time: float
    extra: Dict[str, Any] = field(default_factory=dict)
