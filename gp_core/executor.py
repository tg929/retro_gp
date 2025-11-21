"""Feasible execution wrapper."""
from typing import Any, Dict, Optional

from gp_retro_feas import FeasibleExecutor, ExecutePolicy
from gp_retro_obj import RouteFitnessEvaluator
from gp_retro_repr import Program, Stop


def make_executor(reg, inventory, policy: Optional[ExecutePolicy] = None) -> FeasibleExecutor:
    # Ensure a default policy is always present (policy may be None from callers).
    return FeasibleExecutor(reg, inventory=inventory, policy=policy or ExecutePolicy())


def evaluate_program(
    prog: Program,
    exe: FeasibleExecutor,
    evaluator: RouteFitnessEvaluator,
    target: str,
) -> Dict[str, Any]:
    try:
        route = exe.execute(prog, target_smiles=target)
    except Exception:
        safe_prog = Program([Stop()])
        route = exe.execute(safe_prog, target_smiles=target)
    fit = evaluator.evaluate(route)
    return {"program": prog, "route": route, "fitness": fit}
