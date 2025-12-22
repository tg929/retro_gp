"""Feasible execution wrapper."""
from typing import Any, Dict, Optional, Callable, List

from gp_retro_feas import FeasibleExecutor, ExecutePolicy
from gp_retro_obj import RouteFitnessEvaluator, FitnessResult
from gp_retro_repr import Program, Stop

try:  # optional dependency
    from gp_retro_nn import OneStepRetrosynthesisModel
except Exception:  # pragma: no cover
    OneStepRetrosynthesisModel = None  # type: ignore


def make_executor(
    reg,
    inventory,
    policy: Optional[ExecutePolicy] = None,
    one_step_model: Optional["OneStepRetrosynthesisModel"] = None,
    one_step_ranker: Optional[Callable[[str, List[Any]], List[Any]]] = None,
) -> FeasibleExecutor:
    # Ensure a default policy is always present (policy may be None from callers).
    return FeasibleExecutor(
        reg,
        inventory=inventory,
        policy=policy or ExecutePolicy(),
        one_step_model=one_step_model,
        one_step_ranker=one_step_ranker,
    )


def evaluate_program(
    prog: Program,
    exe: FeasibleExecutor,
    evaluator: RouteFitnessEvaluator,
    target: str,
) -> Dict[str, Any]:
    invalid = False
    try:
        route = exe.execute(prog, target_smiles=target)
    except Exception as e:
        invalid = True
        safe_prog = Program([Stop()])
        route = exe.execute(safe_prog, target_smiles=target)
    try:
        fit = evaluator.evaluate(route, program=prog, invalid=invalid)  # type: ignore[arg-type]
    except TypeError:
        fit = evaluator.evaluate(route)
        if invalid:
            fit.extra = dict(getattr(fit, "extra", {}) or {})
            fit.extra["invalid"] = True
            fit.extra["invalid_error"] = f"{type(e).__name__}:{e}"
            fit.scalar = float(getattr(evaluator, "invalid_scalar", -1e9))  # type: ignore[attr-defined]
    return {"program": prog, "route": route, "fitness": fit}
