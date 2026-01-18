from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from gp_retro_feas import FeasibleExecutor
from gp_retro_repr import RetrosynthesisStep, Route

from gp_core.budget import BudgetExceeded, BudgetedOneStepModel, CallCounter

from . import PlannerResult


def _leaf_fn(executor: FeasibleExecutor) -> Callable[[str], bool]:
    inv = executor.inventory
    if inv is None:
        return lambda _s: False
    return getattr(inv, "is_leaf", None) or inv.is_purchasable


def _pick_expand_index(molecule_set: List[str], *, is_leaf: Callable[[str], bool]) -> Optional[int]:
    """
    Pick which molecule to expand next.

    Default: expand the "largest" (by SMILES length) non-leaf; tie-break by smallest index.
    """
    candidates: List[Tuple[int, int]] = []
    for i, smi in enumerate(molecule_set):
        if not is_leaf(smi):
            candidates.append((len(smi or ""), i))
    if not candidates:
        return None
    # max length; then smallest index
    _, idx = max(candidates, key=lambda t: (t[0], -t[1]))
    return int(idx)


def run_greedy(
    target_smiles: str,
    *,
    executor: FeasibleExecutor,
    max_depth: int,
    max_calls: int,
) -> PlannerResult:
    """
    Greedy planner: repeatedly expand one frontier molecule using the top-1 ranked one-step candidate.

    Budget: `max_calls` counts one-step *inference calls* (cache hits do not count when detectable).
    """
    if executor.one_step_model is None:
        raise ValueError("executor.one_step_model is required for greedy baseline")

    counter = CallCounter()
    t0 = time.perf_counter()

    # Ensure per-run fairness: do not share one-step cache across planners/targets.
    cache = getattr(executor.one_step_model, "_cache", None)
    if isinstance(cache, dict):
        cache.clear()

    budgeted = BudgetedOneStepModel(executor.one_step_model, counter=counter, max_calls=int(max_calls))
    one_step_topk = int(getattr(executor.policy, "one_step_topk", 10))
    is_leaf = _leaf_fn(executor)

    route = Route()
    molecule_set: List[str] = [str(target_smiles)]

    stop_reason = "max_depth"
    one_step_error_count = 0
    last_error: Optional[str] = None
    try:
        for _depth in range(int(max_depth)):
            if molecule_set and all(is_leaf(m) for m in molecule_set):
                stop_reason = "solved"
                break

            idx = _pick_expand_index(molecule_set, is_leaf=is_leaf)
            if idx is None:
                stop_reason = "no_expandable"
                break

            product = molecule_set[idx]
            try:
                preds = budgeted.predict(product, topk=one_step_topk)
            except BudgetExceeded:
                raise
            except Exception as e:
                one_step_error_count += 1
                last_error = f"{type(e).__name__}:{e}"
                cache = getattr(budgeted.base_model, "_cache", None)
                if isinstance(cache, dict):
                    cache[(str(product), int(one_step_topk))] = []
                stop_reason = "one_step_error"
                break
            if executor.one_step_ranker is not None:
                preds = list(executor.one_step_ranker(product, list(preds)))
            if not preds:
                stop_reason = "no_candidates"
                break

            chosen = None
            chosen_rank = None
            for r, cand in enumerate(preds):
                reactants = [str(x) for x in (cand.reactants or []) if str(x)]
                if not reactants:
                    continue
                if executor.policy.max_reactants is not None and len(reactants) > int(executor.policy.max_reactants):
                    continue
                if executor.policy.require_all_purchasable and executor.inventory is not None:
                    if not all(executor.inventory.is_purchasable(s) for s in reactants):
                        continue
                chosen = cand
                chosen_rank = int(r)
                break

            if chosen is None or chosen_rank is None:
                stop_reason = "all_candidates_rejected"
                break

            updated = [m for i, m in enumerate(molecule_set) if i != idx] + list(chosen.reactants)
            model_name = getattr(budgeted, "name", "one_step")
            original_rank = (getattr(chosen, "meta", {}) or {}).get("rank", chosen_rank)

            step = RetrosynthesisStep(
                molecule_set=molecule_set.copy(),
                rational="greedy",
                product=product,
                template_id=f"{model_name}@rank={original_rank}",
                reactants=list(chosen.reactants),
                updated_molecule_set=updated,
                diagnostics={
                    "executor_reason": "ok",
                    "one_step_score": getattr(chosen, "score", None),
                    "one_step_meta": dict(getattr(chosen, "meta", {}) or {}),
                    "one_step_step_score": (getattr(chosen, "meta", {}) or {}).get("step_score"),
                    "one_step_log_score_single": (getattr(chosen, "meta", {}) or {}).get("log_score_single"),
                    "one_step_log_p_fwd": (getattr(chosen, "meta", {}) or {}).get("step_meta", {}).get("log_p_fwd"),
                    "one_step_requested_rank": 0,
                    "one_step_chosen_rank": chosen_rank,
                    "one_step_original_rank": original_rank,
                    "one_step_n_candidates": len(preds),
                },
            )
            route.append(step)
            molecule_set = updated

        else:
            stop_reason = "max_depth"
    except BudgetExceeded:
        stop_reason = "budget_exceeded"

    counter.stop()
    wall_time = float(time.perf_counter() - t0)
    solved = bool(molecule_set) and all(is_leaf(m) for m in molecule_set)
    return PlannerResult(
        solved=solved,
        route=route,
        steps=len(route.steps),
        calls_used=int(counter.one_step_calls),
        wall_time=wall_time,
        extra={
            "stop_reason": stop_reason,
            "error": last_error,
            "one_step_error_count": int(one_step_error_count),
            "final_molecule_set": list(molecule_set),
        },
    )

