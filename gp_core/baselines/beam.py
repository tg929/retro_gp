from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

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
    candidates: List[Tuple[int, int]] = []
    for i, smi in enumerate(molecule_set):
        if not is_leaf(smi):
            candidates.append((len(smi or ""), i))
    if not candidates:
        return None
    _, idx = max(candidates, key=lambda t: (t[0], -t[1]))
    return int(idx)


def _state_key(molecule_set: List[str]) -> Tuple[str, ...]:
    # order-insensitive but multiplicity-preserving key
    return tuple(sorted(list(molecule_set)))


def _default_state_score(route: Route, molecule_set: List[str], *, is_leaf: Callable[[str], bool]) -> float:
    unsolved = sum(1 for m in molecule_set if not is_leaf(m))
    # higher is better: prefer fewer unsolved and fewer steps
    return -float(unsolved) - 0.1 * float(len(route.steps))


@dataclass
class _BeamNode:
    molecule_set: List[str]
    route: Route
    score: float


def run_beam(
    target_smiles: str,
    *,
    beam_width: int,
    executor: FeasibleExecutor,
    max_depth: int,
    max_calls: int,
    state_score_fn: Optional[Callable[[Route], float]] = None,
) -> PlannerResult:
    """
    Beam search baseline over molecule-set states, expanding with one-step candidates.

    One-step candidate generation uses executor.one_step_model + executor.one_step_ranker
    (typically StepScorer.rank_and_truncate) for consistent filtering/scoring.
    """
    if beam_width <= 0:
        raise ValueError(f"beam_width must be > 0, got {beam_width}")
    if executor.one_step_model is None:
        raise ValueError("executor.one_step_model is required for beam baseline")

    counter = CallCounter()
    t0 = time.perf_counter()

    # Ensure per-run fairness: do not share one-step cache across planners/targets.
    cache = getattr(executor.one_step_model, "_cache", None)
    if isinstance(cache, dict):
        cache.clear()

    budgeted = BudgetedOneStepModel(executor.one_step_model, counter=counter, max_calls=int(max_calls))
    one_step_topk = int(getattr(executor.policy, "one_step_topk", 10))
    is_leaf = _leaf_fn(executor)

    def score_state(route: Route, molecule_set: List[str]) -> float:
        if state_score_fn is not None:
            return float(state_score_fn(route))
        return float(_default_state_score(route, molecule_set, is_leaf=is_leaf))

    best: Optional[_BeamNode] = None
    solved_best: Optional[_BeamNode] = None
    stop_reason = "max_depth"
    one_step_error_count = 0
    last_error: Optional[str] = None

    beam: List[_BeamNode] = [_BeamNode(molecule_set=[str(target_smiles)], route=Route(), score=0.0)]
    beam[0].score = score_state(beam[0].route, beam[0].molecule_set)

    try:
        for _depth in range(int(max_depth) + 1):
            # Check solved in current beam (allows 0-step solution if already a leaf).
            solved_nodes = [n for n in beam if n.molecule_set and all(is_leaf(m) for m in n.molecule_set)]
            if solved_nodes:
                solved_best = min(solved_nodes, key=lambda n: (len(n.route.steps), -n.score))
                stop_reason = "solved"
                break

            if _depth >= int(max_depth):
                stop_reason = "max_depth"
                break

            # Track best partial
            for n in beam:
                if best is None or n.score > best.score:
                    best = n

            all_children: Dict[Tuple[str, ...], _BeamNode] = {}
            for node in beam:
                idx = _pick_expand_index(node.molecule_set, is_leaf=is_leaf)
                if idx is None:
                    continue
                product = node.molecule_set[idx]

                try:
                    preds = budgeted.predict(product, topk=one_step_topk)
                except BudgetExceeded:
                    raise
                except Exception as e:
                    # Some molecules can trigger upstream one-step failures (e.g., preprocessing assertions).
                    one_step_error_count += 1
                    last_error = f"{type(e).__name__}:{e}"
                    # Best-effort negative cache to avoid repeating deterministic failures.
                    cache = getattr(budgeted.base_model, "_cache", None)
                    if isinstance(cache, dict):
                        cache[(str(product), int(one_step_topk))] = []
                    continue
                if executor.one_step_ranker is not None:
                    preds = list(executor.one_step_ranker(product, list(preds)))
                if not preds:
                    continue

                for r, cand in enumerate(preds):
                    reactants = [str(x) for x in (cand.reactants or []) if str(x)]
                    if not reactants:
                        continue
                    if executor.policy.max_reactants is not None and len(reactants) > int(executor.policy.max_reactants):
                        continue
                    if executor.policy.require_all_purchasable and executor.inventory is not None:
                        if not all(executor.inventory.is_purchasable(s) for s in reactants):
                            continue

                    updated = [m for i, m in enumerate(node.molecule_set) if i != idx] + reactants
                    model_name = getattr(budgeted, "name", "one_step")
                    original_rank = (getattr(cand, "meta", {}) or {}).get("rank", int(r))

                    step = RetrosynthesisStep(
                        molecule_set=list(node.molecule_set),
                        rational="beam",
                        product=product,
                        template_id=f"{model_name}@rank={original_rank}",
                        reactants=list(reactants),
                        updated_molecule_set=list(updated),
                        diagnostics={
                            "executor_reason": "ok",
                            "one_step_score": getattr(cand, "score", None),
                            "one_step_meta": dict(getattr(cand, "meta", {}) or {}),
                            "one_step_step_score": (getattr(cand, "meta", {}) or {}).get("step_score"),
                            "one_step_log_score_single": (getattr(cand, "meta", {}) or {}).get("log_score_single"),
                            "one_step_log_p_fwd": (getattr(cand, "meta", {}) or {}).get("step_meta", {}).get("log_p_fwd"),
                            "one_step_requested_rank": int(r),
                            "one_step_chosen_rank": int(r),
                            "one_step_original_rank": original_rank,
                            "one_step_n_candidates": len(preds),
                        },
                    )
                    new_route = Route(steps=list(node.route.steps))
                    new_route.append(step)
                    sc = score_state(new_route, updated)
                    child = _BeamNode(molecule_set=updated, route=new_route, score=sc)

                    k = _state_key(updated)
                    prev = all_children.get(k)
                    if prev is None or child.score > prev.score:
                        all_children[k] = child

            if not all_children:
                stop_reason = "no_children"
                break

            # Select next beam
            beam = sorted(all_children.values(), key=lambda n: n.score, reverse=True)[: int(beam_width)]

        # end for
    except BudgetExceeded:
        stop_reason = "budget_exceeded"

    counter.stop()
    wall_time = float(time.perf_counter() - t0)

    out_node = solved_best or best or (beam[0] if beam else _BeamNode([str(target_smiles)], Route(), 0.0))
    solved = solved_best is not None and bool(out_node.molecule_set) and all(is_leaf(m) for m in out_node.molecule_set)
    return PlannerResult(
        solved=bool(solved),
        route=out_node.route,
        steps=len(out_node.route.steps),
        calls_used=int(counter.one_step_calls),
        wall_time=wall_time,
        extra={
            "stop_reason": stop_reason,
            "error": last_error,
            "one_step_error_count": int(one_step_error_count),
            "final_molecule_set": list(out_node.molecule_set),
            "best_score": float(out_node.score),
        },
    )
