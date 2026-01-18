from __future__ import annotations

import heapq
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
    return tuple(sorted(list(molecule_set)))


@dataclass
class _Node:
    molecule_set: List[str]
    route: Route

    @property
    def steps(self) -> int:
        return int(len(self.route.steps))


def _astar0_priority(node: _Node, *, is_leaf: Callable[[str], bool]) -> float:
    """A*-0 style: f = g + h, where g=steps, h=#unsolved molecules."""
    h = sum(1 for m in node.molecule_set if not is_leaf(m))
    return float(node.steps + h)


def run_best_first(
    target_smiles: str,
    *,
    executor: FeasibleExecutor,
    max_depth: int,
    max_calls: int,
    strategy: str = "astar0",
) -> PlannerResult:
    """
    Best-first / A*-like planner over molecule-set states, expanding with one-step candidates.

    strategy:
      - "astar0": f = steps_so_far + #unsolved_leaves
    """
    if executor.one_step_model is None:
        raise ValueError("executor.one_step_model is required for best-first baseline")

    counter = CallCounter()
    t0 = time.perf_counter()

    # Ensure per-run fairness: do not share one-step cache across planners/targets.
    cache = getattr(executor.one_step_model, "_cache", None)
    if isinstance(cache, dict):
        cache.clear()

    budgeted = BudgetedOneStepModel(executor.one_step_model, counter=counter, max_calls=int(max_calls))
    one_step_topk = int(getattr(executor.policy, "one_step_topk", 10))
    is_leaf = _leaf_fn(executor)

    if strategy != "astar0":
        raise ValueError(f"Unknown strategy: {strategy} (supported: astar0)")

    start = _Node(molecule_set=[str(target_smiles)], route=Route())
    if start.molecule_set and all(is_leaf(m) for m in start.molecule_set):
        counter.stop()
        return PlannerResult(
            solved=True,
            route=start.route,
            steps=0,
            calls_used=0,
            wall_time=float(time.perf_counter() - t0),
            extra={"stop_reason": "solved", "final_molecule_set": list(start.molecule_set)},
        )

    heap: List[Tuple[float, int, _Node]] = []
    push_id = 0
    heapq.heappush(heap, (_astar0_priority(start, is_leaf=is_leaf), push_id, start))
    push_id += 1

    # Keep best-known depth (g) per state to avoid trivial loops.
    best_g: Dict[Tuple[str, ...], int] = {_state_key(start.molecule_set): 0}
    best_partial: _Node = start
    best_partial_f: float = _astar0_priority(start, is_leaf=is_leaf)
    stop_reason = "exhausted"
    one_step_error_count = 0
    last_error: Optional[str] = None

    try:
        while heap:
            f, _pid, node = heapq.heappop(heap)
            if f < best_partial_f:
                best_partial = node
                best_partial_f = float(f)

            if node.molecule_set and all(is_leaf(m) for m in node.molecule_set):
                stop_reason = "solved"
                best_partial = node
                break

            if node.steps >= int(max_depth):
                continue

            idx = _pick_expand_index(node.molecule_set, is_leaf=is_leaf)
            if idx is None:
                continue
            product = node.molecule_set[idx]

            try:
                preds = budgeted.predict(product, topk=one_step_topk)
            except BudgetExceeded:
                raise
            except Exception as e:
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

                new_route = Route(steps=list(node.route.steps))
                model_name = getattr(budgeted, "name", "one_step")
                original_rank = (getattr(cand, "meta", {}) or {}).get("rank", int(r))
                step = RetrosynthesisStep(
                    molecule_set=list(node.molecule_set),
                    rational="best_first",
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
                new_route.append(step)
                child = _Node(molecule_set=updated, route=new_route)

                key = _state_key(updated)
                g = child.steps
                prev_g = best_g.get(key)
                if prev_g is not None and g >= int(prev_g):
                    continue
                best_g[key] = int(g)

                pf = _astar0_priority(child, is_leaf=is_leaf)
                heapq.heappush(heap, (pf, push_id, child))
                push_id += 1

        else:
            stop_reason = "exhausted"
    except BudgetExceeded:
        stop_reason = "budget_exceeded"

    counter.stop()
    wall_time = float(time.perf_counter() - t0)
    solved = bool(best_partial.molecule_set) and all(is_leaf(m) for m in best_partial.molecule_set)
    return PlannerResult(
        solved=bool(solved),
        route=best_partial.route,
        steps=len(best_partial.route.steps),
        calls_used=int(counter.one_step_calls),
        wall_time=wall_time,
        extra={
            "stop_reason": stop_reason,
            "error": last_error,
            "one_step_error_count": int(one_step_error_count),
            "final_molecule_set": list(best_partial.molecule_set),
            "best_f": float(best_partial_f),
        },
    )
