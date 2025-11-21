from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional, Tuple
import math
import random

# Individual protocol: any object with `fitness.objectives: Dict[str, float]` and `fitness.scalar: float`
# Here we assume population is a list of dicts: {'route': ..., 'fitness': FitnessResult}

def _dominates(a: Dict[str, float], b: Dict[str, float],
               senses: Dict[str, int]) -> bool:
    """Return True if a Pareto-dominates b under the sense (+1 max, -1 min)."""
    not_worse = True
    strictly_better = False
    for k, s in senses.items():
        va = a.get(k)
        vb = b.get(k)
        if va is None or vb is None:
            continue
        # Convert to maximize by multiplying by `s`
        va *= s
        vb *= s
        if va < vb - 1e-12:
            not_worse = False
        if va > vb + 1e-12:
            strictly_better = True
    return not_worse and strictly_better

def _crowding_distance(front: List[Dict[str, Any]], keys: List[str], senses: Dict[str, int]) -> List[float]:
    """Compute crowding distance for a single front."""
    n = len(front)
    if n == 0:
        return []
    if n == 1:
        return [float('inf')]
    dist = [0.0] * n
    for k in keys:
        s = senses[k]
        sorted_idx = sorted(range(n), key=lambda i: s * front[i]['fitness'].objectives.get(k, 0.0))
        dist[sorted_idx[0]] = float('inf')
        dist[sorted_idx[-1]] = float('inf')
        minv = front[sorted_idx[0]]['fitness'].objectives.get(k, 0.0)
        maxv = front[sorted_idx[-1]]['fitness'].objectives.get(k, 0.0)
        denom = max(1e-12, (s * maxv) - (s * minv))
        for r in range(1, n-1):
            i = sorted_idx[r]
            v_prev = front[sorted_idx[r-1]]['fitness'].objectives.get(k, 0.0)
            v_next = front[sorted_idx[r+1]]['fitness'].objectives.get(k, 0.0)
            dist[i] += abs((s * v_next) - (s * v_prev)) / denom
    return dist

def nsga2_survivor_selection(population: List[Dict[str, Any]],
                             k: int,
                             senses: Dict[str, int],
                             objective_keys: List[str]) -> List[Dict[str, Any]]:
    """Standard NSGA-II non-dominated sorting with crowding distance."""
    # Fast non-dominated sort (simple O(n^2) implementation for clarity)
    remaining = list(population)
    fronts: List[List[Dict[str, Any]]] = []
    while remaining:
        front = []
        for i, ind in enumerate(remaining):
            dominated = False
            for j, other in enumerate(remaining):
                if j == i:
                    continue
                if _dominates(other['fitness'].objectives, ind['fitness'].objectives, senses):
                    dominated = True
                    break
            if not dominated:
                front.append(ind)
        fronts.append(front)
        # Remove front from remaining
        ids = set(id(x) for x in front)
        remaining = [x for x in remaining if id(x) not in ids]

    # Fill k
    survivors: List[Dict[str, Any]] = []
    for front in fronts:
        if len(survivors) + len(front) <= k:
            survivors.extend(front)
        else:
            # Crowding distance
            distances = _crowding_distance(front, objective_keys, senses)
            idx_sorted = sorted(range(len(front)), key=lambda i: distances[i], reverse=True)
            survivors.extend([front[i] for i in idx_sorted[:(k - len(survivors))]])
            break
    return survivors

def epsilon_lexicase_select(population: List[Dict[str, Any]],
                            senses: Dict[str, int],
                            objective_keys: List[str],
                            eps_quantile: float = 0.1,
                            n_parents: int = 1) -> List[Dict[str, Any]]:
    """(ε)-Lexicase parent selection.

    Randomly order objectives. For each objective, filter candidates to those within ε of the
    best (according to that objective). Continue until one remains or objectives are exhausted.
    If more than one remain, pick uniformly.

    eps is set per-objective as quantile(eps_quantile) of absolute differences in the population.
    """
    import random
    parents = []
    if not population:
        return parents
    # Precompute ε for each objective
    eps: Dict[str, float] = {}
    for k in objective_keys:
        s = senses[k]
        vals = [s * ind['fitness'].objectives.get(k, 0.0) for ind in population]
        if not vals:
            eps[k] = 0.0
        else:
            vals_sorted = sorted(vals, reverse=True)
            idx = max(0, min(len(vals_sorted) - 1, int(eps_quantile * (len(vals_sorted) - 1))))
            # epsilon = difference between best and the value at chosen quantile
            eps[k] = max(1e-12, vals_sorted[0] - vals_sorted[idx])

    for _ in range(n_parents):
        candidates = list(population)
        keys = objective_keys[:]
        random.shuffle(keys)
        for k in keys:
            s = senses[k]
            # Find best value among candidates for this objective (max-sense due to sign flip)
            best = max(s * ind['fitness'].objectives.get(k, 0.0) for ind in candidates)
            # Keep those within epsilon of best
            new_cands = [ind for ind in candidates
                         if (best - (s * ind['fitness'].objectives.get(k, 0.0))) <= eps[k] + 1e-12]
            if new_cands:
                candidates = new_cands
            if len(candidates) <= 1:
                break
        if candidates:
            parents.append(random.choice(candidates))
        else:
            parents.append(random.choice(population))
    return parents
