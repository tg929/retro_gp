"""GP search loop with metrics and rank-based selection."""
import copy
import random
from typing import Any, Dict, List, Optional

from gp_retro_feas import FeasibleExecutor
from gp_retro_obj import nsga2_survivor_selection

from . import config
from .budget import BudgetExceeded
from .executor import evaluate_program
from .program_ops import (
    random_program,
    crossover_one_point,
    mutate_program,
    program_from_genes,
)
from .metrics import MetricsHistory


def rank_parents(population: List[Dict[str, Any]], method: str = "invrank") -> List[Dict[str, Any]]:
    pop_sorted = sorted(population, key=lambda ind: ind["fitness"].scalar, reverse=True)
    scores = [ind["fitness"].scalar for ind in pop_sorted]
    if method == "fitness":
        weights = scores
    else:
        weights = [1 / (i + 1) for i in range(len(pop_sorted))]
    total = sum(weights) or 1.0
    probs = [w / total for w in weights]
    # return list of sampled parents same length as population
    return random.choices(pop_sorted, weights=probs, k=len(pop_sorted))


def run_gp_for_target(
    target: str,
    inventory,
    reg,
    evaluator,
    exe: FeasibleExecutor,
    pop_size: int = config.pop_size,
    generations: int = config.generations,
    p_crossover: float = config.pcross,
    p_mutation: float = config.pmut,
    seed: int = config.seed,
    template_pool: Optional[List[str]] = None,
    init_templates: Optional[List[str]] = None,  # New: specific templates for initialization
    history: Optional[MetricsHistory] = None,
    nonempty_bonus: float = config.nonempty_bonus,
    feasible_templates_for_target: Optional[List[str]] = None,
    allow_model_actions: bool = False,
    model_rank_pool: Optional[List[int]] = None,
    p_model_action: float = 0.0,
):
    random.seed(seed)
    template_pool = template_pool or list(reg.templates.keys())
    # If init_templates not provided, fallback to full pool
    init_templates = init_templates or template_pool

    senses = {k: spec.direction() for k, spec in evaluator.specs.items()}
    objective_keys = list(evaluator.specs.keys())

    hist = history or MetricsHistory()

    def _finalize(pop: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], MetricsHistory]:
        pop.sort(key=lambda ind: ind["fitness"].scalar, reverse=True)
        if pop:
            for smi, score in [(ind["route"].to_json(), ind["fitness"].scalar) for ind in pop[:3]]:
                hist.commit(smi, score)
        hist.proposals += 1
        return pop, hist

    population: List[Dict[str, Any]] = []

    # --- Inject Greedy/Baseline Seeds ---
    if allow_model_actions and model_rank_pool and 0 in model_rank_pool:
        # Create a program with all ranks = 0 (Greedy baseline)
        greedy_genes = [("N", 0) for _ in range(config.max_templates_per_prog)]
        greedy_prog = program_from_genes(greedy_genes)
        try:
            ind = evaluate_program(greedy_prog, exe, evaluator, target)
            if nonempty_bonus and getattr(ind["route"], "steps", []):
                ind["fitness"].scalar += nonempty_bonus
            population.append(ind)
        except BudgetExceeded:
            print("[budget] one-step calls budget exceeded during GP init (seeding); stopping.")
            return _finalize(population)

    while len(population) < pop_size:
        # Use init_templates for the first random programs to boost start
        # But subsequent mutations will use the full template_pool
        prog = random_program(
            init_templates if random.random() < 0.8 else template_pool,
            min_len=1,
            max_len=config.max_templates_per_prog,
            allow_model_actions=allow_model_actions,
            model_rank_pool=model_rank_pool,
            p_model_action=p_model_action,
        )
        try:
            ind = evaluate_program(prog, exe, evaluator, target)
        except BudgetExceeded:
            print("[budget] one-step calls budget exceeded during GP init; stopping early.")
            return _finalize(population)
        if nonempty_bonus and getattr(ind["route"], "steps", []):
            ind["fitness"].scalar += nonempty_bonus
        population.append(ind)

    for gen in range(1, generations + 1):
        scalars = [ind["fitness"].scalar for ind in population]
        solved_count = sum(1 for ind in population if ind["fitness"].objectives.get("solved", 0) > 0.5)
        best = max(population, key=lambda ind: ind["fitness"].scalar)

        print(
            f"Gen {gen:02d} solved: {solved_count}/{len(population)} "
            f"best: {best['fitness'].scalar:.3f} mean: {sum(scalars)/len(scalars):.3f}"
        )

        parents = rank_parents(population, method="invrank")

        offspring: List[Dict[str, Any]] = []
        i = 0
        while len(offspring) < pop_size:
            if random.random() < p_crossover and i + 1 < len(parents):
                p1 = parents[i]["program"]
                p2 = parents[i + 1]["program"]
                c1, c2 = crossover_one_point(p1, p2)
                i += 2
                children = [c1, c2]
            else:
                p0 = parents[i % len(parents)]["program"]
                children = [copy.deepcopy(p0)]
                i += 1

            new_children = []
            for ch in children:
                if random.random() < p_mutation:
                    ch = mutate_program(
                        ch,
                        template_pool,
                        feasible_templates=feasible_templates_for_target,
                        allow_model_actions=allow_model_actions,
                        model_rank_pool=model_rank_pool,
                        p_model_action=p_model_action,
                    )
                new_children.append(ch)

            for ch in new_children:
                try:
                    ind = evaluate_program(ch, exe, evaluator, target)
                except BudgetExceeded:
                    print("[budget] one-step calls budget exceeded during GP evolution; stopping early.")
                    return _finalize(population + offspring)
                if nonempty_bonus and getattr(ind["route"], "steps", []):
                    ind["fitness"].scalar += nonempty_bonus
                offspring.append(ind)
                if len(offspring) >= pop_size:
                    break

        combined = population + offspring
        population = nsga2_survivor_selection(combined, k=pop_size, senses=senses, objective_keys=objective_keys)

    return _finalize(population)
