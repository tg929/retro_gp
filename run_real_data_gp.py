"""Run GP search on real data using modular gp_core helpers."""
from gp_core import config
from gp_core.data_loading import load_world_from_data
from gp_core.templates import template_ids
from gp_core.executor import make_executor
from gp_core.fitness import build_objectives, build_scscore_fn, make_evaluator
from gp_core.search import run_gp_for_target
from gp_core.metrics import MetricsHistory
from gp_retro_obj import RouteFitnessEvaluator
from gp_retro_feas import FeasibleExecutor, ActionMaskBuilder


def make_audit_fn(stock, target_smiles: str):
    def audit_route(route):
        if route.steps:
            final_set = route.steps[-1].updated_molecule_set
            n_steps = len(route.steps)
        else:
            final_set = [target_smiles]
            n_steps = 0
        is_solved = all(stock.is_purchasable(m) for m in final_set)
        return {
            "is_solved": is_solved,
            "first_invalid_molecule_set": [] if is_solved else list(final_set),
            "current_molecule_set": list(final_set),
            "n_steps": n_steps,
            "n_valid_steps": n_steps,
        }
    return audit_route


def run():
    inventory, reg, targets = load_world_from_data(limit_targets=None)
    sc_fn = build_scscore_fn()
    specs = build_objectives(config.OBJECTIVE_WEIGHTS)
    hist = MetricsHistory()

    for ti, target in enumerate(targets[:3]):
        print(f"\n=== Target {ti+1}: {target} ===")

        # 针对当前目标先做一次可行动作掩码，避免全部落空
        mask = ActionMaskBuilder(reg, inventory=inventory).build(target)
        # 修复：template_pool 必须是全量模板，否则后续步骤无法进行
        full_pool = template_ids(reg)
        
        # 仅将可行模板用于初始化种群
        init_pool = mask.feasible_templates or full_pool

        if mask.feasible_templates:
            print(f"Feasible templates for target (init only): {len(mask.feasible_templates)} (of {len(mask.candidate_templates)})")
        else:
            print("No feasible templates found; fallback to full template pool.")

        exe: FeasibleExecutor = make_executor(reg, inventory)
        audit_fn = make_audit_fn(inventory, target)
        evaluator: RouteFitnessEvaluator = make_evaluator(specs, inventory, audit_fn, sc_fn, target)

        population, hist = run_gp_for_target(
            target=target,
            inventory=inventory,
            reg=reg,
            evaluator=evaluator,
            exe=exe,
            template_pool=full_pool,     # 搜索空间：全量
            init_templates=init_pool,    # 初始化偏好：局部可行
            history=hist,
        )

        print("Top solutions:")
        for rank, ind in enumerate(population[:100], start=1):
            objs = ind["fitness"].objectives
            print(
                f"  [{rank}] scalar={ind['fitness'].scalar:.3f} "
                f"solved={bool(objs.get('solved', 0))} "
                f"route_len={objs.get('route_len', -1)}"
            )
            print(ind["route"].to_json())

    if hist.has_updates:
        stats = hist.metrics(budget=config.POP_SIZE * config.GENERATIONS)
        print("\nMetrics snapshot:", {k: round(v, 3) for k, v in stats.items()})


if __name__ == "__main__":
    run()
