"""Run GP search on real data using modular gp_core helpers."""
import sys
from pathlib import Path

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
    specs = build_objectives(config.objective_weights)
    hist = MetricsHistory()

    max_targets = 10
    n_targets = min(max_targets, len(targets))

    def _progress(current: int, total: int):
        """Print a simple progress bar to the real terminal (not the log)."""
        if total <= 0:
            return
        bar_width = 30
        frac = max(0.0, min(1.0, current / total))
        filled = int(bar_width * frac)
        bar = "#" * filled + "-" * (bar_width - filled)
        print(
            f"\rProgress: [{bar}] {current}/{total} targets",
            end="",
            file=sys.__stdout__,
            flush=True,
        )

    for ti, target in enumerate(targets[:n_targets]):
        _progress(ti, n_targets)
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
        for rank, ind in enumerate(population[:3], start=1):
            objs = ind["fitness"].objectives
            print(
                f"  [{rank}] scalar={ind['fitness'].scalar:.3f} "
                f"solved={bool(objs.get('solved', 0))} "
                f"route_len={objs.get('route_len', -1)}"
            )
            print(ind["route"].to_json())

    # 完成所有目标后，打印一次完整进度并换行
    _progress(n_targets, n_targets)
    print(file=sys.__stdout__)

    if hist.has_updates:
        stats = hist.metrics(budget=config.pop_size * config.generations)
        print("\nMetrics snapshot:", {k: round(v, 3) for k, v in stats.items()})


if __name__ == "__main__":
    # Mirror stdout/stderr to a time-stamped log file for reproducibility
    from datetime import datetime

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%m%d%H%M")
    log_path = log_dir / f"run_real_data_gp_output_{ts}.txt"

    # 将所有 print 输出重定向到日志文件；进度条单独写到 sys.__stdout__
    with log_path.open("w", encoding="utf-8") as _f:
        _orig_out, _orig_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = _f
        try:
            run()
        finally:
            sys.stdout, sys.stderr = _orig_out, _orig_err
