# run_obj_demo.py
# 验证 gp_retro_obj 模块 + 和 gp_retro_repr / gp_retro_feas 的整合使用

import os
import sys
import traceback

import numpy as np
from scscore.scscore.standalone_model_numpy import SCScorer

from gp_retro_repr import Program, Select, ApplyTemplate, Stop
from gp_retro_feas import FeasibleExecutor
from gp_retro_obj import (
    RouteFitnessEvaluator,
    epsilon_lexicase_select,
    nsga2_survivor_selection,
)
from demo_utils import build_world_t1, build_objectives_default


print(">>> run_obj_demo.py: starting ...")


# ---------------------------------------------------------------------------
# 0. SCScore 加载：不使用环境变量，直接指定模型目录
# ---------------------------------------------------------------------------

SC_MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "scscore",
    "models",
    "full_reaxys_model_1024bool",
)
SC_FP_LENGTH = 1024
_scscore_model = None


def build_scscore_fn():
    """
    返回 scscore_fn(smiles) -> float
    通过 SCScorer.restore(model_dir, 1024) 直接加载模型。
    """
    global _scscore_model
    if _scscore_model is None:
        model = SCScorer()
        model.restore(SC_MODEL_DIR, SC_FP_LENGTH)
        _scscore_model = model

    def _score(smiles: str) -> float:
        if not smiles:
            return 5.0
        out = _scscore_model.get_score_from_smi(smiles)
        # 原始返回 ('CCO', array([...])) 之类，取均值
        if isinstance(out, (tuple, list)) and len(out) == 2:
            _, arr = out
        else:
            arr = out
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 0:
            return float(arr)
        return float(arr.mean())

    return _score


# ---------------------------------------------------------------------------
# 1. 定义一个简单的“审计函数”给 RouteFitnessEvaluator 用
# ---------------------------------------------------------------------------
def make_audit_fn(stock, target_smiles: str):
    """
    RouteFitnessEvaluator 期望有一个 audit_fn(route)，返回：
        is_solved, first_invalid_molecule_set, current_molecule_set,
        n_steps, n_valid_steps
    """
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


# ---------------------------------------------------------------------------
# 2. 主流程
# ---------------------------------------------------------------------------
def main():
    print(">>> main() entered in run_obj_demo.py")

    # 世界 & 可行性执行器
    stock, reg, target = build_world_t1()
    exe = FeasibleExecutor(reg, inventory=stock)
    print(f"Target SMILES: {target}")
    print(f"Stock molecules: {list(stock)}")

    # === 初始化 SCScore 函数（不依赖环境变量） ===
    sc_fn = build_scscore_fn()

    # 两条示例 DP 程序
    prog_good = Program([Select(0), ApplyTemplate("T1", rational="good_demo"), Stop()])
    prog_bad = Program([Stop()])
    programs = [("good_route", prog_good), ("bad_route", prog_bad)]

    # 多目标配置 & 适应度评估器
    specs = build_objectives_default()
    audit_fn = make_audit_fn(stock, target_smiles=target)

    evaluator = RouteFitnessEvaluator(
        objective_specs=specs,
        purchasable_fn=stock.is_purchasable,
        audit_fn=audit_fn,
        scscore_fn=sc_fn,   # ★ 用我们刚刚构造好的 scscore_fn
        target_smiles=target,
    )

    population = []

    for name, prog in programs:
        print("\n" + "=" * 80)
        print(f"[{name}] 执行 DP 程序")

        route = exe.execute(prog, target_smiles=target)
        print("Route JSON:")
        print(route.to_json())
        print("Route is_solved(stock):", route.is_solved(stock))

        fit = evaluator.evaluate(route)
        print("\nObjectives:")
        for k, v in fit.objectives.items():
            print(f"  {k:18s}: {v:.4f}")
        print("Scalar fitness:", fit.scalar)

        population.append(
            {"name": name, "program": prog, "route": route, "fitness": fit}
        )

    senses = {k: spec.direction() for k, spec in specs.items()}
    objective_keys = list(specs.keys())

    print("\n" + "=" * 80)
    print("[epsilon-lexicase] 选择父代")
    parents = epsilon_lexicase_select(
        population,
        senses=senses,
        objective_keys=objective_keys,
        eps_quantile=0.10,
        n_parents=2,
    )
    print("被选中的父代个体：", [ind["name"] for ind in parents])

    print("\n[NSGA-II] 生存者选择 (k=1)")
    survivors = nsga2_survivor_selection(
        population,
        k=1,
        senses=senses,
        objective_keys=objective_keys,
    )
    print("保留下来的个体：", [ind["name"] for ind in survivors])

    print("\n>>> run_obj_demo.py finished normally.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
