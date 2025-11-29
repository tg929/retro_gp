# run_gp_search_demo.py
# 一个“最小可跑”的多目标 GP 搜索主循环

import os
import random
import copy
import statistics
import sys
import traceback
from typing import List, Dict, Any, Tuple

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


# --------------------------------------------------------------------
# 0) SCScore 加载（不依赖环境变量）
# --------------------------------------------------------------------

SC_MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "scscore",
    "models",
    "full_reaxys_model_1024bool",
)
SC_FP_LENGTH = 1024
_scscore_model = None


def build_scscore_fn():
    global _scscore_model
    if _scscore_model is None:
        model = SCScorer()
        model.restore(SC_MODEL_DIR, SC_FP_LENGTH)
        _scscore_model = model

    def _score(smiles: str) -> float:
        if not smiles:
            return 5.0
        out = _scscore_model.get_score_from_smi(smiles)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            _, arr = out
        else:
            arr = out
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 0:
            return float(arr)
        return float(arr.mean())

    return _score


# --------------------------------------------------------------------
# 1) 审计函数
# --------------------------------------------------------------------
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


# --------------------------------------------------------------------
# 2) DP 程序编码与操作（模板序列 <-> Program）
# --------------------------------------------------------------------
# def templates_of_program(prog: Program) -> List[str]:
#     tids = []
#     for step in prog.steps:
#         if isinstance(step, ApplyTemplate):
#             tids.append(step.template_id)
#     return tids
def templates_of_program(prog: Program) -> List[str]:
    """从 Program 中抽取出所有 ApplyTemplate 的 template_id 序列。"""
    tids: List[str] = []
    for instr in prog.instructions:   # ← 关键：用 instructions，而不是 steps
        if isinstance(instr, ApplyTemplate):
            tids.append(instr.template_id)
    return tids



# def program_from_templates(template_ids: List[str]) -> Program:
#     steps = [Select(0)]
#     for tid in template_ids:
#         steps.append(ApplyTemplate(tid, rational="gp"))
#     steps.append(Stop())
#     return Program(steps)
def program_from_templates(template_ids):
    steps = []
    for tid in template_ids:
        steps.append(Select(0))                     # 每一步都先 Select
        steps.append(ApplyTemplate(tid, rational="gp"))
    steps.append(Stop())
    return Program(steps)



def random_program(template_pool: List[str], min_len=0, max_len=3) -> Program:
    k = random.randint(min_len, max_len)
    tids = [random.choice(template_pool) for _ in range(k)]
    return program_from_templates(tids)


def crossover_one_point(p1: Program, p2: Program) -> Tuple[Program, Program]:
    t1 = templates_of_program(p1)
    t2 = templates_of_program(p2)
    c1 = random.randint(0, len(t1))
    c2 = random.randint(0, len(t2))
    child1 = program_from_templates(t1[:c1] + t2[c2:])
    child2 = program_from_templates(t2[:c2] + t1[c1:])
    return child1, child2


def mutate_program(
    p: Program,
    template_pool: List[str],
    p_insert=0.40,
    p_delete=0.30,
    p_modify=0.30,
    max_total_len=5,
) -> Program:
    t = templates_of_program(p)
    op = random.random()
    if op < p_insert:
        if len(t) < max_total_len:
            pos = random.randint(0, len(t))
            t.insert(pos, random.choice(template_pool))
    elif op < p_insert + p_delete:
        if len(t) > 0:
            pos = random.randrange(len(t))
            t.pop(pos)
    else:
        if len(t) > 0:
            pos = random.randrange(len(t))
            t[pos] = random.choice(template_pool)
    return program_from_templates(t)


# --------------------------------------------------------------------
# 3) 个体评估：Program -> Route -> Fitness
# --------------------------------------------------------------------
# def evaluate_program(
#     prog: Program,
#     exe: FeasibleExecutor,
#     evaluator: RouteFitnessEvaluator,
#     target: str,
# ) -> Dict[str, Any]:
#     route = exe.execute(prog, target_smiles=target)
#     fit = evaluator.evaluate(route)
#     return {"program": prog, "route": route, "fitness": fit}

# def evaluate_program(
#     prog: Program,
#     exe: FeasibleExecutor,
#     evaluator: RouteFitnessEvaluator,
#     target: str,
# ) -> Dict[str, Any]:
#     """
#     执行 DP 程序并计算适应度。
#     如果程序结构非法导致 FeasibleExecutor 抛出异常，
#     则把该个体视为“失败路线”（用一个空路线代替），
#     保证 GP 循环不会因为坏个体直接崩溃。
#     """
#     try:
#         route = exe.execute(prog, target_smiles=target)
#     except Exception as e:
#         # 打印一下方便调试（可选）
#         # print(f"[WARN] executor failed for program: {prog}, err={e}")

#         # 构造一个“空程序”作为失败个体：只包含 Stop
#         safe_prog = Program([Stop()])

#         try:
#             route = exe.execute(safe_prog, target_smiles=target)
#         except Exception:
#             # 理论上不会再失败；保险起见，构造一个最小的“空壳”对象
#             class DummyRoute:
#                 def __init__(self, target_smiles):
#                     self.steps = []
#                     self.target_smiles = target_smiles

#                 def to_json(self):
#                     return "[]"

#                 def is_solved(self, stock):
#                     return False

#             route = DummyRoute(target)

#     fit = evaluator.evaluate(route)
#     return {"program": prog, "route": route, "fitness": fit}
def evaluate_program(
    prog: Program,
    exe: FeasibleExecutor,
    evaluator: RouteFitnessEvaluator,
    target: str,
) -> Dict[str, Any]:
    """
    执行 DP 程序并计算适应度。
    如果程序结构非法导致 FeasibleExecutor 抛出异常，
    则把该个体视为“失败路线”（用一个空路线代替），
    保证 GP 循环不会因为坏个体直接崩溃。
    """
    try:
        route = exe.execute(prog, target_smiles=target)
    except Exception as e:
        # 打印错误信息，方便定位是哪一步出问题（例如 rdchiral 缺失、Program 语法非法等）
        print(f"[WARN] executor failed for program: {prog}, err={e}")

        # 构造一个“空程序”作为失败个体：只包含 Stop
        safe_prog = Program([Stop()])

        try:
            route = exe.execute(safe_prog, target_smiles=target)
        except Exception:
            # 理论上不会再失败；保险起见，构造一个最小的“空壳”对象
            class DummyRoute:
                def __init__(self, target_smiles):
                    self.steps = []
                    self.target_smiles = target_smiles

                def to_json(self):
                    return "[]"

                def is_solved(self, stock):
                    return False

            route = DummyRoute(target)

    fit = evaluator.evaluate(route)
    return {"program": prog, "route": route, "fitness": fit}



# --------------------------------------------------------------------
# 4) GP 主循环
# --------------------------------------------------------------------
def run_gp_search(
    pop_size=20,
    generations=10,
    p_crossover=0.7,
    p_mutation=0.4,
    seed=123,
):
    random.seed(seed)

    stock, reg, target = build_world_t1()
    exe = FeasibleExecutor(reg, inventory=stock)

    # === 初始化 SCScore 函数 ===
    sc_fn = build_scscore_fn()

    specs = build_objectives_default()
    evaluator = RouteFitnessEvaluator(
        objective_specs=specs,
        purchasable_fn=stock.is_purchasable,
        audit_fn=make_audit_fn(stock, target_smiles=target),
        scscore_fn=sc_fn,   # ★ 真正用上 SCScore
        target_smiles=target,
    )
    senses = {k: spec.direction() for k, spec in specs.items()}
    objective_keys = list(specs.keys())

    template_pool = list(reg.templates.keys())

    # 初始化种群
    population: List[Dict[str, Any]] = []
    for _ in range(pop_size):
        prog = random_program(template_pool, min_len=0, max_len=3)
        population.append(evaluate_program(prog, exe, evaluator, target))

    # 进化循环
    for gen in range(1, generations + 1):
        scalars = [ind["fitness"].scalar for ind in population]
        solved_count = sum(
            1 for ind in population
            if ind["fitness"].objectives.get("solved", 0) > 0.5
        )
        best = max(population, key=lambda ind: ind["fitness"].scalar)

        print(f"\n=== Gen {gen}/{generations} ===")
        print(
            f"  solved: {solved_count}/{len(population)}  "
            f"best_scalar: {best['fitness'].scalar:.3f}  "
            f"mean_scalar: {statistics.mean(scalars):.3f}"
        )

        parents = epsilon_lexicase_select(
            population,
            senses=senses,
            objective_keys=objective_keys,
            eps_quantile=0.10,
            n_parents=pop_size,
        )

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
                    ch = mutate_program(ch, template_pool)
                new_children.append(ch)

            for ch in new_children:
                offspring.append(evaluate_program(ch, exe, evaluator, target))
                if len(offspring) >= pop_size:
                    break

        combined = population + offspring
        population = nsga2_survivor_selection(
            combined, k=pop_size, senses=senses, objective_keys=objective_keys
        )

    # 输出最终若干条解
    population.sort(key=lambda ind: ind["fitness"].scalar, reverse=True)
    print("\n=== Final top solutions ===")
    topk = min(5, len(population))
    for i in range(topk):
        ind = population[i]
        objs = ind["fitness"].objectives
        print(
            f"[{i+1}] scalar={ind['fitness'].scalar:.3f} "
            f"solved={bool(objs.get('solved', 0))} "
            f"route_len={objs.get('route_len', -1)}"
        )
        print(ind["route"].to_json())


def main():
    run_gp_search(
        pop_size=20,
        generations=15,
        p_crossover=0.7,
        p_mutation=0.4,
        seed=123,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
# 这个脚本实现了一个简单的基于遗传编程的多目标分子合成路径搜索。
# 它使用 SCScore 作为其中一个目标函数，结合库存分子和反应模板，评估和进化合成路径程序。