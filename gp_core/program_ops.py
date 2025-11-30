"""Program construction and genetic operators."""
import random
from typing import List, Tuple

from gp_retro_repr import Program, Select, ApplyTemplate, Stop
from . import config


def program_from_templates(template_ids: List[str]) -> Program:
    steps = []
    for tid in template_ids:
        steps.append(Select(0))
        steps.append(ApplyTemplate(tid, rational="gp"))
    steps.append(Stop())
    return Program(steps)


def templates_of_program(prog: Program) -> List[str]:
    tids: List[str] = []
    for instr in prog.instructions:
        if isinstance(instr, ApplyTemplate):
            tids.append(instr.template_id)
    return tids


def random_program(template_pool: List[str], min_len=1, max_len=8) -> Program:
    k = random.randint(min_len, min(max_len, config.max_templates_per_prog))
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
    p_insert=0.4,
    p_delete=0.25,
    p_modify=0.35,
    max_total_len: int = config.max_templates_per_prog,
) -> Program:
    t = templates_of_program(p)
    op = random.random()
    if op < p_insert and len(t) < max_total_len:
        pos = random.randint(0, len(t))
        t.insert(pos, random.choice(template_pool))
    elif op < p_insert + p_delete and len(t) > 0:
        pos = random.randrange(len(t))
        t.pop(pos)
    elif len(t) > 0:
        pos = random.randrange(len(t))
        t[pos] = random.choice(template_pool)
    return program_from_templates(t)
