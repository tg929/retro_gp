"""Program construction and genetic operators."""
import random
from typing import List, Tuple, Optional, Sequence, Union

from gp_retro_repr import Program, Select, ApplyTemplate, ApplyOneStepModel, Stop
from . import config


Gene = Tuple[str, Union[str, int]]  # ("T", template_id) or ("N", rank)


def program_from_genes(genes: Sequence[Gene]) -> Program:
    steps = []
    for kind, val in genes:
        steps.append(Select(0))
        if kind == "T":
            steps.append(ApplyTemplate(str(val), rational="gp"))
        elif kind == "N":
            steps.append(ApplyOneStepModel(rank=int(val), rational="gp"))
        else:
            raise ValueError(f"Unknown gene kind: {kind}")
    steps.append(Stop())
    return Program(steps)


def program_from_templates(template_ids: List[str]) -> Program:
    return program_from_genes([("T", tid) for tid in template_ids])


def templates_of_program(prog: Program) -> List[str]:
    tids: List[str] = []
    for instr in prog.instructions:
        if isinstance(instr, ApplyTemplate):
            tids.append(instr.template_id)
    return tids


def genes_of_program(prog: Program) -> List[Gene]:
    genes: List[Gene] = []
    for instr in prog.instructions:
        if isinstance(instr, ApplyTemplate):
            genes.append(("T", instr.template_id))
        elif isinstance(instr, ApplyOneStepModel):
            genes.append(("N", int(instr.rank)))
    return genes


def random_program(
    template_pool: List[str],
    min_len: int = 1,
    max_len: int = 8,
    *,
    allow_model_actions: bool = False,
    model_rank_pool: Optional[Sequence[int]] = None,
    p_model_action: float = 0.0,
) -> Program:
    k = random.randint(min_len, min(max_len, config.max_templates_per_prog))
    if not allow_model_actions or not model_rank_pool:
        tids = [random.choice(template_pool) for _ in range(k)]
        return program_from_templates(tids)

    genes: List[Gene] = []
    for _ in range(k):
        if random.random() < float(p_model_action):
            genes.append(("N", int(random.choice(model_rank_pool))))
        else:
            genes.append(("T", str(random.choice(template_pool))))
    return program_from_genes(genes)


def crossover_one_point(p1: Program, p2: Program) -> Tuple[Program, Program]:
    t1 = genes_of_program(p1)
    t2 = genes_of_program(p2)
    c1 = random.randint(0, len(t1))
    c2 = random.randint(0, len(t2))
    child1 = program_from_genes(t1[:c1] + t2[c2:])
    child2 = program_from_genes(t2[:c2] + t1[c1:])
    return child1, child2


def mutate_program(
    p: Program,
    template_pool: List[str],
    p_insert=0.4,
    p_delete=0.25,
    p_modify=0.35,
    max_total_len: int = config.max_templates_per_prog,
    feasible_templates: Optional[List[str]] = None,
    *,
    allow_model_actions: bool = False,
    model_rank_pool: Optional[Sequence[int]] = None,
    p_model_action: float = 0.0,
) -> Program:
    genes = genes_of_program(p)
    # Prefer chemistry-checked templates when provided (from feasible mask on target)
    template_insert_pool = feasible_templates or template_pool

    def _sample_gene() -> Gene:
        if allow_model_actions and model_rank_pool and random.random() < float(p_model_action):
            return ("N", int(random.choice(model_rank_pool)))
        return ("T", str(random.choice(template_insert_pool)))

    op = random.random()
    if op < p_insert and len(genes) < max_total_len:
        pos = random.randint(0, len(genes))
        genes.insert(pos, _sample_gene())
    elif op < p_insert + p_delete and len(genes) > 0:
        pos = random.randrange(len(genes))
        genes.pop(pos)
    elif len(genes) > 0:
        pos = random.randrange(len(genes))
        genes[pos] = _sample_gene()
    return program_from_genes(genes)
