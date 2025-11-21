from gp_core.data_loading import load_world_from_data
from gp_core.templates import template_ids
from gp_core.program_ops import random_program
from gp_core.executor import make_executor
from gp_retro_feas.engine import FeasibilityEngine
from gp_retro_feas.mask import ActionMaskBuilder
from gp_retro_repr import ApplyTemplate

inventory, reg, targets = load_world_from_data(limit_targets=1)
target = targets[0]
print('Target', target)
engine = FeasibilityEngine(reg, inventory=inventory)
mask = ActionMaskBuilder(reg, inventory=inventory).build(target)
print('feasible templates', len(mask.feasible_templates))
print('actions sample', mask.feasible_templates[:5])
executor = make_executor(reg, inventory)
full_pool = template_ids(reg)
for i in range(5):
    prog = random_program(full_pool, min_len=1, max_len=3)
    print('\nProgram', i+1, [instr for instr in prog.instructions])
    for instr in prog.instructions:
        if isinstance(instr, ApplyTemplate):
            res = engine.check_and_choose(instr.template_id, target)
            print('  template', instr.template_id, 'result ok', res.ok, 'reason', res.reason)
    try:
        route = executor.execute(prog, target_smiles=target)
        print('  route steps', len(route.steps))
        if route.steps:
            for step in route.steps:
                print('    step template', step.template_id)
    except Exception as exc:
        print('  executor raised', exc)
