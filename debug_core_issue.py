#!/usr/bin/env python3
"""Deep debug script to find why no synthesis paths are found."""

from gp_core.data_loading import load_world_from_data
from gp_retro_feas import ActionMaskBuilder, FeasibilityEngine
from gp_retro_repr import Program, Select, ApplyTemplate, Stop
from gp_core.program_ops import random_program
from gp_core.templates import template_ids

print("=== Loading Data ===")
inventory, reg, targets = load_world_from_data(limit_targets=3)
print(f"Templates: {len(reg.templates)}")
print(f"Inventory purchasable sample: {inventory.is_purchasable('CCO')}")
print(f"Targets: {len(targets)}")

print("\n=== Testing First Target ===")
target = targets[0]
print(f"Target: {target}")

# Test mask building
print("\n--- Mask Building ---")
mask_builder = ActionMaskBuilder(reg, inventory=inventory)
mask = mask_builder.build(target)
print(f"Candidate templates: {len(mask.candidate_templates)}")
print(f"Feasible templates: {len(mask.feasible_templates)}")

# Show some reasons
print("\nReasons (first 5):")
for i, (tid, reason) in enumerate(list(mask.reasons.items())[:5]):
    print(f"  {tid}: {reason}")

# Test feasibility engine directly
print("\n--- Direct Feasibility Test ---")
engine = FeasibilityEngine(reg, inventory=inventory)
all_templates = template_ids(reg)
print(f"Testing {len(all_templates)} templates on target...")

working_templates = []
for i, tid in enumerate(all_templates[:10]):  # Test first 10
    try:
        result = engine.check_and_choose(tid, target)
        if result.ok:
            working_templates.append((tid, result))
            print(f"  OK  {tid}: {len(result.chosen_reactants)} reactants")
        else:
            print(f"  FAIL {tid}: {result.reason}")
    except Exception as e:
        print(f"  FAIL {tid}: ERROR - {e}")

print(f"\nWorking templates: {len(working_templates)}")

# Test program execution
if working_templates:
    print("\n--- Program Execution Test ---")
    from gp_retro_feas import FeasibleExecutor
    
    executor = FeasibleExecutor(reg, inventory=inventory)
    
    # Create a simple program with a working template
    working_tid = working_templates[0][0]
    simple_prog = Program([Select(0), ApplyTemplate(working_tid), Stop()])
    
    print(f"Testing program with template: {working_tid}")
    try:
        route = executor.execute(simple_prog, target)
        print(f"✓ Route created with {len(route.steps)} steps")
        if route.steps:
            step = route.steps[0]
            print(f"  Product: {step.product}")
            print(f"  Reactants: {step.reactants}")
            print(f"  Updated set: {step.updated_molecule_set}")
    except Exception as e:
        print(f"✗ Program execution failed: {e}")
        import traceback
        traceback.print_exc()

# Test random program generation
print("\n--- Random Program Test ---")
full_pool = template_ids(reg)
for i in range(3):
    prog = random_program(full_pool, min_len=1, max_len=2)
    print(f"Program {i+1}: {len(prog.instructions)} instructions")
    for j, instr in enumerate(prog.instructions):
        print(f"  {j}: {instr}")