#!/usr/bin/env python3
"""Debug script to check dependencies and basic functionality."""

print("=== Dependency Check ===")

# Check rdkit
try:
    from rdkit import Chem
    print("✓ rdkit available")
    test_mol = Chem.MolFromSmiles("CCO")
    print(f"✓ rdkit basic test: {test_mol is not None}")
except ImportError as e:
    print(f"✗ rdkit missing: {e}")

# Check rdchiral
try:
    from rdchiral.initialization import rdchiralReaction, rdchiralReactants
    from rdchiral.main import rdchiralRun
    print("✓ rdchiral available")
except ImportError as e:
    print(f"✗ rdchiral missing: {e}")

print("\n=== Basic Template Test ===")
from gp_core.data_loading import load_world_from_data
from gp_retro_feas import ActionMaskBuilder

try:
    inventory, reg, targets = load_world_from_data(limit_targets=1)
    print(f"✓ Data loaded: {len(reg.templates)} templates, {len(targets)} targets")
    
    target = targets[0]
    print(f"✓ Testing target: {target}")
    
    mask_builder = ActionMaskBuilder(reg, inventory=inventory)
    mask = mask_builder.build(target)
    
    print(f"✓ Mask built:")
    print(f"  - Candidate templates: {len(mask.candidate_templates)}")
    print(f"  - Feasible templates: {len(mask.feasible_templates)}")
    print(f"  - Reasons sample: {dict(list(mask.reasons.items())[:3])}")
    
except Exception as e:
    print(f"✗ Template test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== FG Detection Test ===")
from gp_retro_feas.fg_patterns import find_functional_groups

test_smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
for smi in test_smiles:
    try:
        fgs = find_functional_groups(smi)
        print(f"✓ {smi}: {fgs}")
    except Exception as e:
        print(f"✗ {smi}: {e}")