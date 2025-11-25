"""
Save a snapshot of the current GP / executor settings to a JSON file.

Example:
    python setting/save_vars.py --tag test_run --include-stats
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def default_settings(include_stats: bool = False) -> Dict[str, Any]:
    from gp_core import config
    from gp_retro_feas import ExecutePolicy

    data: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gp_config": {
            "pop_size": config.POP_SIZE,
            "generations": config.GENERATIONS,
            "p_crossover": config.PCROSS,
            "p_mutation": config.PMUT,
            "seed": config.SEED,
            "max_templates_per_prog": config.MAX_TEMPLATES_PER_PROG,
            "nonempty_bonus": config.NONEMPTY_BONUS,
            "llm_style_scalar": config.LLM_STYLE_SCALAR,
            "objective_weights": config.OBJECTIVE_WEIGHTS,
        },
        "executor_policy": {
            "require_all_purchasable": ExecutePolicy.require_all_purchasable,
            "max_reactants": ExecutePolicy.max_reactants,
            "repair_on_failure": ExecutePolicy.repair_on_failure,
        },
        "run_real_data_gp_notes": {
            "template_pool": "full HB template set (see gp_core.templates.template_ids)",
            "init_templates": "feasible templates for the target if available; otherwise full pool",
        },
    }

    if include_stats:
        from gp_core.data_loading import load_world_from_data
        try:
            inventory, reg, targets = load_world_from_data()
            data["stats"] = {
                "inventory_size": len(list(inventory)),
                "num_templates": len(reg.templates),
                "num_targets": len(targets),
            }
        except Exception as e:  # pragma: no cover
            data["stats_error"] = f"{type(e).__name__}: {e}"
    return data


def main():
    parser = argparse.ArgumentParser(description="Save GP execution settings.")
    parser.add_argument("--tag", type=str, default="", help="Optional tag to include in filename.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file. Default: logs/run_settings_<tag>_<timestamp>.json",
    )
    parser.add_argument(
        "--include-stats",
        action="store_true",
        help="Also load data to record inventory/template/target counts.",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Free-form note to store alongside settings.",
    )
    args = parser.parse_args()

    payload = default_settings(include_stats=args.include_stats)
    if args.note:
        payload["note"] = args.note

    ts = datetime.now().strftime("%m%d%H%M")
    tag_part = args.tag.strip().replace(" ", "_")
    fname = f"run_settings_{tag_part + '_' if tag_part else ''}{ts}.json"
    out_path = args.output or Path("logs") / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved settings to {out_path}")


if __name__ == "__main__":
    main()
