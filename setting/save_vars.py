"""
Save a snapshot of the current GP / executor settings to a JSON file.

Example:
    python setting/save_vars.py --tag test_run --include-stats

    执行：
    python setting/save_vars.py --tag test --include-stats --note "大库存 + 默认参数"

"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def default_settings(include_stats: bool = False) -> Dict[str, Any]:
    from gp_core import config
    from gp_core.data_loading import load_inventory_and_templates
    from gp_retro_feas import ExecutePolicy

    data: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gp_config": {
            "pop_size": config.pop_size,
            "generations": config.generations,
            "p_crossover": config.pcross,
            "p_mutation": config.pmut,
            "seed": config.seed,
            "max_templates_per_prog": config.max_templates_per_prog,
            "nonempty_bonus": config.nonempty_bonus,
            "llm_style_scalar": config.llm_style_scalar,
            "objective_weights": config.objective_weights,
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
        import yaml

        try:
            inventory, reg = load_inventory_and_templates()
            # Targets are now defined via YAML config (DemoA/B/C/D style)
            tgt_cfg = config.data_root / "target molecular" / "config.yaml"
            num_targets = 0
            if tgt_cfg.exists():
                with tgt_cfg.open("r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                if isinstance(cfg, dict):
                    for entry in cfg.values():
                        if isinstance(entry, dict) and (
                            "target_smi" in entry or "target" in entry
                        ):
                            num_targets += 1

            data["stats"] = {
                "inventory_size": len(list(inventory)),
                "num_templates": len(getattr(reg, "templates", reg)),
                "num_targets": num_targets,
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
