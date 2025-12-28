"""Try hard to solve DemoD by boosting GP/NAG2G search budget.

This script does NOT change building blocks. It increases:
- NAG2G topk/beam
- GP population size / generations
- Max program length
- Mixed actions: template + NAG2G (p_model_action < 1.0)

If a solution does not exist under the current inventory/leaf criteria, no script can
"guarantee" a solved route.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Optional, Tuple

import yaml

import gp_core.config as gp_config
from gp_core.data_loading import load_inventory_and_templates
from gp_core.executor import make_executor
from gp_core.scoring import GPFitnessEvaluator, StepScorer, StepScoreConfig
from gp_core.search import run_gp_for_target
from gp_core.templates import template_ids
from gp_retro_feas import ActionMaskBuilder, ExecutePolicy
from gp_retro_nn import NAG2GSubprocessConfig, NAG2GSubprocessModel
from gp_retro_repr import ChemHistorian, LeafCriteria, LeafCriteriaConfig


def _maybe_default_nag2g_python() -> Optional[str]:
    env = os.environ.get("NAG2G_PYTHON")
    if env and Path(env).exists():
        return env

    candidates = [
        Path.home() / "anaconda3/envs/nag2g/bin/python",
        Path.home() / "miniconda3/envs/nag2g/bin/python",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _load_target_from_yaml(target_key: str) -> Tuple[str, str, Optional[str]]:
    cfg_path = gp_config.data_root / "target molecular" / "config.yaml"
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected target config format: {cfg_path}")

    for name, entry in data.items():
        if str(name).lower() != str(target_key).lower():
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid target entry for {name}: expected dict")
        smi = entry.get("target_smi") or entry.get("target")
        if not smi:
            raise ValueError(f"Target {name} has no 'target_smi' in {cfg_path}")
        bb_ds = entry.get("building_block_dataset")
        return str(name), str(smi), (str(bb_ds) if bb_ds else None)

    available = ", ".join(str(k) for k in data.keys())
    raise ValueError(f"Target '{target_key}' not found in {cfg_path}. Available: {available}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Boosted GP search for a single target (default: DemoD)."
    )
    parser.add_argument("--target", type=str, default="DemoD")
    parser.add_argument(
        "--templates",
        type=str,
        default=None,
        help="Optional path to template file (relative to data/ or absolute).",
    )

    # Leaf/stop criteria (same knobs as run_real_data_gp.py)
    parser.add_argument(
        "--leaf-chemical-property-logic",
        type=str,
        default="or",
        choices=["none", "or", "and"],
    )
    parser.add_argument("--leaf-max-chemprop-c", type=int, default=0)
    parser.add_argument("--leaf-max-chemprop-n", type=int, default=1)
    parser.add_argument("--leaf-max-chemprop-o", type=int, default=0)
    parser.add_argument("--leaf-max-chemprop-h", type=int, default=3)
    parser.add_argument(
        "--leaf-chemical-popularity-logic",
        type=str,
        default="none",
        choices=["none", "or"],
    )
    parser.add_argument("--leaf-min-chempop-reactants", type=int, default=5)
    parser.add_argument("--leaf-min-chempop-products", type=int, default=5)
    parser.add_argument(
        "--leaf-chem-history-path",
        type=str,
        default=None,
        help="JSON/JSON.GZ chemical history file.",
    )

    # NAG2G
    parser.add_argument("--nag2g-project-dir", type=str, default="NAG2G-main")
    parser.add_argument(
        "--nag2g-data-dir",
        type=str,
        default="NAG2G-main/USPTO50K_brief_20230227/USPTO50K_brief_20230227",
    )
    parser.add_argument(
        "--nag2g-checkpoint",
        type=str,
        default="NAG2G-main/NAG2G_unimolplus_uspto_50k_20230513-222355/checkpoint_last.pt",
    )
    parser.add_argument(
        "--nag2g-python",
        type=str,
        default=None,
        help="Python executable used to run NAG2G inference (recommended: nag2g env).",
    )
    parser.add_argument("--nag2g-dict-name", type=str, default="dict_20230310.txt")
    parser.add_argument("--nag2g-bpe-tokenizer-path", type=str, default="none")
    parser.add_argument(
        "--nag2g-search-strategies",
        type=str,
        default="SequenceGeneratorBeamSearch",
        choices=["SimpleGenerator", "SequenceGeneratorBeamSearch"],
    )
    parser.add_argument(
        "--nag2g-topk",
        type=int,
        default=30,
        help="Requested candidates per expansion (also used as beam size).",
    )
    parser.add_argument(
        "--nag2g-action-prob",
        type=float,
        default=0.7,
        help="Probability to use NAG2G gene vs template gene.",
    )
    parser.add_argument("--nag2g-len-penalty", type=float, default=0.0)
    parser.add_argument("--nag2g-temperature", type=float, default=1.0)
    parser.add_argument("--nag2g-cpu", action="store_true")
    parser.add_argument("--nag2g-fp16", action="store_true")

    # GP budget
    parser.add_argument("--pop-size", type=int, default=20)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--max-prog-len", type=int, default=25)
    parser.add_argument("--tries", type=int, default=15)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    target_name, target_smiles, bb_ds = _load_target_from_yaml(args.target)

    files = [bb_ds] if bb_ds else None
    inventory, reg = load_inventory_and_templates(files=files, templates_path=args.templates)

    # Configure leaf criteria (ASKCOS style)
    leaf_enabled = str(args.leaf_chemical_property_logic).lower() not in ["none", ""] or str(
        args.leaf_chemical_popularity_logic
    ).lower() not in ["none", ""]

    if not leaf_enabled:
        inventory.set_leaf_criteria(None)
        print("Leaf criteria disabled")
    else:
        max_natom_dict = LeafCriteriaConfig.make_max_natom_dict(
            logic=str(args.leaf_chemical_property_logic),
            C=int(args.leaf_max_chemprop_c),
            N=int(args.leaf_max_chemprop_n),
            O=int(args.leaf_max_chemprop_o),
            H=int(args.leaf_max_chemprop_h),
        )
        min_hist_dict = LeafCriteriaConfig.make_min_history_dict(
            logic=str(args.leaf_chemical_popularity_logic),
            as_reactant=int(args.leaf_min_chempop_reactants),
            as_product=int(args.leaf_min_chempop_products),
        )

        chemhistorian = None
        if str(min_hist_dict.get("logic", "none")).lower() not in ["none", ""]:
            if not args.leaf_chem_history_path:
                raise ValueError(
                    "leaf_chemical_popularity_logic is enabled but --leaf-chem-history-path is not provided"
                )
            chemhistorian = ChemHistorian()
            chemhistorian.load_from_file(args.leaf_chem_history_path)

        leaf_cfg = LeafCriteriaConfig(max_natom_dict=max_natom_dict, min_chemical_history_dict=min_hist_dict)
        inventory.set_leaf_criteria(LeafCriteria(cfg=leaf_cfg, chemhistorian=chemhistorian))
        print(
            "Leaf criteria enabled:",
            {
                "chemical_property_logic": max_natom_dict.get("logic"),
                "max_natom": {k: int(v) for k, v in max_natom_dict.items() if k != "logic"},
                "chemical_popularity_logic": min_hist_dict.get("logic"),
                "min_history": {k: int(v) for k, v in min_hist_dict.items() if k != "logic"},
                "history_path": args.leaf_chem_history_path,
            },
        )

    print(f"\n=== Target ({target_name}): {target_smiles} ===")

    # Build feasible template mask (init only)
    mask = ActionMaskBuilder(reg, inventory=inventory).build(target_smiles)
    full_pool = template_ids(reg)
    init_pool = mask.feasible_templates or full_pool
    if mask.feasible_templates:
        print(
            f"Feasible templates for target (init only): {len(mask.feasible_templates)} "
            f"(of {len(mask.candidate_templates)})"
        )
    else:
        print("No feasible templates found; fallback to full template pool.")

    # NAG2G subprocess model
    nag2g_python = args.nag2g_python or _maybe_default_nag2g_python()
    if not nag2g_python:
        print(
            "[warn] --nag2g-python not provided and NAG2G_PYTHON not found; "
            "falling back to current python (may fail if NAG2G deps missing).",
            file=sys.stderr,
        )

    one_step_model = NAG2GSubprocessModel(
        NAG2GSubprocessConfig(
            project_dir=Path(args.nag2g_project_dir),
            data_dir=Path(args.nag2g_data_dir),
            checkpoint_path=Path(args.nag2g_checkpoint),
            python_executable=nag2g_python,
            dict_name=str(args.nag2g_dict_name),
            bpe_tokenizer_path=str(args.nag2g_bpe_tokenizer_path),
            beam_size=int(args.nag2g_topk),
            search_strategies=str(args.nag2g_search_strategies),
            len_penalty=float(args.nag2g_len_penalty),
            temperature=float(args.nag2g_temperature),
            cpu=bool(args.nag2g_cpu),
            fp16=bool(args.nag2g_fp16),
        )
    )

    try:
        # StepScorer for rerank/truncation (keep topB=topk to actually use the larger topk)
        step_score_type = "raw" if str(args.nag2g_search_strategies) == "SequenceGeneratorBeamSearch" else "rank"
        topB = max(1, int(args.nag2g_topk))
        step_scorer = StepScorer(
            config=StepScoreConfig(score_type=step_score_type, topN=int(args.nag2g_topk), topB=topB),
            bb_is_purchasable=inventory.is_purchasable,
            sa_fn=None,
            forward_model=None,
        )
        one_step_ranker = lambda product_smiles, preds: step_scorer.rank_and_truncate(product_smiles, preds)

        policy = ExecutePolicy(one_step_topk=int(args.nag2g_topk))
        exe = make_executor(
            reg,
            inventory,
            policy=policy,
            one_step_model=one_step_model,
            one_step_ranker=one_step_ranker,
        )

        evaluator = GPFitnessEvaluator(target_smiles=target_smiles, bb_is_purchasable=inventory.is_leaf)

        # Increase max program length for this run
        gp_config.max_templates_per_prog = int(args.max_prog_len)

        for t in range(int(args.tries)):
            seed = int(args.seed_start) + t
            print(f"\n--- Try {t+1}/{args.tries} seed={seed} ---")

            buf = StringIO()
            ctx = redirect_stdout(buf) if not args.verbose else None
            if ctx is None:
                population, _ = run_gp_for_target(
                    target=target_smiles,
                    inventory=inventory,
                    reg=reg,
                    evaluator=evaluator,
                    exe=exe,
                    pop_size=int(args.pop_size),
                    generations=int(args.generations),
                    seed=seed,
                    template_pool=full_pool,
                    init_templates=init_pool,
                    feasible_templates_for_target=mask.feasible_templates or None,
                    allow_model_actions=True,
                    model_rank_pool=list(range(topB)),
                    p_model_action=float(args.nag2g_action_prob),
                )
            else:
                with ctx:
                    population, _ = run_gp_for_target(
                        target=target_smiles,
                        inventory=inventory,
                        reg=reg,
                        evaluator=evaluator,
                        exe=exe,
                        pop_size=int(args.pop_size),
                        generations=int(args.generations),
                        seed=seed,
                        template_pool=full_pool,
                        init_templates=init_pool,
                        feasible_templates_for_target=mask.feasible_templates or None,
                        allow_model_actions=True,
                        model_rank_pool=list(range(topB)),
                        p_model_action=float(args.nag2g_action_prob),
                    )

            solved = [
                ind
                for ind in population
                if ind["fitness"].objectives.get("solved", 0) > 0.5
            ]
            if solved:
                best = solved[0]  # already sorted by scalar
                print("\n=== SOLVED FOUND ===")
                print(
                    f"seed={seed} scalar={best['fitness'].scalar:.3f} "
                    f"steps={best['fitness'].objectives.get('steps')}"
                )
                print(best["route"].to_json())
                if not args.verbose:
                    print("\n--- captured GP log (this try) ---")
                    print(buf.getvalue())
                return 0

            best_scalar = population[0]["fitness"].scalar if population else float("-inf")
            print(f"No solved route in this try. best_scalar={best_scalar:.3f}")

        print(
            "\nNo solved route found after retries. "
            "This usually means either (1) search budget still insufficient, or "
            "(2) the route is infeasible under current stock/leaf criteria.",
            file=sys.stderr,
        )
        return 2

    finally:
        try:
            one_step_model.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
