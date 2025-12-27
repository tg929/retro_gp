"""Run GP search on real data using modular gp_core helpers."""
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import yaml

from gp_core import config
from gp_core.data_loading import load_inventory_and_templates
from gp_core.templates import template_ids
from gp_core.executor import make_executor
from gp_core.scoring import GPFitnessEvaluator, StepScorer, StepScoreConfig
from gp_core.search import run_gp_for_target
from gp_core.metrics import MetricsHistory
from gp_retro_feas import ActionMaskBuilder, ExecutePolicy, FeasibleExecutor

try:  # optional neural one-step model
    from gp_retro_nn import NAG2GSubprocessConfig, NAG2GSubprocessModel
except Exception:  # pragma: no cover
    NAG2GSubprocessConfig = None  # type: ignore
    NAG2GSubprocessModel = None  # type: ignore


def make_audit_fn(stock, target_smiles: str):
    def audit_route(route):
        if route.steps:
            final_set = route.steps[-1].updated_molecule_set
            n_steps = len(route.steps)
        else:
            final_set = [target_smiles]
            n_steps = 0
        is_leaf = getattr(stock, "is_leaf", None) or stock.is_purchasable
        is_solved = all(is_leaf(m) for m in final_set)
        return {
            "is_solved": is_solved,
            "first_invalid_molecule_set": [] if is_solved else list(final_set),
            "current_molecule_set": list(final_set),
            "n_steps": n_steps,
            "n_valid_steps": n_steps,
        }
    return audit_route


def _load_targets_from_yaml(config_path: Path):
    """Load named target SMILES (and optional building block dataset) from YAML."""
    if not config_path.exists():
        raise FileNotFoundError(f"Target config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    targets = []
    if isinstance(data, dict):
        for name, entry in data.items():
            if not isinstance(entry, dict):
                continue
            smi = entry.get("target_smi") or entry.get("target")
            if smi:
                targets.append(
                    {
                        "name": str(name),
                        "smiles": str(smi),
                        "building_block_dataset": entry.get("building_block_dataset"),
                    }
                )
    if not targets:
        raise ValueError(f"No targets with 'target_smi' found in {config_path}")
    return targets

def run(
    target_key: str = "all",
    template_file: Optional[str] = None,
    *,
    # Leaf/stop criteria (ASKCOS-style)
    leaf_chemical_property_logic: str = "none",
    leaf_max_chemprop_c: int = 0,
    leaf_max_chemprop_n: int = 0,
    leaf_max_chemprop_o: int = 0,
    leaf_max_chemprop_h: int = 0,
    leaf_chemical_popularity_logic: str = "none",
    leaf_min_chempop_reactants: int = 5,
    leaf_min_chempop_products: int = 5,
    leaf_chem_history_path: Optional[str] = None,
    nag2g_project_dir: Optional[str] = None,
    nag2g_data_dir: Optional[str] = None,
    nag2g_checkpoint_path: Optional[str] = None,
    nag2g_python: Optional[str] = None,
    nag2g_dict_name: str = "dict.txt",
    nag2g_bpe_tokenizer_path: str = "none",
    nag2g_search_strategies: str = "SimpleGenerator",
    nag2g_topk: int = 10,
    nag2g_action_prob: float = 0.0,
    nag2g_len_penalty: float = 0.0,
    nag2g_temperature: float = 1.0,
    nag2g_cpu: bool = False,
    nag2g_fp16: bool = False,
):
    # Ensure log directory exists before any cache writes
    Path("logs").mkdir(exist_ok=True)
    hist = MetricsHistory()

    # Load targets from YAML config (DemoA/B/C/D style)
    target_cfg_path = config.data_root / "target molecular" / "config.yaml"
    all_targets = _load_targets_from_yaml(target_cfg_path)

    # Filter by requested target key
    if target_key and target_key.lower() != "all":
        filtered = [
            t for t in all_targets if t["name"].lower() == target_key.lower()
        ]
        if not filtered:
            available = ", ".join(t["name"] for t in all_targets)
            raise ValueError(
                f"Target '{target_key}' not found in {target_cfg_path}. "
                f"Available keys: {available} (or 'all')."
            )
        targets_named = filtered
    else:
        targets_named = all_targets

    n_targets = len(targets_named)

    def _progress(current: int, total: int):
        """Print a final progress bar to the real terminal (not the log)."""
        if total <= 0:
            return
        bar_width = 30
        # For your preference, show a bar made of '-' only.
        bar = "-" * bar_width
        print(
            f"Progress: [{bar}] {current}/{total} targets",
            file=sys.__stdout__,
            flush=True,
        )

    # Cache inventories/templates per building block dataset to avoid re-loading
    inv_reg_cache = {}

    one_step_model = None
    if (
        float(nag2g_action_prob) > 0.0
        or nag2g_checkpoint_path
        or nag2g_data_dir
        or nag2g_project_dir
    ):
        if NAG2GSubprocessModel is None or NAG2GSubprocessConfig is None:
            raise RuntimeError("gp_retro_nn is not available; cannot enable NAG2G one-step model")
        if not nag2g_data_dir:
            raise ValueError("--nag2g-data-dir is required to enable NAG2G one-step model")
        if not nag2g_checkpoint_path:
            raise ValueError("--nag2g-checkpoint is required to enable NAG2G one-step model")

        proj = Path(nag2g_project_dir or (Path(__file__).resolve().parent / "NAG2G-main"))
        data_dir = Path(nag2g_data_dir)
        ckpt = Path(nag2g_checkpoint_path)
        one_step_model = NAG2GSubprocessModel(
            NAG2GSubprocessConfig(
                project_dir=proj,
                data_dir=data_dir,
                checkpoint_path=ckpt,
                python_executable=nag2g_python,
                dict_name=str(nag2g_dict_name),
                bpe_tokenizer_path=str(nag2g_bpe_tokenizer_path),
                beam_size=int(nag2g_topk),
                search_strategies=str(nag2g_search_strategies),
                len_penalty=float(nag2g_len_penalty),
                temperature=float(nag2g_temperature),
                cpu=bool(nag2g_cpu),
                fp16=bool(nag2g_fp16),
            )
        )

    for ti, tinfo in enumerate(targets_named):
        name = tinfo["name"]
        target = tinfo["smiles"]
        bb_ds = tinfo.get("building_block_dataset")

        cache_key = bb_ds or "__all__"
        if cache_key not in inv_reg_cache:
            files = [bb_ds] if bb_ds else None
            inventory, reg = load_inventory_and_templates(files=files, templates_path=template_file)
            inv_reg_cache[cache_key] = (inventory, reg)
        else:
            inventory, reg = inv_reg_cache[cache_key]

        # Configure ASKCOS-style leaf/stop criteria on the inventory (no building-block changes).
        leaf_enabled = str(leaf_chemical_property_logic).lower() not in ["none", ""] or str(
            leaf_chemical_popularity_logic
        ).lower() not in ["none", ""]
        if not leaf_enabled:
            inventory.set_leaf_criteria(None)
        else:
            from gp_retro_repr import LeafCriteria, LeafCriteriaConfig, ChemHistorian

            max_natom_dict = LeafCriteriaConfig.make_max_natom_dict(
                logic=str(leaf_chemical_property_logic),
                C=int(leaf_max_chemprop_c),
                N=int(leaf_max_chemprop_n),
                O=int(leaf_max_chemprop_o),
                H=int(leaf_max_chemprop_h),
            )
            min_hist_dict = LeafCriteriaConfig.make_min_history_dict(
                logic=str(leaf_chemical_popularity_logic),
                as_reactant=int(leaf_min_chempop_reactants),
                as_product=int(leaf_min_chempop_products),
            )

            chemhistorian = None
            if str(min_hist_dict.get("logic", "none")).lower() not in ["none", ""]:
                if not leaf_chem_history_path:
                    raise ValueError(
                        "leaf_chemical_popularity_logic is enabled but --leaf-chem-history-path is not provided"
                    )
                chemhistorian = ChemHistorian()
                chemhistorian.load_from_file(leaf_chem_history_path)

            leaf_cfg = LeafCriteriaConfig(
                max_natom_dict=max_natom_dict,
                min_chemical_history_dict=min_hist_dict,
            )
            inventory.set_leaf_criteria(LeafCriteria(cfg=leaf_cfg, chemhistorian=chemhistorian))
            print(
                "Leaf criteria enabled:",
                {
                    "chemical_property_logic": max_natom_dict.get("logic"),
                    "max_natom": {k: int(v) for k, v in max_natom_dict.items() if k != "logic"},
                    "chemical_popularity_logic": min_hist_dict.get("logic"),
                    "min_history": {k: int(v) for k, v in min_hist_dict.items() if k != "logic"},
                    "history_path": leaf_chem_history_path,
                },
            )

        print(f"\n=== Target {ti+1}/{n_targets} ({name}): {target} ===")

        # 针对当前目标先做一次可行动作掩码，避免全部落空
        mask = ActionMaskBuilder(reg, inventory=inventory).build(target)
        # 修复：template_pool 必须是全量模板，否则后续步骤无法进行
        full_pool = template_ids(reg)
        
        # 仅将可行模板用于初始化种群
        init_pool = mask.feasible_templates or full_pool

        if mask.feasible_templates:
            print(f"Feasible templates for target (init only): {len(mask.feasible_templates)} (of {len(mask.candidate_templates)})")
        else:
            print("No feasible templates found; fallback to full template pool.")

        # One-step StepScore: re-rank & truncate model candidates before rank selection.
        # Default: keep topB=min(20, topN) after scoring.
        topB = max(1, min(20, int(nag2g_topk)))
        # If NAG2G provides beam scores (SequenceGeneratorBeamSearch), prefer using them
        # instead of rank-only scoring.
        step_score_type = "raw" if str(nag2g_search_strategies) == "SequenceGeneratorBeamSearch" else "rank"
        step_scorer = StepScorer(
            config=StepScoreConfig(score_type=step_score_type, topN=int(nag2g_topk), topB=topB),
            bb_is_purchasable=inventory.is_purchasable,
            sa_fn=None,
            forward_model=None,
        )
        one_step_ranker = (lambda product_smiles, preds: step_scorer.rank_and_truncate(product_smiles, preds)) if one_step_model else None

        policy = ExecutePolicy(one_step_topk=int(nag2g_topk))
        exe: FeasibleExecutor = make_executor(
            reg,
            inventory,
            policy=policy,
            one_step_model=one_step_model,
            one_step_ranker=one_step_ranker,
        )

        evaluator = GPFitnessEvaluator(
            target_smiles=target,
            bb_is_purchasable=inventory.is_leaf,
        )

        population, hist = run_gp_for_target(
            target=target,
            inventory=inventory,
            reg=reg,
            evaluator=evaluator,
            exe=exe,
            template_pool=full_pool,     # 搜索空间：全量
            init_templates=init_pool,    # 初始化偏好：局部可行
            history=hist,
            feasible_templates_for_target=mask.feasible_templates or None,
            allow_model_actions=bool(one_step_model) and float(nag2g_action_prob) > 0.0,
            model_rank_pool=list(range(max(1, int(topB)))) if one_step_model else None,
            p_model_action=float(nag2g_action_prob),
        )

        print("Top solutions:")
        for rank, ind in enumerate(population[:3], start=1):
            objs = ind["fitness"].objectives
            print(
                f"  [{rank}] scalar={ind['fitness'].scalar:.3f} "
                f"solved={bool(objs.get('solved', 0))} "
                f"route_len={objs.get('route_len', -1)}"
            )
            print(ind["route"].to_json())

    # 完成所有目标后，打印一次完整进度
    _progress(n_targets, n_targets)

    if hist.has_updates:
        stats = hist.metrics(budget=config.pop_size * config.generations)
        print("\nMetrics snapshot:", {k: round(v, 3) for k, v in stats.items()})


if __name__ == "__main__":
    # Mirror stdout/stderr to a time-stamped log file for reproducibility
    from datetime import datetime

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%m%d%H%M")
    log_path = log_dir / f"run_real_data_gp_output_{ts}.txt"

    parser = argparse.ArgumentParser(description="Run GP search on real data targets.")
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        help="Target key from data/target molecular/config.yaml (e.g., DemoA) or 'all'.",
    )
    parser.add_argument(
        "--templates",
        type=str,
        default=None,
        help="Optional path to template file (relative to data/ or absolute). Default: data/reaction_template/hb.txt",
    )
    # ASKCOS-style leaf/stop criteria (no need to modify building-block files)
    parser.add_argument(
        "--leaf-chemical-property-logic",
        type=str,
        default="none",
        choices=["none", "or", "and"],
        help="Treat a molecule as a leaf if it is buyable OR/AND satisfies max atom-count constraints (ASKCOS style).",
    )
    parser.add_argument("--leaf-max-chemprop-c", type=int, default=0, help="Max C atoms for leaf property check.")
    parser.add_argument("--leaf-max-chemprop-n", type=int, default=0, help="Max N atoms for leaf property check.")
    parser.add_argument("--leaf-max-chemprop-o", type=int, default=0, help="Max O atoms for leaf property check.")
    parser.add_argument("--leaf-max-chemprop-h", type=int, default=0, help="Max H atoms for leaf property check.")
    parser.add_argument(
        "--leaf-chemical-popularity-logic",
        type=str,
        default="none",
        choices=["none", "or"],
        help="Treat a molecule as a leaf if it is buyable OR popular enough (ASKCOS style).",
    )
    parser.add_argument(
        "--leaf-min-chempop-reactants",
        type=int,
        default=5,
        help="Min frequency as reactant for popularity leaf check (requires --leaf-chem-history-path).",
    )
    parser.add_argument(
        "--leaf-min-chempop-products",
        type=int,
        default=5,
        help="Min frequency as product for popularity leaf check (requires --leaf-chem-history-path).",
    )
    parser.add_argument(
        "--leaf-chem-history-path",
        type=str,
        default=None,
        help="JSON/JSON.GZ chemical history file with fields {smiles, as_reactant, as_product}.",
    )
    parser.add_argument(
        "--nag2g-project-dir",
        type=str,
        default=None,
        help="Path to NAG2G-main (default: ./NAG2G-main).",
    )
    parser.add_argument(
        "--nag2g-data-dir",
        type=str,
        default=None,
        help="Path to NAG2G data dir containing dict.txt (required to enable NAG2G).",
    )
    parser.add_argument(
        "--nag2g-checkpoint",
        type=str,
        default=None,
        help="Path to NAG2G checkpoint .pt (required to enable NAG2G).",
    )
    parser.add_argument(
        "--nag2g-python",
        type=str,
        default=None,
        help="Python executable used to run NAG2G inference (must have torch/unicore/unimol/rdkit).",
    )
    parser.add_argument(
        "--nag2g-dict-name",
        type=str,
        default="dict.txt",
        help="Dictionary filename under --nag2g-data-dir (e.g., dict_20230310.txt).",
    )
    parser.add_argument(
        "--nag2g-bpe-tokenizer-path",
        type=str,
        default="none",
        help="Optional BPE tokenizer path for NAG2G (default: none).",
    )
    parser.add_argument(
        "--nag2g-search-strategies",
        type=str,
        default="SimpleGenerator",
        choices=["SimpleGenerator", "SequenceGeneratorBeamSearch"],
        help="NAG2G inference strategy.",
    )
    parser.add_argument(
        "--nag2g-topk",
        type=int,
        default=10,
        help="How many one-step candidates to request from NAG2G per product (also used as beam size).",
    )
    parser.add_argument(
        "--nag2g-action-prob",
        type=float,
        default=0.0,
        help="Probability that a GP gene uses ApplyOneStepModel instead of ApplyTemplate (0 disables model actions).",
    )
    parser.add_argument(
        "--nag2g-len-penalty",
        type=float,
        default=0.0,
        help="Length penalty for NAG2G beam search (if applicable).",
    )
    parser.add_argument(
        "--nag2g-temperature",
        type=float,
        default=1.0,
        help="Temperature for NAG2G sampling/decoding (if applicable).",
    )
    parser.add_argument(
        "--nag2g-cpu",
        action="store_true",
        help="Force NAG2G inference on CPU (slow).",
    )
    parser.add_argument(
        "--nag2g-fp16",
        action="store_true",
        help="Enable fp16 for NAG2G model (GPU only).",
    )
    args = parser.parse_args()

    # 将所有 print 输出重定向到日志文件；进度条单独写到 sys.__stdout__
    with log_path.open("w", encoding="utf-8") as _f:
        _orig_out, _orig_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = _f
        try:
            run(
                target_key=args.target,
                template_file=args.templates,
                leaf_chemical_property_logic=args.leaf_chemical_property_logic,
                leaf_max_chemprop_c=args.leaf_max_chemprop_c,
                leaf_max_chemprop_n=args.leaf_max_chemprop_n,
                leaf_max_chemprop_o=args.leaf_max_chemprop_o,
                leaf_max_chemprop_h=args.leaf_max_chemprop_h,
                leaf_chemical_popularity_logic=args.leaf_chemical_popularity_logic,
                leaf_min_chempop_reactants=args.leaf_min_chempop_reactants,
                leaf_min_chempop_products=args.leaf_min_chempop_products,
                leaf_chem_history_path=args.leaf_chem_history_path,
                nag2g_project_dir=args.nag2g_project_dir,
                nag2g_data_dir=args.nag2g_data_dir,
                nag2g_checkpoint_path=args.nag2g_checkpoint,
                nag2g_python=args.nag2g_python,
                nag2g_dict_name=args.nag2g_dict_name,
                nag2g_bpe_tokenizer_path=args.nag2g_bpe_tokenizer_path,
                nag2g_search_strategies=args.nag2g_search_strategies,
                nag2g_topk=args.nag2g_topk,
                nag2g_action_prob=args.nag2g_action_prob,
                nag2g_len_penalty=args.nag2g_len_penalty,
                nag2g_temperature=args.nag2g_temperature,
                nag2g_cpu=args.nag2g_cpu,
                nag2g_fp16=args.nag2g_fp16,
            )
        finally:
            sys.stdout, sys.stderr = _orig_out, _orig_err
