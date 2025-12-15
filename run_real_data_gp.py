"""Run GP search on real data using modular gp_core helpers."""
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import yaml

from gp_core import config
from gp_core.data_loading import load_inventory_and_templates
from gp_core.templates import template_ids
from gp_core.executor import make_executor
from gp_core.fitness import (
    build_objectives,
    build_scscore_with_cache,
    build_partial_reward,
    make_evaluator,
)
from gp_core.search import run_gp_for_target
from gp_core.metrics import MetricsHistory
from gp_retro_feas import ActionMaskBuilder, ExecutePolicy

try:  # optional neural one-step model
    from gp_retro_nn import Seq2SeqSubprocessConfig, Seq2SeqSubprocessModel
except Exception:  # pragma: no cover
    Seq2SeqSubprocessConfig = None  # type: ignore
    Seq2SeqSubprocessModel = None  # type: ignore


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


from typing import Optional


def run(
    target_key: str = "all",
    template_file: Optional[str] = None,
    *,
    seq2seq_project_dir: Optional[str] = None,
    seq2seq_model_dir: Optional[str] = None,
    seq2seq_checkpoint_path: Optional[str] = None,
    seq2seq_vocab_path: Optional[str] = None,
    seq2seq_python: Optional[str] = None,
    seq2seq_beam_width: int = 10,
    seq2seq_topk: int = 10,
    seq2seq_action_prob: float = 0.0,
    seq2seq_rxn_classes: Optional[List[str]] = None,
):
    # Ensure log directory exists before any cache writes
    Path("logs").mkdir(exist_ok=True)
    # SCScore + partial reward (multi-score if available)
    sc_fn, sc_cache = build_scscore_with_cache(
        cache_path=str(Path("logs") / "scscore_cache.jsonl")
    )
    partial_reward_fn = build_partial_reward(sc_fn, cache=sc_cache)
    specs = build_objectives(config.objective_weights)
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
        float(seq2seq_action_prob) > 0.0
        or seq2seq_model_dir
        or seq2seq_checkpoint_path
        or seq2seq_project_dir
        or seq2seq_vocab_path
    ):
        if Seq2SeqSubprocessModel is None or Seq2SeqSubprocessConfig is None:
            raise RuntimeError("gp_retro_nn is not available; cannot enable seq2seq one-step model")
        proj = Path(seq2seq_project_dir or (Path(__file__).resolve().parent / "reaction_prediction_seq2seq-master"))
        model_dir = Path(seq2seq_model_dir) if seq2seq_model_dir else (proj / "model_working_directory" / "jul_5_2017_1")
        ckpt = Path(seq2seq_checkpoint_path) if seq2seq_checkpoint_path else (model_dir / "model.ckpt-44001")
        vocab = Path(seq2seq_vocab_path) if seq2seq_vocab_path else (proj / "processed_data" / "vocab")
        rxn_classes = seq2seq_rxn_classes or [
            "<RX_1>", "<RX_2>", "<RX_3>", "<RX_4>", "<RX_5>",
            "<RX_6>", "<RX_7>", "<RX_8>", "<RX_9>", "<RX_10>",
        ]
        one_step_model = Seq2SeqSubprocessModel(
            Seq2SeqSubprocessConfig(
                project_dir=proj,
                model_dir=model_dir,
                checkpoint_path=ckpt,
                vocab_path=vocab,
                python_executable=seq2seq_python,
                beam_width=int(seq2seq_beam_width),
                rxn_classes=rxn_classes,
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

        policy = ExecutePolicy(one_step_topk=int(seq2seq_topk))
        exe: FeasibleExecutor = make_executor(reg, inventory, policy=policy, one_step_model=one_step_model)
        audit_fn = make_audit_fn(inventory, target)
        evaluator = make_evaluator(
            specs,
            inventory,
            audit_fn,
            sc_fn,
            target,
            partial_reward_fn=partial_reward_fn,
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
            allow_model_actions=bool(one_step_model) and float(seq2seq_action_prob) > 0.0,
            model_rank_pool=list(range(max(1, int(seq2seq_topk)))) if one_step_model else None,
            p_model_action=float(seq2seq_action_prob),
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
    parser.add_argument(
        "--seq2seq-project-dir",
        type=str,
        default=None,
        help="Path to reaction_prediction_seq2seq-master (default: ./reaction_prediction_seq2seq-master).",
    )
    parser.add_argument(
        "--seq2seq-model-dir",
        type=str,
        default=None,
        help="Seq2seq model dir (contains train_options.json and checkpoints).",
    )
    parser.add_argument(
        "--seq2seq-checkpoint",
        type=str,
        default=None,
        help="Full path to checkpoint (e.g., model.ckpt-44001).",
    )
    parser.add_argument(
        "--seq2seq-vocab",
        type=str,
        default=None,
        help="Vocab file path (default: processed_data/vocab under project dir).",
    )
    parser.add_argument(
        "--seq2seq-python",
        type=str,
        default=None,
        help="Python executable used to run seq2seq inference (use a TF1-compatible env).",
    )
    parser.add_argument(
        "--seq2seq-beam-width",
        type=int,
        default=10,
        help="Beam width used by the seq2seq decoder.",
    )
    parser.add_argument(
        "--seq2seq-topk",
        type=int,
        default=10,
        help="How many one-step candidates to request from the model per product.",
    )
    parser.add_argument(
        "--seq2seq-action-prob",
        type=float,
        default=0.0,
        help="Probability that a GP gene uses ApplyOneStepModel instead of ApplyTemplate (0 disables model actions).",
    )
    parser.add_argument(
        "--seq2seq-rxn-classes",
        type=str,
        nargs="*",
        default=None,
        help="Reaction class tokens to try (e.g., <RX_1> <RX_2> ...). Default: all 10 classes.",
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
                seq2seq_project_dir=args.seq2seq_project_dir,
                seq2seq_model_dir=args.seq2seq_model_dir,
                seq2seq_checkpoint_path=args.seq2seq_checkpoint,
                seq2seq_vocab_path=args.seq2seq_vocab,
                seq2seq_python=args.seq2seq_python,
                seq2seq_beam_width=args.seq2seq_beam_width,
                seq2seq_topk=args.seq2seq_topk,
                seq2seq_action_prob=args.seq2seq_action_prob,
                seq2seq_rxn_classes=args.seq2seq_rxn_classes,
            )
        finally:
            sys.stdout, sys.stderr = _orig_out, _orig_err
