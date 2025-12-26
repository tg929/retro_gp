from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Make repo imports work when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gp_core.scoring import StepScoreConfig, StepScorer  # noqa: E402
from gp_retro_nn import NAG2GSubprocessConfig, NAG2GSubprocessModel  # noqa: E402


@dataclass(frozen=True)
class AutoPaths:
    data_dir: Path
    checkpoint: Path


def _auto_detect_paths(project_dir: Path, dict_name: str) -> AutoPaths:
    if not project_dir.exists():
        raise FileNotFoundError(f"--project-dir not found: {project_dir}")

    candidates = list(project_dir.rglob(dict_name))
    if not candidates:
        raise FileNotFoundError(f"Could not find dict file {dict_name} under {project_dir}")
    data_dir = candidates[0].parent

    ckpts = list(project_dir.rglob("checkpoint_last.pt"))
    if not ckpts:
        ckpts = list(project_dir.rglob("checkpoint*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"Could not find checkpoint under {project_dir}")
    checkpoint = ckpts[0]

    return AutoPaths(data_dir=data_dir, checkpoint=checkpoint)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Inspect NAG2G one-step outputs and StepScorer (hard-filter + soft-rules) re-ranking.\n\n"
            "This prints:\n"
            "  1) raw NAG2G candidates: rank / model_score / reactants\n"
            "  2) StepScorer outputs: step_score / meta / reactants (after canonicalization & filtering)\n"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nag2g-python", required=True, help="Python executable in the NAG2G runtime env")
    p.add_argument("--project-dir", default="NAG2G-main", help="Path to NAG2G-main")
    p.add_argument("--data-dir", default=None, help="NAG2G data dir (contains dict); auto-detect if omitted")
    p.add_argument("--checkpoint", default=None, help="Checkpoint .pt; auto-detect if omitted")
    p.add_argument("--dict-name", default="dict_20230310.txt", help="Dictionary filename inside data-dir")

    p.add_argument(
        "--search-strategies",
        default="SequenceGeneratorBeamSearch",
        help="NAG2G search strategy (SequenceGeneratorBeamSearch recommended for model_score)",
    )
    p.add_argument("--beam-size", type=int, default=10, help="Beam size in NAG2G")
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for NAG2G server")
    p.add_argument("--topk", type=int, default=10, help="How many raw candidates to request from NAG2G")

    p.add_argument("--topb", type=int, default=5, help="How many candidates to keep after StepScorer")
    p.add_argument(
        "--score-type",
        default=None,
        choices=["rank", "raw", "logprob", "prob"],
        help="How StepScorer interprets OneStepPrediction.score; auto if omitted",
    )
    p.add_argument("--rank-temperature", type=float, default=2.0, help="Only used when score-type=rank")
    p.add_argument("--max-reactants", type=int, default=8, help="Hard-filter max reactant molecules per candidate")
    p.add_argument("--reactant-soft-limit", type=int, default=2, help="Soft-penalty threshold for reactant count")
    p.add_argument("--max-smiles-len", type=int, default=300, help="Hard-filter max SMILES length")
    p.add_argument("--no-dedup-reactants", action="store_true", help="Disable dedup within a candidate")
    p.add_argument("--no-dedup-sets", action="store_true", help="Disable dedup across candidate sets")
    p.add_argument("--allow-loops", action="store_true", help="Allow reactant == product (loop) candidates")

    p.add_argument(
        "--product",
        default="NC(=O)c1cn(Cc2c(F)cccc2F)nn1",
        help="Product SMILES to query",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    project_dir = Path(args.project_dir).resolve()

    if int(args.topk) > int(args.beam_size):
        print(
            f"[inspect] topk({args.topk}) > beam_size({args.beam_size}); bump beam_size -> {args.topk}.",
            file=sys.stderr,
        )
        args.beam_size = int(args.topk)

    if args.data_dir and args.checkpoint:
        data_dir = Path(args.data_dir).resolve()
        checkpoint = Path(args.checkpoint).resolve()
    else:
        auto = _auto_detect_paths(project_dir, str(args.dict_name))
        data_dir = Path(args.data_dir).resolve() if args.data_dir else auto.data_dir
        checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else auto.checkpoint

    one_step = NAG2GSubprocessModel(
        NAG2GSubprocessConfig(
            project_dir=project_dir,
            data_dir=data_dir,
            checkpoint_path=checkpoint,
            python_executable=str(args.nag2g_python),
            dict_name=str(args.dict_name),
            search_strategies=str(args.search_strategies),
            beam_size=int(args.beam_size),
            batch_size=int(args.batch_size),
        )
    )
    try:
        preds = one_step.predict(str(args.product), topk=int(args.topk))

        print("\n=== NAG2G raw candidates ===")
        print(f"product: {args.product}")
        print(f"project_dir: {project_dir}")
        print(f"data_dir: {data_dir}")
        print(f"checkpoint: {checkpoint}")
        print(f"search_strategies: {args.search_strategies}  beam_size: {args.beam_size}  topk: {args.topk}")
        for p in preds:
            r = (p.meta or {}).get("rank")
            print(f"rank={int(r) if r is not None else 'NA':>2}  model_score={p.score}  reactants={'.'.join(p.reactants)}")

        score_type = args.score_type
        if score_type is None:
            score_type = "raw" if str(args.search_strategies) == "SequenceGeneratorBeamSearch" else "rank"

        scorer = StepScorer(
            config=StepScoreConfig(
                score_type=str(score_type),
                topB=int(args.topb),
                max_reactants=int(args.max_reactants),
                max_smiles_len=int(args.max_smiles_len),
                reactant_soft_limit=int(args.reactant_soft_limit),
                dedup_reactants=not bool(args.no_dedup_reactants),
                dedup_candidate_sets=not bool(args.no_dedup_sets),
                reject_loops=not bool(args.allow_loops),
                rank_temperature=float(args.rank_temperature),
            ),
            bb_is_purchasable=lambda _s: False,
            sa_fn=lambda s: float(len(s or "")),
            forward_model=None,
        )
        ranked = scorer.rank_and_truncate(str(args.product), preds)

        print("\n=== After StepScorer (hard-filter + soft-rules) ===")
        print(
            f"score_type={score_type}  topB={args.topb}  max_reactants={args.max_reactants}  "
            f"reactant_soft_limit={args.reactant_soft_limit}  rank_temperature={args.rank_temperature}"
        )
        for p in ranked:
            meta = p.meta or {}
            step_meta = meta.get("step_meta") or {}
            print(
                f"step_score={meta.get('step_score'):>10.4f}  "
                f"rank={meta.get('rank')}  n_react={meta.get('n_reactants')}  "
                f"salt_ion={meta.get('n_salt_like_ions')}  progress={step_meta.get('progress')}"
            )
            print(f"  reactants={'.'.join(p.reactants)}")
        return 0
    finally:
        one_step.close()


if __name__ == "__main__":
    raise SystemExit(main())
