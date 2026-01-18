from __future__ import annotations

"""
Planner-only benchmark runner.

Goal: compare multiple multi-step planners under a unified "one-step calls budget"
using the same one-step model (NAG2G) and the same stock/leaf criteria.
"""

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

# Make repo imports work when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gp_core.budget import BudgetedOneStepModel, CallCounter  # noqa: E402
from gp_core.scoring import GPFitnessEvaluator, StepScorer, StepScoreConfig  # noqa: E402
from gp_core.search import run_gp_for_target  # noqa: E402
from gp_retro_feas import ExecutePolicy, FeasibleExecutor  # noqa: E402
from gp_retro_nn import NAG2GSubprocessConfig, NAG2GSubprocessModel  # noqa: E402
from gp_retro_repr import Inventory, LeafCriteria, LeafCriteriaConfig, ReactionTemplateRegistry, ChemHistorian  # noqa: E402

from gp_core.baselines.greedy import run_greedy  # noqa: E402
from gp_core.baselines.beam import run_beam  # noqa: E402
from gp_core.baselines.best_first import run_best_first  # noqa: E402


def _read_smiles_first_col_csv(path: Path, *, limit: Optional[int] = None) -> List[str]:
    out: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row_i, row in enumerate(reader):
            if not row:
                continue
            first = (row[0] or "").strip()
            if row_i == 0 and first.lower() in {"smiles", "smi", "target", "target_smiles"}:
                continue
            if first:
                out.append(first)
                if limit is not None and len(out) >= int(limit):
                    break
    return out


def _read_smiles_lines(path: Path, *, limit: Optional[int] = None) -> List[str]:
    out: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            s = (ln or "").strip()
            if not s or s.startswith("#"):
                continue
            out.append(s.split()[0])
            if limit is not None and len(out) >= int(limit):
                break
    return out


def load_targets(path: str, *, limit: Optional[int] = None) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"targets file not found: {p}")
    if p.suffix.lower() == ".csv":
        return _read_smiles_first_col_csv(p, limit=limit)
    return _read_smiles_lines(p, limit=limit)


def load_inventory(stock_file: str) -> Inventory:
    p = Path(stock_file)
    if not p.exists():
        raise FileNotFoundError(f"stock file not found: {p}")

    inv = Inventory()
    suffix = p.suffix.lower()
    if suffix == ".csv":
        import csv as _csv

        with p.open("r", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                smi = (row.get("smiles") or row.get("SMILES") or "").strip()
                if smi:
                    inv.add(smi)
    else:
        with p.open("r", encoding="utf-8") as f:
            for ln in f:
                s = (ln or "").strip()
                if not s or s.startswith("#"):
                    continue
                inv.add(s.split()[0])
    return inv


def apply_leaf_criteria_from_args(inv: Inventory, args: argparse.Namespace) -> None:
    leaf_enabled = str(args.leaf_chemical_property_logic).lower() not in ["none", ""] or str(
        args.leaf_chemical_popularity_logic
    ).lower() not in ["none", ""]
    if not leaf_enabled:
        inv.set_leaf_criteria(None)
        return

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
            raise ValueError("--leaf-chem-history-path is required when popularity leaf is enabled")
        chemhistorian = ChemHistorian()
        chemhistorian.load_from_file(str(args.leaf_chem_history_path))

    leaf_cfg = LeafCriteriaConfig(
        max_natom_dict=max_natom_dict,
        min_chemical_history_dict=min_hist_dict,
    )
    inv.set_leaf_criteria(LeafCriteria(cfg=leaf_cfg, chemhistorian=chemhistorian))


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], *, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    tmp_path.replace(path)


def _safe_median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return float(statistics.median(xs))


def _safe_mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def summarize(rows: Sequence[Dict[str, Any]], *, planners: Sequence[str], n_targets: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in planners:
        pr = [r for r in rows if r.get("planner") == p]
        solved = [r for r in pr if bool(r.get("solved"))]
        solved_calls = [float(r.get("calls_used") or 0) for r in solved]
        solved_steps = [float(r.get("steps") or 0) for r in solved]
        wall_times = [float(r.get("wall_time") or 0.0) for r in pr]
        out.append(
            {
                "planner": p,
                "n_targets": int(n_targets),
                "solved": int(len(solved)),
                "success_rate": float(len(solved) / max(1, n_targets)),
                "median_calls_to_solve": _safe_median(solved_calls),
                "avg_steps_solved_only": _safe_mean(solved_steps),
                "avg_wall_time": _safe_mean(wall_times),
            }
        )
    return out


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark planner-only variants under a unified one-step calls budget.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--targets-file",
        type=str,
        default="data/target molecular/test_synthesis.csv",
        help="Targets list file; CSV uses first column as SMILES.",
    )
    p.add_argument("--limit", type=int, default=10, help="Limit number of targets (smoke test).")
    p.add_argument(
        "--out-dir",
        type=str,
        default="logs/benchmarks",
        help="Output directory for per_target.csv and summary.csv.",
    )
    p.add_argument(
        "--flush-every",
        type=int,
        default=0,
        help="Write per_target.csv and summary.csv every N targets (0 disables incremental writes).",
    )

    # Fixed stock / leaf criteria (ASKCOS-style)
    p.add_argument(
        "--stock-file",
        type=str,
        default="data/building_block/building_blocks_dataset.txt",
        help="Stock (building blocks) file path.",
    )
    p.add_argument(
        "--leaf-chemical-property-logic",
        type=str,
        default="none",
        choices=["none", "or", "and"],
    )
    p.add_argument("--leaf-max-chemprop-c", type=int, default=0)
    p.add_argument("--leaf-max-chemprop-n", type=int, default=0)
    p.add_argument("--leaf-max-chemprop-o", type=int, default=0)
    p.add_argument("--leaf-max-chemprop-h", type=int, default=0)
    p.add_argument(
        "--leaf-chemical-popularity-logic",
        type=str,
        default="none",
        choices=["none", "or"],
    )
    p.add_argument("--leaf-min-chempop-reactants", type=int, default=5)
    p.add_argument("--leaf-min-chempop-products", type=int, default=5)
    p.add_argument("--leaf-chem-history-path", type=str, default=None)

    # One-step (NAG2G)
    p.add_argument("--nag2g-python", type=str, required=True, help="Python executable in the NAG2G runtime env.")
    p.add_argument("--nag2g-project-dir", type=str, default="NAG2G-main")
    p.add_argument(
        "--nag2g-data-dir",
        type=str,
        default="NAG2G-main/USPTO50K_brief_20230227/USPTO50K_brief_20230227",
    )
    p.add_argument(
        "--nag2g-checkpoint",
        type=str,
        default="NAG2G-main/NAG2G_unimolplus_uspto_50k_20230513-222355/checkpoint_last.pt",
    )
    p.add_argument("--nag2g-dict-name", type=str, default="dict_20230310.txt")
    p.add_argument("--nag2g-bpe-tokenizer-path", type=str, default="none")
    p.add_argument(
        "--nag2g-search-strategies",
        type=str,
        default="SimpleGenerator",
        choices=["SimpleGenerator", "SequenceGeneratorBeamSearch"],
    )
    p.add_argument("--nag2g-len-penalty", type=float, default=0.0)
    p.add_argument("--nag2g-temperature", type=float, default=1.0)
    p.add_argument("--nag2g-cpu", action="store_true")
    p.add_argument("--nag2g-fp16", action="store_true")

    # Shared expansion knobs
    p.add_argument("--one-step-topk", type=int, default=10, help="Candidates requested from NAG2G per expansion.")
    p.add_argument("--one-step-topb", type=int, default=10, help="Candidates kept after StepScorer re-ranking.")
    p.add_argument(
        "--step-score-type",
        type=str,
        default="rank",
        choices=["rank", "raw", "logprob", "prob"],
        help="How StepScorer interprets OneStepPrediction.score.",
    )
    p.add_argument("--rank-temperature", type=float, default=2.0, help="Only used when step-score-type=rank.")

    p.add_argument("--calls-budget", type=int, default=50, help="Max one-step inference calls per planner run.")
    p.add_argument("--max-depth", type=int, default=10, help="Max multi-step depth for baselines and GP programs.")

    # Baselines
    p.add_argument("--beam-width", type=int, default=5)
    p.add_argument("--best-first-strategy", type=str, default="astar0", choices=["astar0"])

    # GP
    p.add_argument("--gp-pop-size", type=int, default=10)
    p.add_argument("--gp-generations", type=int, default=10)
    p.add_argument("--gp-seeds", type=str, default="123", help="Comma-separated seeds (multi-restart within budget).")

    p.add_argument(
        "--planners",
        type=str,
        default="gp,greedy,beam,best_first",
        help="Comma-separated planners: gp,greedy,beam,best_first",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    planners = [p.strip() for p in str(args.planners).split(",") if p.strip()]
    if not planners:
        raise ValueError("--planners is empty")

    targets = load_targets(str(args.targets_file), limit=int(args.limit) if args.limit is not None else None)
    if not targets:
        raise ValueError(f"No targets loaded from {args.targets_file}")

    stock = load_inventory(str(args.stock_file))
    apply_leaf_criteria_from_args(stock, args)

    # Shared StepScorer ranker
    step_cfg = StepScoreConfig(
        score_type=str(args.step_score_type),
        topN=int(args.one_step_topk),
        topB=int(args.one_step_topb),
        rank_temperature=float(args.rank_temperature),
    )
    step_scorer = StepScorer(config=step_cfg, bb_is_purchasable=stock.is_purchasable, sa_fn=None, forward_model=None)
    one_step_ranker: Callable[[str, List[Any]], List[Any]] = lambda product_smiles, preds: step_scorer.rank_and_truncate(
        product_smiles, preds, topB=int(args.one_step_topb)
    )

    # Shared executor policy (one-step only)
    policy = ExecutePolicy(one_step_topk=int(args.one_step_topk))

    # NAG2G model (shared subprocess)
    nag2g = NAG2GSubprocessModel(
        NAG2GSubprocessConfig(
            project_dir=Path(args.nag2g_project_dir),
            data_dir=Path(args.nag2g_data_dir),
            checkpoint_path=Path(args.nag2g_checkpoint),
            python_executable=str(args.nag2g_python),
            dict_name=str(args.nag2g_dict_name),
            bpe_tokenizer_path=str(args.nag2g_bpe_tokenizer_path),
            beam_size=int(args.one_step_topk),
            search_strategies=str(args.nag2g_search_strategies),
            len_penalty=float(args.nag2g_len_penalty),
            temperature=float(args.nag2g_temperature),
            cpu=bool(args.nag2g_cpu),
            fp16=bool(args.nag2g_fp16),
        )
    )

    out_dir = Path(args.out_dir)
    per_target_path = out_dir / "per_target.csv"
    summary_path = out_dir / "summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    reg = ReactionTemplateRegistry()  # planner-only: no templates needed, but executor requires a registry

    flush_every = int(getattr(args, "flush_every", 0) or 0)
    last_flushed_targets = 0

    def _n_targets_done() -> int:
        return int(len({str(r.get("target")) for r in rows if r.get("target") is not None}))

    def flush(*, force: bool = False) -> None:
        nonlocal last_flushed_targets
        if not rows:
            return
        n_done = _n_targets_done()
        if not force and n_done <= last_flushed_targets:
            return
        fieldnames = ["target", "planner", "solved", "steps", "calls_used", "wall_time", "score", "stop_reason", "error"]
        _write_csv(per_target_path, rows, fieldnames=fieldnames)
        summary_rows = summarize(rows, planners=planners, n_targets=n_done)
        _write_csv(
            summary_path,
            summary_rows,
            fieldnames=[
                "planner",
                "n_targets",
                "solved",
                "success_rate",
                "median_calls_to_solve",
                "avg_steps_solved_only",
                "avg_wall_time",
            ],
        )
        last_flushed_targets = n_done
        print(f"[checkpoint] wrote {per_target_path} (targets_done={n_done})")
        print(f"[checkpoint] wrote {summary_path} (targets_done={n_done})")

    try:
        for ti, target in enumerate(targets, start=1):
            print(f"[{ti}/{len(targets)}] target={target}")

            # Per-target evaluator for consistent scoring output (optional column).
            fitness_eval = GPFitnessEvaluator(target_smiles=target, bb_is_purchasable=stock.is_leaf)
            score_fn = lambda r: float(fitness_eval.evaluate(r).scalar)

            for planner in planners:
                if planner == "gp":
                    # Per-run fairness: clear one-step cache so other planners don't get free hits.
                    cache = getattr(nag2g, "_cache", None)
                    if isinstance(cache, dict):
                        cache.clear()

                    counter = CallCounter()
                    budgeted = BudgetedOneStepModel(nag2g, counter=counter, max_calls=int(args.calls_budget))
                    exe = FeasibleExecutor(reg, inventory=stock, policy=policy, one_step_model=budgeted, one_step_ranker=one_step_ranker)

                    # Align GP program length with max_depth
                    import gp_core.config as gp_cfg

                    gp_cfg.max_templates_per_prog = int(args.max_depth)

                    best_ind: Optional[Dict[str, Any]] = None
                    t0 = time.perf_counter()
                    seeds = [int(s.strip()) for s in str(args.gp_seeds).split(",") if s.strip()]
                    for seed in seeds:
                        pop, _ = run_gp_for_target(
                            target=target,
                            inventory=stock,
                            reg=reg,
                            evaluator=fitness_eval,
                            exe=exe,
                            pop_size=int(args.gp_pop_size),
                            generations=int(args.gp_generations),
                            seed=int(seed),
                            template_pool=[],
                            init_templates=[],
                            feasible_templates_for_target=None,
                            allow_model_actions=True,
                            model_rank_pool=list(range(max(1, int(args.one_step_topb)))),
                            p_model_action=1.0,
                        )
                        if pop:
                            cand = pop[0]
                            if best_ind is None or float(cand["fitness"].scalar) > float(best_ind["fitness"].scalar):
                                best_ind = cand
                        if int(counter.one_step_calls) >= int(args.calls_budget):
                            break
                        if best_ind is not None and best_ind["fitness"].objectives.get("solved", 0) > 0.5:
                            break

                    wall_time = float(time.perf_counter() - t0)
                    counter.stop()

                    solved = bool(best_ind and best_ind["fitness"].objectives.get("solved", 0) > 0.5)
                    steps = int(len(best_ind["route"].steps)) if best_ind else 0
                    score = float(best_ind["fitness"].scalar) if best_ind else float("nan")
                    stop_reason = (
                        "solved"
                        if solved
                        else ("budget_exceeded" if int(counter.one_step_calls) >= int(args.calls_budget) else "finished")
                    )
                    error = None
                    if best_ind is not None:
                        error = (getattr(best_ind["fitness"], "extra", {}) or {}).get("invalid_error")

                    rows.append(
                        {
                            "target": target,
                            "planner": "gp",
                            "solved": bool(solved),
                            "steps": int(steps),
                            "calls_used": int(counter.one_step_calls),
                            "wall_time": float(wall_time),
                            "score": score,
                            "stop_reason": stop_reason,
                            "error": error,
                        }
                    )
                    continue

                # Baselines use the same executor wiring (same model + ranker + policy),
                # but manage their own per-run counter/budget internally.
                exe = FeasibleExecutor(reg, inventory=stock, policy=policy, one_step_model=nag2g, one_step_ranker=one_step_ranker)

                if planner == "greedy":
                    res = run_greedy(target, executor=exe, max_depth=int(args.max_depth), max_calls=int(args.calls_budget))
                elif planner == "beam":
                    res = run_beam(
                        target,
                        beam_width=int(args.beam_width),
                        executor=exe,
                        max_depth=int(args.max_depth),
                        max_calls=int(args.calls_budget),
                        state_score_fn=score_fn,
                    )
                elif planner == "best_first":
                    res = run_best_first(
                        target,
                        executor=exe,
                        max_depth=int(args.max_depth),
                        max_calls=int(args.calls_budget),
                        strategy=str(args.best_first_strategy),
                    )
                else:
                    raise ValueError(f"Unknown planner: {planner}")

                rows.append(
                    {
                        "target": target,
                        "planner": planner,
                        "solved": bool(res.solved),
                        "steps": int(res.steps),
                        "calls_used": int(res.calls_used),
                        "wall_time": float(res.wall_time),
                        "score": float(score_fn(res.route)),
                        "stop_reason": (res.extra or {}).get("stop_reason"),
                        "error": (res.extra or {}).get("error"),
                    }
                )

            if flush_every > 0 and (ti % flush_every == 0):
                flush()

    except KeyboardInterrupt:
        flush(force=True)
        raise
    except Exception:
        flush(force=True)
        raise
    finally:
        try:
            nag2g.close()
        except Exception:
            pass
        flush(force=True)

    fieldnames = ["target", "planner", "solved", "steps", "calls_used", "wall_time", "score", "stop_reason", "error"]
    _write_csv(per_target_path, rows, fieldnames=fieldnames)

    summary_rows = summarize(rows, planners=planners, n_targets=len(targets))
    _write_csv(
        summary_path,
        summary_rows,
        fieldnames=["planner", "n_targets", "solved", "success_rate", "median_calls_to_solve", "avg_steps_solved_only", "avg_wall_time"],
    )

    print(f"[done] wrote {per_target_path}")
    print(f"[done] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
