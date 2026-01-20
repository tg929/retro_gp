#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class RunInput:
    label: str
    per_target_path: Path


def _find_latest_csv(run_dir: Path, *, stem: str) -> Optional[Path]:
    direct = run_dir / f"{stem}.csv"
    if direct.exists():
        return direct
    candidates = sorted(run_dir.glob(f"{stem}_*.csv"), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    if candidates:
        return candidates[0]
    return None


def _load_run(run_dir: Path, *, label: Optional[str] = None) -> RunInput:
    if run_dir.is_file():
        if run_dir.suffix.lower() != ".csv":
            raise ValueError(f"Expected a .csv file or a run directory, got: {run_dir}")
        return RunInput(label=label or run_dir.stem, per_target_path=run_dir)

    per_target = _find_latest_csv(run_dir, stem="per_target")
    if per_target is None:
        raise FileNotFoundError(f"Cannot find per_target.csv (or per_target_*.csv) under: {run_dir}")

    return RunInput(label=label or run_dir.name, per_target_path=per_target)


def _compute_summary(per_target: pd.DataFrame) -> pd.DataFrame:
    required = {"target", "planner", "solved", "steps", "calls_used", "wall_time"}
    missing = sorted(required - set(per_target.columns))
    if missing:
        raise ValueError(f"per_target.csv missing columns: {missing}")

    df = per_target.copy()
    df["target"] = df["target"].astype(str)
    df["planner"] = df["planner"].astype(str)
    df["solved"] = df["solved"].astype(bool)

    n_targets = df["target"].nunique()

    def median_or_nan(xs: pd.Series) -> float:
        xs = xs.dropna()
        if xs.empty:
            return float("nan")
        return float(xs.median())

    def mean_or_nan(xs: pd.Series) -> float:
        xs = xs.dropna()
        if xs.empty:
            return float("nan")
        return float(xs.mean())

    out_rows: List[Dict[str, object]] = []
    for planner, g in df.groupby("planner", sort=True):
        solved_g = g[g["solved"]]
        out_rows.append(
            {
                "planner": planner,
                "n_targets": int(n_targets),
                "solved": int(solved_g.shape[0]),
                "success_rate": float(solved_g.shape[0] / max(1, n_targets)),
                "median_calls_to_solve": median_or_nan(solved_g["calls_used"]),
                "avg_steps_solved_only": mean_or_nan(solved_g["steps"]),
                "avg_wall_time": mean_or_nan(g["wall_time"]),
            }
        )

    return pd.DataFrame(out_rows).sort_values(["planner"]).reset_index(drop=True)


def _format_md_table(df: pd.DataFrame) -> str:
    # Keep it stable for copy/paste into slides.
    show = df.copy()
    for col in ["success_rate", "avg_steps_solved_only", "avg_wall_time", "median_calls_to_solve"]:
        if col in show.columns:
            show[col] = pd.to_numeric(show[col], errors="coerce")
    if "success_rate" in show.columns:
        show["success_rate"] = show["success_rate"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    if "median_calls_to_solve" in show.columns:
        show["median_calls_to_solve"] = show["median_calls_to_solve"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "")
    if "avg_steps_solved_only" in show.columns:
        show["avg_steps_solved_only"] = show["avg_steps_solved_only"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    if "avg_wall_time" in show.columns:
        show["avg_wall_time"] = show["avg_wall_time"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    return show.to_markdown(index=False)


def _compare_two_runs(a: Tuple[str, pd.DataFrame], b: Tuple[str, pd.DataFrame], *, topk: int) -> str:
    a_label, a_df = a
    b_label, b_df = b

    key_cols = ["target", "planner"]
    for df in (a_df, b_df):
        for c in key_cols:
            if c not in df.columns:
                raise ValueError(f"Missing column '{c}' for comparison")

    joined = a_df.merge(b_df, on=key_cols, how="outer", suffixes=(f"_{a_label}", f"_{b_label}"))
    for c in ["solved", "calls_used", "steps", "wall_time", "score"]:
        ca = f"{c}_{a_label}"
        cb = f"{c}_{b_label}"
        if ca in joined.columns and cb in joined.columns:
            # if missing in either run, keep NaN
            joined[ca] = pd.to_numeric(joined[ca], errors="coerce") if c != "solved" else joined[ca].astype("boolean")
            joined[cb] = pd.to_numeric(joined[cb], errors="coerce") if c != "solved" else joined[cb].astype("boolean")

    def _bool_col(name: str) -> pd.Series:
        if name not in joined.columns:
            return pd.Series([pd.NA] * len(joined), dtype="boolean")
        return joined[name].astype("boolean")

    solved_a = _bool_col(f"solved_{a_label}")
    solved_b = _bool_col(f"solved_{b_label}")
    changed = joined[(solved_a != solved_b) & solved_a.notna() & solved_b.notna()].copy()

    lines: List[str] = []
    lines.append(f"### Target-level changes ({a_label} → {b_label})")
    lines.append("")
    lines.append(f"- Total rows compared: {len(joined)}")
    lines.append(f"- Solved status changes: {len(changed)}")

    if changed.empty or topk <= 0:
        return "\n".join(lines)

    changed["delta_calls"] = (
        pd.to_numeric(changed.get(f"calls_used_{b_label}"), errors="coerce")
        - pd.to_numeric(changed.get(f"calls_used_{a_label}"), errors="coerce")
    )
    changed["delta_wall_time"] = (
        pd.to_numeric(changed.get(f"wall_time_{b_label}"), errors="coerce")
        - pd.to_numeric(changed.get(f"wall_time_{a_label}"), errors="coerce")
    )

    cols = [
        "planner",
        "target",
        f"solved_{a_label}",
        f"solved_{b_label}",
        f"calls_used_{a_label}",
        f"calls_used_{b_label}",
        "delta_calls",
        f"wall_time_{a_label}",
        f"wall_time_{b_label}",
        "delta_wall_time",
    ]
    cols = [c for c in cols if c in changed.columns]
    show = changed.sort_values(["planner", "delta_wall_time"], ascending=[True, False]).head(int(topk))[cols]

    lines.append("")
    lines.append(f"Top {topk} examples (sorted by delta_wall_time):")
    lines.append("")
    lines.append(show.to_markdown(index=False))
    return "\n".join(lines)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze benchmark_planners outputs (per_target.csv) and compare runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "runs",
        nargs="+",
        help="Run directories (containing per_target.csv) or a per_target.csv file path.",
    )
    p.add_argument(
        "--labels",
        type=str,
        default="",
        help="Comma-separated labels matching runs order (optional).",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Write report to this path (Markdown). Default: stdout.",
    )
    p.add_argument(
        "--show-changed-targets",
        type=int,
        default=20,
        help="When comparing exactly 2 runs, show up to N (target,planner) rows where solved status differs.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    run_paths = [Path(r) for r in args.runs]
    labels = [s.strip() for s in str(args.labels).split(",") if s.strip()]
    if labels and len(labels) != len(run_paths):
        raise ValueError(f"--labels count ({len(labels)}) must match runs count ({len(run_paths)})")

    runs: List[RunInput] = []
    for i, rp in enumerate(run_paths):
        runs.append(_load_run(rp, label=(labels[i] if labels else None)))

    sections: List[str] = []
    sections.append("# Benchmark analysis report")
    sections.append("")
    sections.append("## Inputs")
    for r in runs:
        sections.append(f"- `{r.label}`: `{r.per_target_path}`")

    per_target_by_label: Dict[str, pd.DataFrame] = {}
    summary_by_label: Dict[str, pd.DataFrame] = {}
    for r in runs:
        df = pd.read_csv(r.per_target_path)
        per_target_by_label[r.label] = df
        summary_by_label[r.label] = _compute_summary(df)

    sections.append("")
    sections.append("## Per-run summary")
    for label, s in summary_by_label.items():
        sections.append("")
        sections.append(f"### {label}")
        sections.append("")
        sections.append(_format_md_table(s))

    if len(runs) == 2:
        a, b = runs[0], runs[1]
        sections.append("")
        sections.append("## Two-run comparison")
        sections.append("")
        sections.append(_compare_two_runs((a.label, per_target_by_label[a.label]), (b.label, per_target_by_label[b.label]), topk=int(args.show_changed_targets)))

    report = "\n".join(sections).rstrip() + "\n"
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

