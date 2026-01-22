"""Plot GP per-generation metrics from benchmark logs.

Usage example:
  python scripts/plot_gp_gen.py --gen-dir logs/benchmarks/smoke_leaf-all/gp_gen --target-idx 1
  # or plot all targets in the directory:
  python scripts/plot_gp_gen.py --gen-dir logs/benchmarks/smoke_leaf-all/gp_gen
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def _load_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: v for k, v in row.items()})
    return rows


def _plot_one(path: Path, out_dir: Path, title: Optional[str] = None) -> None:
    rows = _load_csv(path)
    if not rows:
        return

    seeds = sorted({r.get("seed", "") for r in rows})
    fig, ax = plt.subplots(figsize=(6, 4))

    for seed in seeds:
        xs: List[int] = []
        bests: List[float] = []
        means: List[float] = []
        solveds: List[int] = []
        for r in rows:
            if str(r.get("seed", "")) != str(seed):
                continue
            try:
                xs.append(int(r.get("gen", 0)))
                bests.append(float(r.get("best", 0.0)))
                means.append(float(r.get("mean", 0.0)))
                solveds.append(int(r.get("solved", 0)))
            except Exception:
                continue
        if not xs:
            continue
        ax.plot(xs, bests, marker="o", label=f"seed {seed} best")
        ax.plot(xs, means, linestyle="--", label=f"seed {seed} mean")
        ax.step(xs, solveds, where="post", alpha=0.5, label=f"seed {seed} solved")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (best/mean); solved count (steps)")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    ax.legend()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (path.stem + ".png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot GP per-generation metrics from gp_gen CSV logs.")
    ap.add_argument("--gen-dir", required=True, help="Directory containing per-target gp_gen CSV files.")
    ap.add_argument("--target-idx", type=int, default=None, help="If set, plot only this target index (e.g., 1).")
    ap.add_argument("--out-dir", default=None, help="Directory to write plots (default: <gen-dir>/plots).")
    args = ap.parse_args()

    gen_dir = Path(args.gen_dir)
    if not gen_dir.exists():
        raise FileNotFoundError(f"gen-dir not found: {gen_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else gen_dir / "plots"

    files = sorted(gen_dir.glob("*.csv"))
    if args.target_idx is not None:
        files = [p for p in files if p.stem == f"{args.target_idx:04d}"]
    if not files:
        raise FileNotFoundError("No gp_gen CSV files found matching criteria.")

    for p in files:
        title = f"Target {p.stem}"
        _plot_one(p, out_dir, title=title)

    print(f"Wrote plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
