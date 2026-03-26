"""Compatibility shim for legacy Stage1/2 checkpoint evaluation entry.

Main Stage1/2 implementation now lives in `model/stage12model/evaluate_checkpoint.py`.
"""

from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from stage12model.evaluate_checkpoint import main as _stage12_eval_main


if __name__ == "__main__":
    _stage12_eval_main()
