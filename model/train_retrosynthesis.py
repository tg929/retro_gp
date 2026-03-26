"""Compatibility shim for legacy Stage1/2 training entry.

Main Stage1/2 implementation now lives in `model/stage12model/train_retrosynthesis.py`.
This shim keeps old imports/entrypoints working.
"""

from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from stage12model.train_retrosynthesis import *  # noqa: F401,F403
from stage12model.train_retrosynthesis import main as _stage12_main


if __name__ == "__main__":
    _stage12_main()
