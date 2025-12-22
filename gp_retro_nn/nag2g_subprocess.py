from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .one_step import OneStepPrediction


@dataclass(frozen=True)
class NAG2GSubprocessConfig:
    project_dir: Path
    data_dir: Path
    checkpoint_path: Path
    python_executable: Optional[str] = None
    dict_name: str = "dict.txt"
    bpe_tokenizer_path: str = "none"
    beam_size: int = 10
    search_strategies: str = "SimpleGenerator"
    len_penalty: float = 0.0
    temperature: float = 1.0
    seed: int = 42
    batch_size: int = 32
    cpu: bool = False
    fp16: bool = False


class NAG2GSubprocessModel:
    """One-step retro model wrapper for NAG2G-main using a persistent subprocess server."""

    name = "nag2g"

    def __init__(self, cfg: NAG2GSubprocessConfig):
        self.cfg = cfg
        self._cache: Dict[Tuple[str, int], List[OneStepPrediction]] = {}

        if not cfg.project_dir.exists():
            raise FileNotFoundError(f"nag2g project_dir not found: {cfg.project_dir}")
        if not cfg.data_dir.exists():
            raise FileNotFoundError(f"nag2g data_dir not found: {cfg.data_dir}")
        if not cfg.checkpoint_path.exists():
            raise FileNotFoundError(f"nag2g checkpoint not found: {cfg.checkpoint_path}")

        cmd = [
            cfg.python_executable or sys.executable,
            "-m",
            "gp_retro_nn.nag2g_server",
            "--project-dir",
            str(cfg.project_dir),
            "--data-dir",
            str(cfg.data_dir),
            "--dict-name",
            str(cfg.dict_name),
            "--bpe-tokenizer-path",
            str(cfg.bpe_tokenizer_path),
            "--checkpoint",
            str(cfg.checkpoint_path),
            "--search-strategies",
            str(cfg.search_strategies),
            "--beam-size",
            str(int(cfg.beam_size)),
            "--len-penalty",
            str(float(cfg.len_penalty)),
            "--temperature",
            str(float(cfg.temperature)),
            "--seed",
            str(int(cfg.seed)),
            "--batch-size",
            str(int(cfg.batch_size)),
        ]
        if cfg.cpu:
            cmd.append("--cpu")
        if cfg.fp16:
            cmd.append("--fp16")

        self._proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parents[1]),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # keep protocol clean; logs go to parent stderr
            text=True,
            bufsize=1,
        )

        ready = self._read_json(skip_non_json=True)
        if not ready.get("ok"):
            raise RuntimeError(f"NAG2G server init failed: {ready}")

    def close(self) -> None:
        if getattr(self, "_proc", None) is None:
            return
        proc = self._proc
        self._proc = None  # type: ignore
        try:
            if proc.stdin:
                proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _read_json(self, *, skip_non_json: bool = False) -> Dict:
        if not self._proc or not self._proc.stdout:
            raise RuntimeError("NAG2G server is not running")
        last_non_json: List[str] = []
        while True:
            line = self._proc.stdout.readline()
            if not line:
                rc = self._proc.poll()
                hint = "\n".join(last_non_json[-20:])
                raise RuntimeError(f"NAG2G server stdout closed (rc={rc}). Last output:\n{hint}")
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                if not skip_non_json:
                    raise
                last_non_json.append(line.rstrip("\n"))

    def _write_json(self, obj: Dict) -> None:
        if not self._proc or not self._proc.stdin:
            raise RuntimeError("NAG2G server is not running")
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def predict(self, product_smiles: str, topk: int) -> List[OneStepPrediction]:
        key = (product_smiles, int(topk))
        if key in self._cache:
            return list(self._cache[key])

        self._write_json({"smiles": product_smiles, "topk": int(topk)})
        resp = self._read_json(skip_non_json=True)
        if not resp.get("ok"):
            raise RuntimeError(f"NAG2G inference failed: {resp}")

        preds: List[OneStepPrediction] = []
        raw_list = list(resp.get("raw") or [])
        reactant_lists = list(resp.get("predictions") or [])
        for rank, reactants in enumerate(reactant_lists):
            rlist = [str(x) for x in (reactants or []) if str(x)]
            if not rlist:
                continue
            raw = raw_list[rank] if rank < len(raw_list) else None
            preds.append(
                OneStepPrediction(
                    reactants=rlist,
                    score=None,
                    meta={"rank": rank, "raw": raw},
                )
            )

        self._cache[key] = list(preds)
        return preds
