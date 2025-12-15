from __future__ import annotations

import os
import subprocess
import tempfile
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .one_step import OneStepPrediction


def _tokenize_smiles_char_level(smiles: str) -> str:
    """Match `processed_data/*_sources` format: space-separated characters."""
    return " ".join(list(smiles.strip()))


def _load_vocab_lines(vocab_path: Path) -> List[str]:
    vocab = vocab_path.read_text(encoding="utf-8").splitlines()
    vocab = [v.strip() for v in vocab if v.strip()]
    vocab += ["UNK", "SEQUENCE_START", "SEQUENCE_END"]
    return vocab


def _decode_beam_npz_example(
    predicted_ids,
    parent_ids,
    scores,
    vocab: Sequence[str],
    topk: int,
) -> List[Tuple[List[str], float]]:
    """Decode a single example from DumpBeams output into (token_seq, score)."""
    # predicted_ids/parent_ids/scores shapes: [T, K]
    T, K = predicted_ids.shape

    def tok(idx: int) -> str:
        if idx < 0 or idx >= len(vocab):
            return "UNK"
        return vocab[int(idx)]

    end_nodes: List[Tuple[int, int]] = []
    for t in range(T):
        for k in range(K):
            if tok(predicted_ids[t, k]) != "SEQUENCE_END":
                continue
            if t > 0:
                pk = int(parent_ids[t, k])
                if tok(predicted_ids[t - 1, pk]) == "SEQUENCE_END":
                    continue
            end_nodes.append((t, k))

    decoded: List[Tuple[List[str], float]] = []
    for t, k in end_nodes:
        tokens: List[str] = []
        cur_t, cur_k = t, k
        while cur_t >= 0:
            tokens.append(tok(predicted_ids[cur_t, cur_k]))
            cur_k = int(parent_ids[cur_t, cur_k])
            cur_t -= 1
        tokens.reverse()

        out_tokens: List[str] = []
        for tt in tokens:
            if tt in {"SEQUENCE_START"}:
                continue
            if tt == "SEQUENCE_END":
                break
            if tt.startswith("<RX_") and tt.endswith(">"):
                continue
            out_tokens.append(tt)

        score = float(scores[t, k])
        decoded.append((out_tokens, score))

    decoded.sort(key=lambda x: x[1], reverse=True)
    return decoded[:topk]


@dataclass(frozen=True)
class Seq2SeqSubprocessConfig:
    project_dir: Path
    model_dir: Path
    checkpoint_path: Path
    vocab_path: Path
    python_executable: Optional[str] = None
    beam_width: int = 10
    rxn_classes: Sequence[str] = (
        "<RX_1>",
        "<RX_2>",
        "<RX_3>",
        "<RX_4>",
        "<RX_5>",
        "<RX_6>",
        "<RX_7>",
        "<RX_8>",
        "<RX_9>",
        "<RX_10>",
    )
    batch_size: int = 32


class Seq2SeqSubprocessModel:
    """One-step retro model wrapper that shells out to `reaction_prediction_seq2seq-master`.

    Notes:
    - Requires a working TF1.x environment, plus the seq2seq checkpoint.
    - To handle unknown reaction class, we run inference on multiple `<RX_k>` prefixes
      for the same product and merge the beam candidates.
    """

    name = "seq2seq"

    def __init__(self, cfg: Seq2SeqSubprocessConfig):
        self.cfg = cfg
        self._cache: Dict[Tuple[str, int], List[OneStepPrediction]] = {}

    def _run_infer(self, source_path: Path, beam_path: Path) -> None:
        tasks_yaml = f"- class: DumpBeams\n  params:\n    file: {beam_path}\n"
        input_pipe_yaml = (
            "class: ParallelTextInputPipeline\n"
            "params:\n"
            "  source_files:\n"
            f"    - {source_path}\n"
        )
        model_params_yaml = f"inference.beam_search.beam_width: {int(self.cfg.beam_width)}"

        cmd = [
            self.cfg.python_executable or sys.executable,
            "-m",
            "bin.infer",
            "--tasks",
            tasks_yaml,
            "--model_dir",
            str(self.cfg.model_dir),
            "--model_params",
            model_params_yaml,
            "--input_pipeline",
            input_pipe_yaml,
            "--checkpoint_path",
            str(self.cfg.checkpoint_path),
            "--batch_size",
            str(int(self.cfg.batch_size)),
        ]

        env = dict(os.environ)
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

        try:
            subprocess.run(
                cmd,
                cwd=str(self.cfg.project_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "Seq2seq inference failed. "
                f"cmd={cmd} cwd={self.cfg.project_dir} "
                f"stdout={e.stdout[-2000:]} stderr={e.stderr[-2000:]}"
            ) from e

    @staticmethod
    def _detok_to_smiles(tokens: Sequence[str]) -> str:
        # Model is trained on char-level tokens; detokenize by concatenation.
        return "".join(tokens).replace(" ", "").strip()

    def predict(self, product_smiles: str, topk: int) -> List[OneStepPrediction]:
        key = (product_smiles, int(topk))
        if key in self._cache:
            return list(self._cache[key])

        cfg = self.cfg
        if not cfg.project_dir.exists():
            raise FileNotFoundError(f"seq2seq project_dir not found: {cfg.project_dir}")
        if not cfg.model_dir.exists():
            raise FileNotFoundError(f"seq2seq model_dir not found: {cfg.model_dir}")
        if not cfg.checkpoint_path.exists():
            raise FileNotFoundError(f"seq2seq checkpoint not found: {cfg.checkpoint_path}")
        if not cfg.vocab_path.exists():
            raise FileNotFoundError(f"seq2seq vocab not found: {cfg.vocab_path}")

        vocab = _load_vocab_lines(cfg.vocab_path)

        tokenized = _tokenize_smiles_char_level(product_smiles)
        source_lines = [f"{rx} {tokenized}".strip() for rx in cfg.rxn_classes]

        with tempfile.TemporaryDirectory(prefix="retrogp_seq2seq_") as td:
            td_path = Path(td)
            src_path = td_path / "sources.txt"
            beam_path = td_path / "beams.npz"
            src_path.write_text("\n".join(source_lines) + "\n", encoding="utf-8")

            self._run_infer(source_path=src_path, beam_path=beam_path)

            try:
                import numpy as np  # type: ignore
            except Exception as e:  # pragma: no cover
                raise RuntimeError("numpy is required to read seq2seq beam outputs") from e

            beam_data = np.load(str(beam_path))
            all_predicted = beam_data["predicted_ids"]
            all_parents = beam_data["beam_parent_ids"]
            all_scores = beam_data["scores"]

        merged: Dict[str, OneStepPrediction] = {}
        per_class_top = max(int(topk), 1)
        for i, rx in enumerate(cfg.rxn_classes):
            decoded = _decode_beam_npz_example(
                predicted_ids=all_predicted[i],
                parent_ids=all_parents[i],
                scores=all_scores[i],
                vocab=vocab,
                topk=per_class_top,
            )
            for rank, (tok_seq, score) in enumerate(decoded):
                reactants_smi = self._detok_to_smiles(tok_seq)
                if not reactants_smi:
                    continue
                reactants = [p for p in reactants_smi.split(".") if p]
                if not reactants:
                    continue
                existing = merged.get(reactants_smi)
                pred = OneStepPrediction(
                    reactants=reactants,
                    score=score,
                    meta={"rxn_class": rx, "beam_rank": rank, "beam_width": int(cfg.beam_width)},
                )
                if existing is None or (existing.score is None or score > float(existing.score)):
                    merged[reactants_smi] = pred

        out = sorted(
            merged.values(),
            key=lambda p: (float(p.score) if p.score is not None else float("-inf")),
            reverse=True,
        )[: int(topk)]

        self._cache[key] = list(out)
        return out
