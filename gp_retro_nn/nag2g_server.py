from __future__ import annotations

import argparse
import ast
import configparser
import json
import os
import sys
import traceback
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _jsonl_write(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _jsonl_read() -> Optional[Dict[str, Any]]:
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return {}
    return json.loads(line)


def _setmap2smiles_rdkit(smiles: str) -> str:
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except Exception:
        return smiles

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    mol = AllChem.RemoveHs(mol)
    for idx, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(idx + 1)
    return Chem.MolToSmiles(mol)


def _clean_smiles_rdkit(smiles: str) -> str:
    try:
        from rdkit import Chem  # type: ignore
    except Exception:
        return smiles

    if not smiles:
        return smiles
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return smiles
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        patt = Chem.MolFromSmarts("[#0]")  # dummy atoms
        if patt is not None:
            mol = Chem.DeleteSubstructs(mol, patt)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return smiles


def _split_reactants(smiles: str) -> List[str]:
    parts = [p.strip() for p in (smiles or "").split(".") if p.strip()]
    cleaned = [_clean_smiles_rdkit(p) for p in parts]
    return [c for c in cleaned if c]


@dataclass(frozen=True)
class _ModelBundle:
    args: Any
    use_cuda: bool
    task: Any
    generator: Any
    data_parallel_world_size: int
    data_parallel_rank: int


def _g2g_weight_reload(state: Dict[str, Any]) -> Dict[str, Any]:
    import torch  # type: ignore

    if state["model"]["degree_pe.weight"].shape[0] != 100:
        tmp_shape = state["model"]["degree_pe.weight"].shape
        tmp = torch.zeros((100, tmp_shape[1])).to(state["model"]["degree_pe.weight"].device)
        tmp[: tmp_shape[0]] = state["model"]["degree_pe.weight"]
        state["model"]["degree_pe.weight"] = tmp
    return state


def _coerce_ini_value(raw: str, current: Any) -> Any:
    """Best-effort type coercion for values loaded from NAG2G config.ini."""
    if raw is None:
        return current
    s = str(raw).strip()
    if s == "" or s == "None":
        return current

    # If the current type is known, respect it.
    if isinstance(current, bool):
        if s in {"True", "true", "1"}:
            return True
        if s in {"False", "false", "0"}:
            return False
        return current
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(s)
        except Exception:
            return current
    if isinstance(current, float):
        try:
            return float(s)
        except Exception:
            return current
    if isinstance(current, (list, tuple, dict)):
        try:
            return ast.literal_eval(s)
        except Exception:
            return current

    # Unknown/None type: try to infer safely.
    if s in {"True", "true"}:
        return True
    if s in {"False", "false"}:
        return False
    try:
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
    except Exception:
        pass
    try:
        return float(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def _apply_config_ini(args: Any, config_file: Path) -> Any:
    """Apply NAG2G config.ini overrides without relying on upstream read_config()."""
    if not config_file or not config_file.exists():
        return args
    cfg = configparser.ConfigParser()
    cfg.read(str(config_file))
    if "DEFAULT" not in cfg:
        return args
    # Mirror NAG2G.utils.save_config.read_config(): it intentionally does NOT
    # override a set of runtime/IO options from config.ini (e.g., data path).
    skip_keys = {
        "batch_size",
        "batch_size_valid",
        "data",
        "tensorboard_logdir",
        "bf16",
        "num_workers",
        "required_batch_size_multiple",
        "valid_subset",
        "label_prob",
        "mid_prob",
        "mid_upper",
        "mid_lower",
        "plddt_loss_weight",
        "pos_loss_weight",
        "shufflegraph",
        "infer_save_name",
        "decoder_attn_from_loader",
        "infer_step",
        "config_file",
        "path",
        "results_path",
        "beam_size",
        "search_strategies",
        "len_penalty",
        "temperature",
        "beam_size_second",
        "beam_head_second",
        "nprocs_per_node",
        "data_buffer_size",
        "distributed_rank",
        "distributed_port",
        "distributed_world_size",
        "distributed_backend",
        "distributed_init_method",
        "distributed_no_spawn",
        "lr_shrink",
    }
    for key, raw in cfg["DEFAULT"].items():
        if key in skip_keys:
            continue
        if not hasattr(args, key):
            continue
        cur = getattr(args, key)
        try:
            setattr(args, key, _coerce_ini_value(raw, cur))
        except Exception:
            # Best-effort: ignore broken overrides instead of crashing the server.
            pass
    return args


def _init_model(
    *,
    project_dir: Path,
    data_dir: Path,
    checkpoint_path: Path,
    config_file: Optional[Path] = None,
    task_name: str,
    arch: str,
    loss_name: str,
    encoder_type: str,
    dict_name: str,
    bpe_tokenizer_path: str,
    search_strategies: str,
    beam_size: int,
    len_penalty: float,
    temperature: float,
    batch_size: int,
    seed: int,
    use_cpu: bool,
    fp16: bool,
) -> _ModelBundle:
    sys.path.insert(0, str(project_dir))
    # Make NAG2G-main/unimol_plus importable even if not installed into site-packages.
    unimol_plus_dir = project_dir / "unimol_plus"
    if unimol_plus_dir.exists():
        sys.path.insert(0, str(unimol_plus_dir))
    import NAG2G  # noqa: F401  # registers tasks/models via side effects

    import torch  # type: ignore
    from unicore import checkpoint_utils, distributed_utils, tasks, utils  # type: ignore
    from unicore import options as unicore_options  # type: ignore

    from NAG2G.search_strategies.parse import add_search_strategies_args
    from NAG2G.search_strategies.beam_search_generator import (  # type: ignore
        SequenceGeneratorBeamSearch,
    )
    from NAG2G.search_strategies.simple_sequence_generator import SimpleGenerator  # type: ignore

    # Build a Unicore args Namespace with defaults via its standard CLI parser.
    parser = unicore_options.get_validation_parser()
    add_search_strategies_args(parser)
    unicore_options.add_model_args(parser)

    # Mirror NAG2G/validate.py behavior: if a config.ini is available, load it to
    # override model hyperparameters so the architecture matches the checkpoint.
    if config_file is None:
        cand1 = checkpoint_path.parent / "config.ini"
        cand2 = checkpoint_path.with_suffix(".ini")
        if cand1.exists():
            config_file = cand1
        elif cand2.exists():
            config_file = cand2

    argv = [
        "nag2g_server",
        str(data_dir),
        "--valid-subset",
        "test",
        "--task",
        str(task_name),
        "--loss",
        str(loss_name),
        "--arch",
        str(arch),
        "--encoder-type",
        str(encoder_type),
        "--dict-name",
        str(dict_name),
        # Note: unicore/NAG2G CLI expects underscore option name.
        "--bpe_tokenizer_path",
        str(bpe_tokenizer_path),
        "--path",
        str(checkpoint_path),
        "--model-overrides",
        "{}",
        "--seed",
        str(int(seed)),
        "--batch-size",
        str(int(batch_size)),
        "--num-workers",
        "0",
        "--data-buffer-size",
        str(int(batch_size)),
        "--required-batch-size-multiple",
        "1",
        "--ddp-backend",
        "no_c10d",
        "--search_strategies",
        str(search_strategies),
        "--beam-size",
        str(int(beam_size)),
        "--len-penalty",
        str(float(len_penalty)),
        "--temperature",
        str(float(temperature)),
        "--infer_step",
    ]
    if config_file is not None and config_file.exists():
        argv += ["--config_file", str(config_file)]
    if use_cpu:
        argv.append("--cpu")
    if fp16:
        argv.append("--fp16")

    old_argv = sys.argv
    try:
        sys.argv = argv
        args = unicore_options.parse_args_and_arch(parser)
        if config_file is not None and config_file.exists():
            args = _apply_config_ini(args, config_file)
    finally:
        sys.argv = old_argv

    # Unicore/NAG2G often assumes torch.distributed has been initialized (even for world_size=1).
    # Since we are not calling unicore's distributed launcher here, initialize a local single-process
    # process group when needed so that distributed_utils.get_*_rank() works.
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() and torch.distributed.is_nccl_available() else "gloo"
        # Use file init method to avoid picking ports.
        init_file = tempfile.NamedTemporaryFile(prefix="nag2g_dist_", suffix=".init", delete=False)
        init_file.close()
        init_method = f"file://{init_file.name}"
        try:
            torch.distributed.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=0,
                world_size=1,
            )
        except Exception:
            # If init fails, keep going; caller will see the underlying error.
            pass

    use_cuda = torch.cuda.is_available() and not getattr(args, "cpu", False) and not use_cpu
    if use_cuda:
        device_id = int(getattr(args, "device_id", 0))
        torch.cuda.set_device(device_id)

    if getattr(args, "distributed_world_size", 1) > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    state = checkpoint_utils.load_checkpoint_to_cpu(str(checkpoint_path))
    task = tasks.setup_task(args)
    model = task.build_model(args)

    # Some checkpoints use the plain "G2G" task name; keep the compatibility patch.
    if str(getattr(args, "task", "")) == "G2G":
        state = _g2g_weight_reload(state)

    model.load_state_dict(state["model"], strict=False)

    if fp16:
        model = model.half()
    if use_cuda:
        model.cuda()

    model.eval()

    if str(search_strategies) == "SequenceGeneratorBeamSearch":
        generator = SequenceGeneratorBeamSearch(
            [model],
            task.dictionary,
            beam_size=int(beam_size),
            len_penalty=float(len_penalty),
            max_len_b=1024,
        )
    elif str(search_strategies) == "SimpleGenerator":
        generator = SimpleGenerator(
            model,
            task.dictionary,
            beam_size=int(beam_size),
            len_penalty=float(len_penalty),
            args=args,
        )
    else:
        raise ValueError(f"Unsupported search_strategies: {search_strategies}")

    return _ModelBundle(
        args=args,
        use_cuda=use_cuda,
        task=task,
        generator=generator,
        data_parallel_world_size=data_parallel_world_size,
        data_parallel_rank=data_parallel_rank,
    )


def _predict_products(
    bundle: _ModelBundle,
    product_smiles_list: List[str],
    *,
    seed: int,
) -> tuple[List[List[str]], List[List[float]]]:
    from unicore import utils as unicore_utils  # type: ignore

    mapped = [_setmap2smiles_rdkit(s) for s in product_smiles_list]

    def _score_to_float(x: Any) -> float:
        try:
            import torch  # type: ignore

            if isinstance(x, torch.Tensor):
                if x.numel() == 1:
                    return float(x.detach().cpu().item())
                return float(x.detach().cpu().float().sum().item())
        except Exception:
            pass
        try:
            return float(x)
        except Exception:
            return 0.0

    # Fast path for unimolv2: build the exact batched_data dict expected by unimol.unimolv2,
    # without relying on NAG2G's dataset loaders (which are geared toward file-based datasets).
    if str(getattr(bundle.args, "encoder_type", "")) == "unimolv2":
        import numpy as np

        try:
            from rdkit import Chem  # type: ignore
            from rdkit.Chem import AllChem  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"RDKit is required for NAG2G unimolv2 inference: {e}")

        from NAG2G.utils.graph_process import process_one  # type: ignore
        from unimol.data.molecule_dataset import (  # type: ignore
            get_graph_features,
            pad_1d,
            pad_1d_feat,
            pad_2d,
            pad_2d_feat,
            pad_attn_bias,
        )

        def _coords(smiles: str) -> np.ndarray:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros((0, 3), dtype=np.float32)
            try:
                # Embed without explicit H to keep atom count aligned with process_one().
                res = AllChem.EmbedMolecule(mol, randomSeed=int(seed), maxAttempts=2000)
                if res == 0:
                    try:
                        AllChem.MMFFOptimizeMolecule(mol)
                    except Exception:
                        pass
                else:
                    AllChem.Compute2DCoords(mol)
                pos = mol.GetConformer().GetPositions().astype(np.float32)
            except Exception:
                AllChem.Compute2DCoords(mol)
                pos = mol.GetConformer().GetPositions().astype(np.float32)
            if pos.size:
                pos = pos - pos.mean(axis=0, keepdims=True)
            return pos

        feats = []
        for smi in mapped:
            item = process_one(smi)
            feat = get_graph_features(item, N_vnode=int(getattr(bundle.args, "N_vnode", 1)))
            pos = _coords(smi)
            if feat["atom_feat"].shape[0] != pos.shape[0]:
                # Fallback: keep running with zeros if coordinate generation disagrees.
                pos = np.zeros((feat["atom_feat"].shape[0], 3), dtype=np.float32)
            import torch  # type: ignore

            feat["pos"] = torch.from_numpy(pos).float()
            feats.append(feat)

        max_node_num = max(int(f["atom_mask"].shape[0]) for f in feats) if feats else 0
        max_node_num = (max_node_num + 1 + 3) // 4 * 4 - 1
        n_vnode = int(getattr(bundle.args, "N_vnode", 1))

        batched_data = {
            "atom_feat": pad_1d_feat([f["atom_feat"] for f in feats], max_node_num),
            "atom_mask": pad_1d([f["atom_mask"] for f in feats], max_node_num),
            "edge_feat": pad_2d_feat([f["edge_feat"] for f in feats], max_node_num),
            "shortest_path": pad_2d([f["shortest_path"] for f in feats], max_node_num),
            "degree": pad_1d([f["degree"] for f in feats], max_node_num),
            "pos": pad_1d_feat([f["pos"] for f in feats], max_node_num),
            "pair_type": pad_2d_feat([f["pair_type"] for f in feats], max_node_num),
            "attn_bias": pad_attn_bias([f["attn_bias"] for f in feats], max_node_num, n_vnode),
        }

        sample = {
            "net_input": {"batched_data": batched_data},
            "target": {"product_smiles": list(mapped)},
        }
        sample = unicore_utils.move_to_cuda(sample) if bundle.use_cuda else sample
        if str(getattr(bundle.args, "search_strategies", "")) == "SequenceGeneratorBeamSearch":
            from NAG2G.utils.G2G_cal import get_smiles, gen_map  # type: ignore
            from NAG2G.utils.chemutils import add_chirality  # type: ignore

            pred = bundle.generator(sample)  # List[List[Dict[str, Tensor]]]
            out_smiles: List[List[str]] = []
            out_scores: List[List[float]] = []
            beam_size = int(getattr(bundle.generator, "beam_size", len(pred[0]) if pred else 0))
            for i, gt_product in enumerate(mapped):
                s_i: List[str] = []
                sc_i: List[float] = []
                for j in range(min(beam_size, len(pred[i]))):
                    toks = pred[i][j]["tokens"].detach().cpu().numpy()
                    toks = np.insert(toks, 0, 1)  # add BOS
                    toks = toks[:-1]  # strip EOS
                    tok_str = bundle.task.get_str(toks)
                    smi = get_smiles(tok_str, atom_map=gen_map(gt_product))
                    try:
                        smi = add_chirality(gt_product, smi)
                    except Exception:
                        pass
                    s_i.append(str(smi))
                    sc_i.append(_score_to_float(pred[i][j].get("score")))
                out_smiles.append(s_i)
                out_scores.append(sc_i)
            return out_smiles, out_scores

        preds = bundle.task.infer_step(sample, bundle.generator)
        return preds, [[] for _ in preds]

    # Fallback: use NAG2G's empty-dataset loader for other encoder types.
    bundle.task.load_empty_dataset(init_values=mapped, seed=int(seed))
    dataset = bundle.task.dataset("test")

    itr = bundle.task.get_batch_iterator(
        dataset=dataset,
        batch_size=len(mapped),
        ignore_invalid_inputs=True,
        seed=int(getattr(bundle.args, "seed", seed)),
        num_shards=bundle.data_parallel_world_size,
        shard_id=bundle.data_parallel_rank,
        num_workers=int(getattr(bundle.args, "num_workers", 0)),
        data_buffer_size=int(getattr(bundle.args, "data_buffer_size", len(mapped))),
    ).next_epoch_itr(shuffle=False)

    for sample in itr:
        sample = unicore_utils.move_to_cuda(sample) if bundle.use_cuda else sample
        if str(getattr(bundle.args, "search_strategies", "")) == "SequenceGeneratorBeamSearch":
            from NAG2G.utils.G2G_cal import get_smiles, gen_map  # type: ignore
            from NAG2G.utils.chemutils import add_chirality  # type: ignore
            import numpy as np

            pred = bundle.generator(sample)
            out_smiles: List[List[str]] = []
            out_scores: List[List[float]] = []
            beam_size = int(getattr(bundle.generator, "beam_size", len(pred[0]) if pred else 0))
            for i, gt_product in enumerate(mapped):
                s_i: List[str] = []
                sc_i: List[float] = []
                for j in range(min(beam_size, len(pred[i]))):
                    toks = pred[i][j]["tokens"].detach().cpu().numpy()
                    toks = np.insert(toks, 0, 1)
                    toks = toks[:-1]
                    tok_str = bundle.task.get_str(toks)
                    smi = get_smiles(tok_str, atom_map=gen_map(gt_product))
                    try:
                        smi = add_chirality(gt_product, smi)
                    except Exception:
                        pass
                    s_i.append(str(smi))
                    sc_i.append(_score_to_float(pred[i][j].get("score")))
                out_smiles.append(s_i)
                out_scores.append(sc_i)
            return out_smiles, out_scores

        result = bundle.task.infer_step(sample, bundle.generator)
        return result, [[] for _ in result]
    return [[] for _ in mapped], [[] for _ in mapped]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Persistent NAG2G inference server (JSONL stdin/stdout).")
    parser.add_argument("--project-dir", type=str, required=True, help="Path to NAG2G-main")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data dir containing dict.txt")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint_*.pt")
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Optional path to NAG2G config.ini (defaults to checkpoint parent/config.ini if present).",
    )
    parser.add_argument("--task", type=str, default="G2G_unimolv2")
    parser.add_argument("--arch", type=str, default="NAG2G_G2G")
    parser.add_argument("--loss", type=str, default="G2G")
    parser.add_argument("--encoder-type", type=str, default="unimolv2")
    parser.add_argument("--dict-name", type=str, default="dict.txt")
    parser.add_argument("--bpe-tokenizer-path", type=str, default="none")
    parser.add_argument("--search-strategies", type=str, default="SimpleGenerator")
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--len-penalty", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args(argv)

    # Keep stdout clean for protocol; divert any library prints to stderr during import/init.
    real_stdout = sys.stdout
    try:
        sys.stdout = sys.stderr
        bundle = _init_model(
            project_dir=Path(args.project_dir),
            data_dir=Path(args.data_dir),
            checkpoint_path=Path(args.checkpoint),
            config_file=Path(args.config_file) if args.config_file else None,
            task_name=args.task,
            arch=args.arch,
            loss_name=args.loss,
            encoder_type=args.encoder_type,
            dict_name=args.dict_name,
            bpe_tokenizer_path=args.bpe_tokenizer_path,
            search_strategies=args.search_strategies,
            beam_size=int(args.beam_size),
            len_penalty=float(args.len_penalty),
            temperature=float(args.temperature),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
            use_cpu=bool(args.cpu),
            fp16=bool(args.fp16),
        )
    except Exception as e:
        sys.stdout = real_stdout
        _jsonl_write(
            {
                "ok": False,
                "error": f"init_failed:{type(e).__name__}:{e}",
                "traceback": traceback.format_exc(limit=50),
            }
        )
        return 2
    finally:
        sys.stdout = real_stdout

    _jsonl_write({"ok": True, "status": "ready", "model": "nag2g"})

    while True:
        try:
            req = _jsonl_read()
        except Exception as e:
            _jsonl_write({"ok": False, "error": f"bad_json:{e}"})
            continue
        if req is None:
            return 0
        if not req:
            continue
        if req.get("cmd") == "ping":
            _jsonl_write({"ok": True, "status": "pong"})
            continue
        if req.get("cmd") == "exit":
            _jsonl_write({"ok": True, "status": "bye"})
            return 0

        product = str(req.get("smiles") or "")
        topk = int(req.get("topk") or args.beam_size)
        seed = int(req.get("seed") or args.seed)
        if not product:
            _jsonl_write({"ok": False, "error": "empty_smiles"})
            continue
        if topk <= 0:
            _jsonl_write({"ok": False, "error": "topk_must_be_positive"})
            continue
        if topk > int(args.beam_size):
            _jsonl_write({"ok": False, "error": f"topk_exceeds_beam_size:{topk}>{int(args.beam_size)}"})
            continue

        try:
            preds, scores = _predict_products(bundle, [product], seed=seed)
            raw_preds = list((preds[0] if preds else []))[:topk]
            raw_scores = list((scores[0] if scores else []))[:topk]
            reactant_lists = [_split_reactants(s) for s in raw_preds]
            _jsonl_write(
                {
                    "ok": True,
                    "smiles": product,
                    "topk": topk,
                    "predictions": reactant_lists,
                    "raw": raw_preds,
                    "scores": raw_scores,
                }
            )
        except Exception as e:
            _jsonl_write(
                {
                    "ok": False,
                    "error": f"infer_failed:{type(e).__name__}:{e}",
                    "traceback": traceback.format_exc(limit=50),
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
