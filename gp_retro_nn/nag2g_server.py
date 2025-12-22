from __future__ import annotations

import argparse
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
    import NAG2G  # noqa: F401  # registers tasks/models via side effects

    import torch  # type: ignore
    from unicore import checkpoint_utils, distributed_utils, tasks, utils  # type: ignore
    from unicore import options as unicore_options  # type: ignore

    from NAG2G.search_strategies.parse import add_search_strategies_args
    from NAG2G.search_strategies.beam_search_generator import (  # type: ignore
        SequenceGeneratorBeamSearch,
    )
    from NAG2G.search_strategies.simple_sequence_generator import SimpleGenerator  # type: ignore
    from NAG2G.utils import save_config  # type: ignore

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
        args = save_config.read_config(args)
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
) -> List[List[str]]:
    from unicore import utils as unicore_utils  # type: ignore

    mapped = [_setmap2smiles_rdkit(s) for s in product_smiles_list]
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
        result = bundle.task.infer_step(sample, bundle.generator)
        return result
    return [[] for _ in mapped]


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
            raw_preds = _predict_products(bundle, [product], seed=seed)[0]
            # raw_preds: List[str] length == beam_size
            raw_preds = list(raw_preds)[:topk]
            reactant_lists = [_split_reactants(s) for s in raw_preds]
            _jsonl_write(
                {
                    "ok": True,
                    "smiles": product,
                    "topk": topk,
                    "predictions": reactant_lists,
                    "raw": raw_preds,
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
