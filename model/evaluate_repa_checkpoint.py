import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from retro_model import RetrosynthesisModel
from train_retrosynthesis import (
    GENERATION_EXAMPLE_FIELDS,
    append_csv_row,
    evaluate_generation,
    init_csv,
    load_init_checkpoint,
    save_json,
)
from train_retrosynthesis_repa import RepaCollator, RepaReactionDataset, evaluate_repa_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-product-len", type=int, default=128)
    parser.add_argument("--max-reactants-len", type=int, default=128)
    parser.add_argument("--max-teacher-len", type=int, default=128)
    parser.add_argument("--seq-align-weight", type=float, default=0.1)
    parser.add_argument("--tok-align-weight", type=float, default=0.2)
    parser.add_argument("--eos-weight", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--generation-eval-samples", type=int, default=16)
    parser.add_argument("--generation-max-new-tokens", type=int, default=128)
    parser.add_argument("--generation-beam-width", type=int, default=1)
    parser.add_argument("--generation-length-penalty", type=float, default=0.0)
    parser.add_argument("--preview-samples", type=int, default=8)
    parser.add_argument("--amp-dtype", choices=["fp32", "fp16", "bf16"], default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model = RetrosynthesisModel().to(device)
    load_init_checkpoint(model, args.checkpoint)

    dataset = RepaReactionDataset(args.csv, args.limit)
    collator = RepaCollator(
        model.encoder_tokenizer,
        model.decoder_tokenizer,
        args.max_product_len,
        args.max_reactants_len,
        args.max_teacher_len,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    metrics = evaluate_repa_loss(
        model,
        dataloader,
        device,
        seq_align_weight=args.seq_align_weight,
        tok_align_weight=args.tok_align_weight,
        eos_weight=args.eos_weight,
        max_batches=args.max_eval_batches,
        amp_dtype=args.amp_dtype,
    )
    generation_metrics, examples = evaluate_generation(
        model,
        dataset,
        collator,
        device,
        args.generation_eval_samples,
        args.generation_max_new_tokens,
        args.generation_beam_width,
        args.generation_length_penalty,
    )
    metrics.update(generation_metrics)

    save_json(
        results_dir / "metrics.json",
        {
            "checkpoint": args.checkpoint,
            "csv": args.csv,
            "device": args.device,
            **metrics,
        },
    )
    save_json(results_dir / "run_config.json", vars(args))
    generation_examples_path = results_dir / "generation_examples.csv"
    init_csv(generation_examples_path, GENERATION_EXAMPLE_FIELDS)
    for example in examples:
        append_csv_row(generation_examples_path, GENERATION_EXAMPLE_FIELDS, example)

    print(
        f"eval_loss={metrics['eval_loss']:.6f} "
        f"eval_ce_loss={metrics['eval_ce_loss']:.6f} "
        f"eval_align_loss={metrics['eval_align_loss']:.6f} "
        f"eval_seq_align_loss={metrics['eval_seq_align_loss']:.6f} "
        f"eval_tok_align_loss={metrics['eval_tok_align_loss']:.6f}"
    )
    print(f"generation_exact={generation_metrics['generation_exact']:.6f}")
    if args.generation_beam_width > 1:
        print(f"generation_topk_exact={generation_metrics['generation_topk_exact']:.6f}")
    print(f"generation_raw_exact={generation_metrics['generation_raw_exact']:.6f}")
    print(f"generation_invalid_top1_rate={generation_metrics['generation_invalid_top1_rate']:.6f}")
    for example in examples[:args.preview_samples]:
        print(f"preview_match={example['match']}")
        print(f"preview_product={example['product']}")
        print(f"preview_target={example['target']}")
        print(f"preview_pred={example['pred']}")
        print(f"preview_pred_canonical={example['pred_canonical']}")


if __name__ == "__main__":
    main()
