import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from retro_model import RetrosynthesisModel
from train_retrosynthesis import (
    ReactionDataset,
    RetrosynthesisCollator,
    append_csv_row,
    evaluate_generation,
    evaluate_loss,
    init_csv,
    load_init_checkpoint,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", default="model/data/eval.csv")
    parser.add_argument("--results-dir", default="model/results/checkpoint_test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-product-len", type=int, default=128)
    parser.add_argument("--max-reactants-len", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--generation-eval-samples", type=int, default=16)
    parser.add_argument("--generation-max-new-tokens", type=int, default=128)
    parser.add_argument("--generation-beam-width", type=int, default=1)
    parser.add_argument("--preview-samples", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model = RetrosynthesisModel().to(device)
    load_init_checkpoint(model, args.checkpoint)

    dataset = ReactionDataset(args.csv, args.limit)
    collator = RetrosynthesisCollator(
        model.encoder_tokenizer,
        model.decoder_tokenizer,
        args.max_product_len,
        args.max_reactants_len,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    eval_loss = evaluate_loss(model, dataloader, device, args.max_eval_batches)
    generation_exact, examples = evaluate_generation(
        model,
        dataset,
        collator,
        device,
        args.generation_eval_samples,
        args.generation_max_new_tokens,
        args.generation_beam_width,
    )

    save_json(
        results_dir / "metrics.json",
        {
            "checkpoint": args.checkpoint,
            "csv": args.csv,
            "eval_loss": eval_loss,
            "generation_exact": generation_exact,
            "device": args.device,
        },
    )
    save_json(results_dir / "run_config.json", vars(args))
    generation_examples_path = results_dir / "generation_examples.csv"
    init_csv(
        generation_examples_path,
        ["sample_idx", "match", "decoder_input", "product", "target", "pred"],
    )
    for example in examples:
        append_csv_row(
            generation_examples_path,
            ["sample_idx", "match", "decoder_input", "product", "target", "pred"],
            example,
        )

    print(f"eval_loss={eval_loss:.6f}")
    print(f"generation_exact={generation_exact:.6f}")
    for example in examples[:args.preview_samples]:
        print(f"preview_match={example['match']}")
        print(f"preview_product={example['product']}")
        print(f"preview_target={example['target']}")
        print(f"preview_pred={example['pred']}")


if __name__ == "__main__":
    main()
