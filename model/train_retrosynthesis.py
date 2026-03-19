import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from retro_model import RetrosynthesisModel


class ReactionDataset(Dataset):
    def __init__(self, csv_path, limit=None):
        self.rows = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                reactants, product = row["reactants>reagents>production"].split(">>", 1)
                self.rows.append((product.strip(), reactants.strip()))
                if limit is not None and len(self.rows) >= limit:
                    break

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        product, reactants = self.rows[idx]
        return {"product": product, "reactants": reactants}


class RetrosynthesisCollator:
    def __init__(self, encoder_tokenizer, decoder_tokenizer, max_product_len, max_reactants_len):
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_product_len = max_product_len
        self.max_reactants_len = max_reactants_len

    def __call__(self, batch):
        product_ids = []
        reactant_inputs = []
        reactant_targets = []

        for item in batch:
            product = item["product"]
            reactants = item["reactants"].replace(" ", "")

            p_ids = self.encoder_tokenizer.encode(
                product,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_product_len,
            )
            r_ids = self.decoder_tokenizer.encode(reactants, add_special_tokens=False)
            r_ids = r_ids[: self.max_reactants_len - 1]

            product_ids.append(p_ids)
            reactant_inputs.append([self.decoder_tokenizer.bos_token_id] + r_ids)
            reactant_targets.append(r_ids + [self.decoder_tokenizer.eos_token_id])

        product_input_ids, product_attention_mask = pad_batch(
            product_ids,
            self.encoder_tokenizer.pad_token_id,
        )
        reactant_input_ids, _ = pad_batch(
            reactant_inputs,
            self.decoder_tokenizer.pad_token_id,
        )
        targets, _ = pad_batch(
            reactant_targets,
            self.decoder_tokenizer.pad_token_id,
        )

        return {
            "product_input_ids": product_input_ids,
            "product_attention_mask": product_attention_mask,
            "reactant_input_ids": reactant_input_ids,
            "targets": targets,
        }


def pad_batch(sequences, pad_id):
    max_len = max(len(seq) for seq in sequences)
    batch = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)
    for i, seq in enumerate(sequences):
        batch[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[i, :len(seq)] = True
    return batch, mask


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module):
    for p in module.parameters():
        p.requires_grad = True


def configure_training_stage(model, stage, trainable_decoder_blocks):
    freeze_module(model.encoder)
    freeze_module(model.decoder)
    unfreeze_module(model.aligner)
    unfreeze_module(model.decoder.ln_f)
    unfreeze_module(model.decoder.head)

    for block in model.decoder.blocks:
        if block.cross_attn is not None:
            unfreeze_module(block.cross_attn)
            unfreeze_module(block.ln_cross)

    if stage == 2:
        for block in list(model.decoder.blocks)[-trainable_decoder_blocks:]:
            unfreeze_module(block.attn)
            unfreeze_module(block.ln1)
            unfreeze_module(block.mlp)
            unfreeze_module(block.ln2)


def build_optimizer(model, lr, decoder_lr=None, weight_decay=0.0):
    params = []
    if decoder_lr is None:
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    fast_params = []
    slow_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("decoder.") and "cross_attn" not in name and "ln_cross" not in name:
            slow_params.append(p)
        else:
            fast_params.append(p)

    param_groups = [{"params": fast_params, "lr": lr, "weight_decay": weight_decay}]
    if slow_params:
        param_groups.append({"params": slow_params, "lr": decoder_lr, "weight_decay": weight_decay})
    return torch.optim.AdamW(param_groups)


def move_batch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def decode_tokens(tokenizer, token_ids):
    text = tokenizer.decode(token_ids)
    text = text.replace(" ", "")
    text = text.replace(tokenizer.bos_token or "", "")
    text = text.replace(tokenizer.eos_token or "", "")
    text = text.replace(tokenizer.sep_token or "", "")
    text = text.replace(tokenizer.pad_token or "", "")
    return text


def evaluate_loss(model, dataloader, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = move_batch(batch, device)
            _, loss, _ = model(**batch)
            total_loss += loss.item()
            total_batches += 1
    model.train()
    return total_loss / max(total_batches, 1)


def evaluate_generation(model, dataset, collator, device, sample_count, max_new_tokens, beam_width):
    total = min(sample_count, len(dataset))
    matches = 0
    examples = []

    model.eval()
    with torch.no_grad():
        for idx in range(total):
            item = dataset[idx]
            batch = collator([item])
            batch = move_batch(batch, device)
            pred_ids = model.generate(
                batch["product_input_ids"],
                batch["product_attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_k=None,
                beam_width=beam_width,
            )
            pred = decode_tokens(model.decoder_tokenizer, pred_ids[0].tolist())
            target = item["reactants"].replace(" ", "")
            product = item["product"].replace(" ", "")
            match = pred == target
            matches += int(match)
            examples.append((product, target, pred, match))
    model.train()
    return matches / max(total, 1), examples


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="model/data/train.csv")
    parser.add_argument("--eval-csv", default="model/data/eval.csv")
    parser.add_argument("--save-dir", default="model/checkpoints")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decoder-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-product-len", type=int, default=128)
    parser.add_argument("--max-reactants-len", type=int, default=128)
    parser.add_argument("--trainable-decoder-blocks", type=int, default=4)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-eval", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--generation-eval-samples", type=int, default=0)
    parser.add_argument("--generation-max-new-tokens", type=int, default=128)
    parser.add_argument("--generation-beam-width", type=int, default=1)
    parser.add_argument("--preview-samples", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    model = RetrosynthesisModel().to(device)
    configure_training_stage(model, args.stage, args.trainable_decoder_blocks)

    train_dataset = ReactionDataset(args.train_csv, args.limit_train)
    eval_dataset = ReactionDataset(args.eval_csv, args.limit_eval)
    collator = RetrosynthesisCollator(
        model.encoder_tokenizer,
        model.decoder_tokenizer,
        args.max_product_len,
        args.max_reactants_len,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    optimizer = build_optimizer(model, args.lr, args.decoder_lr, args.weight_decay)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"trainable_params={count_trainable_params(model)}")

    best_eval = None
    global_step = 0
    model.train()

    for epoch in range(args.epochs):
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            _, loss, _ = model(**batch)
            loss.backward()
            optimizer.step()

            global_step += 1
            print(f"epoch={epoch} step={global_step} train_loss={loss.item():.6f}")

            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                break

        eval_loss = evaluate_loss(model, eval_loader, device, args.max_eval_batches)
        print(f"epoch={epoch} eval_loss={eval_loss:.6f}")

        generation_exact = None
        preview_examples = []
        if args.generation_eval_samples > 0:
            generation_exact, preview_examples = evaluate_generation(
                model,
                eval_dataset,
                collator,
                device,
                args.generation_eval_samples,
                args.generation_max_new_tokens,
                args.generation_beam_width,
            )
            print(f"epoch={epoch} generation_exact={generation_exact:.6f}")
            for product, target, pred, match in preview_examples[:args.preview_samples]:
                print(f"preview_match={int(match)}")
                print(f"preview_product={product}")
                print(f"preview_target={target}")
                print(f"preview_pred={pred}")

        if best_eval is None or eval_loss < best_eval:
            best_eval = eval_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "eval_loss": eval_loss,
                    "generation_exact": generation_exact,
                    "stage": args.stage,
                },
                save_dir / "best.pt",
            )

        if args.max_train_steps is not None and global_step >= args.max_train_steps:
            break


if __name__ == "__main__":
    main()
