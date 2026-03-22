import argparse
import csv
import math
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from retro_model import RetrosynthesisModel
from train_retrosynthesis import (
    append_csv_row,
    build_train_loader,
    init_csv,
    load_init_checkpoint,
    move_batch,
    pad_batch,
    save_json,
    save_loss_curve,
    save_model_snapshot,
    save_torch,
)


class RepaReactionDataset(Dataset):
    def __init__(self, csv_path, limit=None):
        self.rows = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"product_tokens", "precursor_tokens", "product_canonical", "precursor_canonical_sorted"}
            if not required.issubset(reader.fieldnames or set()):
                missing = sorted(required - set(reader.fieldnames or []))
                raise ValueError(f"processed CSV missing columns: {missing}")

            for row in reader:
                self.rows.append(
                    {
                        "product": row["product_canonical"].strip(),
                        "reactants": row["precursor_canonical_sorted"].strip(),
                        "product_text": row["product_tokens"].strip(),
                        "teacher_text": row["precursor_tokens"].strip(),
                    }
                )
                if limit is not None and len(self.rows) >= limit:
                    break

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class RepaCollator:
    def __init__(self, encoder_tokenizer, decoder_tokenizer, max_product_len, max_reactants_len, max_teacher_len):
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_product_len = max_product_len
        self.max_reactants_len = max_reactants_len
        self.max_teacher_len = max_teacher_len

    def __call__(self, batch):
        product_ids = []
        teacher_ids = []
        teacher_group_lengths = []
        reactant_inputs = []
        reactant_targets = []

        for item in batch:
            product_ids.append(
                self.encoder_tokenizer.encode(
                    item["product_text"],
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_product_len,
                )
            )
            max_reactant_tokens = min(self.max_reactants_len - 1, self.max_teacher_len - 2)
            reactant_tokens = item["teacher_text"].split()[:max_reactant_tokens]
            teacher_group_lengths.append([len(self.encoder_tokenizer.tokenize(token)) for token in reactant_tokens])
            teacher_ids.append(
                self.encoder_tokenizer.encode(
                    " ".join(reactant_tokens),
                    add_special_tokens=True,
                    truncation=True,
                    max_length=self.max_teacher_len,
                )
            )

            reactant_ids = [self.decoder_tokenizer.convert_tokens_to_ids(token) for token in reactant_tokens]
            reactant_inputs.append([self.decoder_tokenizer.bos_token_id] + reactant_ids)
            reactant_targets.append(reactant_ids + [self.decoder_tokenizer.eos_token_id])

        product_input_ids, product_attention_mask = pad_batch(
            product_ids,
            self.encoder_tokenizer.pad_token_id,
        )
        teacher_input_ids, teacher_attention_mask = pad_batch(
            teacher_ids,
            self.encoder_tokenizer.pad_token_id,
        )
        teacher_group_lengths, _ = pad_batch(
            teacher_group_lengths,
            0,
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
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "teacher_group_lengths": teacher_group_lengths,
            "reactant_input_ids": reactant_input_ids,
            "targets": targets,
        }


def configure_repa_training(model):
    for name, param in model.named_parameters():
        param.requires_grad = not name.startswith("encoder.")
    model.encoder.eval()


def build_repa_optimizer(model, adapter_lr, top_decoder_lr, base_decoder_lr, top_decoder_blocks, weight_decay):
    fast_params = []
    mid_params = []
    slow_params = []
    top_start = max(len(model.decoder.blocks) - top_decoder_blocks, 0)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("aligner.") or name.startswith("sequence_projector.") or ".cross_attn." in name or ".ln_cross." in name:
            fast_params.append(param)
            continue
        if name.startswith("decoder.head") or name.startswith("decoder.ln_f"):
            mid_params.append(param)
            continue
        if name.startswith("decoder.blocks."):
            block_idx = int(name.split(".")[2])
            if block_idx >= top_start:
                mid_params.append(param)
            else:
                slow_params.append(param)
            continue
        slow_params.append(param)

    param_groups = []
    if fast_params:
        param_groups.append({"params": fast_params, "lr": adapter_lr, "weight_decay": weight_decay})
    if mid_params:
        param_groups.append({"params": mid_params, "lr": top_decoder_lr, "weight_decay": weight_decay})
    if slow_params:
        param_groups.append({"params": slow_params, "lr": base_decoder_lr, "weight_decay": weight_decay})
    return torch.optim.AdamW(param_groups)


def build_scheduler(optimizer, total_steps, warmup_ratio):
    warmup_steps = max(int(total_steps * warmup_ratio), 1) if total_steps > 0 else 1

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def count_trainable_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def evaluate_repa_loss(model, dataloader, device, seq_align_weight, tok_align_weight, eos_weight, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_align = 0.0
    total_seq_align = 0.0
    total_tok_align = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = move_batch(batch, device)
            outputs = model.forward_repa(
                **batch,
                seq_align_weight=seq_align_weight,
                tok_align_weight=tok_align_weight,
                eos_weight=eos_weight,
            )
            total_loss += outputs["loss"].item()
            total_ce += outputs["ce_loss"].item()
            total_align += outputs["align_loss"].item()
            total_seq_align += outputs["seq_align_loss"].item()
            total_tok_align += outputs["tok_align_loss"].item()
            total_batches += 1
    model.train()
    denom = max(total_batches, 1)
    return {
        "eval_loss": total_loss / denom,
        "eval_ce_loss": total_ce / denom,
        "eval_align_loss": total_align / denom,
        "eval_seq_align_loss": total_seq_align / denom,
        "eval_tok_align_loss": total_tok_align / denom,
    }


def save_resume_snapshot(save_dir, global_step, model, optimizer, scheduler, meta):
    stem = f"resume_step_{global_step:08d}"
    step_resume_path = save_dir / f"{stem}.pt"
    step_meta_path = save_dir / f"{stem}.json"
    latest_resume_path = save_dir / "latest_resume.pt"
    latest_resume_meta_path = save_dir / "latest_resume.json"
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        **meta,
    }
    save_torch(step_resume_path, payload)
    save_json(step_meta_path, meta)
    save_torch(latest_resume_path, payload)
    save_json(latest_resume_meta_path, meta)


def load_resume_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--eval-csv", default=None)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--save-every-steps", type=int, default=None)
    parser.add_argument("--resume-every-steps", type=int, default=None)
    parser.add_argument("--disable-eval", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--top-decoder-lr", type=float, default=5e-5)
    parser.add_argument("--base-decoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seq-align-weight", type=float, default=0.1)
    parser.add_argument("--tok-align-weight", type=float, default=0.2)
    parser.add_argument("--eos-weight", type=float, default=3.0)
    parser.add_argument("--top-decoder-blocks", type=int, default=4)
    parser.add_argument("--max-product-len", type=int, default=128)
    parser.add_argument("--max-reactants-len", type=int, default=128)
    parser.add_argument("--max-teacher-len", type=int, default=128)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-eval", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if args.init_checkpoint is not None and args.resume_from is not None:
        raise ValueError("--init-checkpoint and --resume-from cannot be used together")
    if args.disable_eval:
        args.eval_csv = None
    elif args.eval_csv is None:
        raise ValueError("--eval-csv is required unless --disable-eval is set")

    model = RetrosynthesisModel().to(device)
    configure_repa_training(model)

    optimizer = build_repa_optimizer(
        model,
        args.adapter_lr,
        args.top_decoder_lr,
        args.base_decoder_lr,
        args.top_decoder_blocks,
        args.weight_decay,
    )

    train_dataset = RepaReactionDataset(args.train_csv, args.limit_train)
    collator = RepaCollator(
        model.encoder_tokenizer,
        model.decoder_tokenizer,
        args.max_product_len,
        args.max_reactants_len,
        args.max_teacher_len,
    )

    micro_batches_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    optimizer_steps_per_epoch = max(math.ceil(micro_batches_per_epoch / args.grad_accumulation), 1)
    total_steps = optimizer_steps_per_epoch * args.epochs
    if args.max_train_steps is not None:
        total_steps = min(total_steps, args.max_train_steps)
    scheduler = build_scheduler(optimizer, total_steps, args.warmup_ratio)

    if args.resume_from is not None:
        resume_state = load_resume_checkpoint(model, optimizer, scheduler, args.resume_from, device)
    else:
        resume_state = None
        if args.init_checkpoint is not None:
            load_init_checkpoint(model, args.init_checkpoint)

    eval_dataset = None
    eval_loader = None
    if args.eval_csv is not None:
        eval_dataset = RepaReactionDataset(args.eval_csv, args.limit_eval)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    trainable_params = count_trainable_params(model)
    print(f"trainable_params={trainable_params}")

    train_loss_path = results_dir / "train_loss.csv"
    eval_metrics_path = results_dir / "eval_metrics.csv"
    loss_curve_path = results_dir / "loss_curve.svg"
    if args.resume_from is None or not train_loss_path.exists():
        init_csv(
            train_loss_path,
            ["epoch", "global_step", "train_loss", "ce_loss", "align_loss", "seq_align_loss", "tok_align_loss"],
        )
    if args.eval_csv is not None and (args.resume_from is None or not eval_metrics_path.exists()):
        init_csv(
            eval_metrics_path,
            [
                "epoch",
                "global_step",
                "eval_loss",
                "eval_ce_loss",
                "eval_align_loss",
                "eval_seq_align_loss",
                "eval_tok_align_loss",
            ],
        )

    save_json(results_dir / "run_config.json", {**vars(args), "trainable_params": trainable_params})

    train_history = []
    eval_history = []
    global_step = 0
    start_epoch = 0
    resume_epoch_step = 0
    last_epoch = -1
    last_train_loss = None

    if resume_state is not None:
        start_epoch = resume_state.get("epoch", 0)
        resume_epoch_step = resume_state.get("epoch_step", 0)
        global_step = resume_state.get("global_step", 0)
        print(
            f"resuming_from={args.resume_from} epoch={start_epoch} "
            f"epoch_step={resume_epoch_step} global_step={global_step}"
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs):
        last_epoch = epoch
        train_loader = build_train_loader(train_dataset, args.batch_size, collate_fn=collator, seed=args.seed, epoch=epoch)
        skip_batches = resume_epoch_step if epoch == start_epoch else 0
        accum_train = 0.0
        accum_ce = 0.0
        accum_align = 0.0
        accum_seq_align = 0.0
        accum_tok_align = 0.0
        accum_count = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx < skip_batches:
                continue

            batch = move_batch(batch, device)
            outputs = model.forward_repa(
                **batch,
                seq_align_weight=args.seq_align_weight,
                tok_align_weight=args.tok_align_weight,
                eos_weight=args.eos_weight,
            )
            (outputs["loss"] / args.grad_accumulation).backward()
            accum_train += outputs["loss"].item()
            accum_ce += outputs["ce_loss"].item()
            accum_align += outputs["align_loss"].item()
            accum_seq_align += outputs["seq_align_loss"].item()
            accum_tok_align += outputs["tok_align_loss"].item()
            accum_count += 1

            should_step = accum_count == args.grad_accumulation or batch_idx == len(train_loader) - 1
            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            train_row = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": accum_train / accum_count,
                "ce_loss": accum_ce / accum_count,
                "align_loss": accum_align / accum_count,
                "seq_align_loss": accum_seq_align / accum_count,
                "tok_align_loss": accum_tok_align / accum_count,
            }
            train_history.append(train_row)
            last_train_loss = float(train_row["train_loss"])
            append_csv_row(
                train_loss_path,
                ["epoch", "global_step", "train_loss", "ce_loss", "align_loss", "seq_align_loss", "tok_align_loss"],
                train_row,
            )
            print(
                f"epoch={epoch} step={global_step} "
                f"train_loss={train_row['train_loss']:.6f} "
                f"ce_loss={train_row['ce_loss']:.6f} "
                f"align_loss={train_row['align_loss']:.6f} "
                f"seq_align_loss={train_row['seq_align_loss']:.6f} "
                f"tok_align_loss={train_row['tok_align_loss']:.6f}"
            )

            if args.save_every_steps is not None and global_step % args.save_every_steps == 0:
                save_model_snapshot(
                    save_dir,
                    global_step,
                    model,
                    {
                        "epoch": epoch,
                        "epoch_step": batch_idx + 1,
                        "global_step": global_step,
                        "train_loss": train_row["train_loss"],
                    },
                )

            if args.resume_every_steps is not None and global_step % args.resume_every_steps == 0:
                save_resume_snapshot(
                    save_dir,
                    global_step,
                    model,
                    optimizer,
                    scheduler,
                    {
                        "epoch": epoch,
                        "epoch_step": batch_idx + 1,
                        "global_step": global_step,
                        "train_loss": train_row["train_loss"],
                    },
                )

            accum_train = 0.0
            accum_ce = 0.0
            accum_align = 0.0
            accum_seq_align = 0.0
            accum_tok_align = 0.0
            accum_count = 0

            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                break

        resume_epoch_step = 0

        if eval_loader is not None:
            eval_row = {"epoch": epoch, "global_step": global_step}
            eval_row.update(
                evaluate_repa_loss(
                    model,
                    eval_loader,
                    device,
                    seq_align_weight=args.seq_align_weight,
                    tok_align_weight=args.tok_align_weight,
                    eos_weight=args.eos_weight,
                    max_batches=args.max_eval_batches,
                )
            )
            eval_history.append(eval_row)
            append_csv_row(
                eval_metrics_path,
                [
                    "epoch",
                    "global_step",
                    "eval_loss",
                    "eval_ce_loss",
                    "eval_align_loss",
                    "eval_seq_align_loss",
                    "eval_tok_align_loss",
                ],
                eval_row,
            )
            save_loss_curve(train_history, eval_history, loss_curve_path)
            print(
                f"epoch={epoch} eval_loss={eval_row['eval_loss']:.6f} "
                f"eval_ce_loss={eval_row['eval_ce_loss']:.6f} "
                f"eval_align_loss={eval_row['eval_align_loss']:.6f} "
                f"eval_seq_align_loss={eval_row['eval_seq_align_loss']:.6f} "
                f"eval_tok_align_loss={eval_row['eval_tok_align_loss']:.6f}"
            )

        if args.max_train_steps is not None and global_step >= args.max_train_steps:
            break

    final_meta = {
        "epoch": last_epoch,
        "epoch_step": 0,
        "global_step": global_step,
        "train_loss": last_train_loss,
    }
    final_resume_meta = {
        "epoch": max(last_epoch + 1, 0),
        "epoch_step": 0,
        "global_step": global_step,
        "train_loss": last_train_loss,
    }
    save_torch(save_dir / "final_model.pt", model.state_dict())
    save_json(save_dir / "final_model.json", final_meta)
    save_torch(save_dir / "latest_model.pt", model.state_dict())
    save_json(save_dir / "latest_model.json", final_meta)
    save_torch(
        save_dir / "final_resume.pt",
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), **final_resume_meta},
    )
    save_json(save_dir / "final_resume.json", final_resume_meta)
    save_torch(
        save_dir / "latest_resume.pt",
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), **final_resume_meta},
    )
    save_json(save_dir / "latest_resume.json", final_resume_meta)


if __name__ == "__main__":
    main()
