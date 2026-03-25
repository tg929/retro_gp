import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from retro_model import RetrosynthesisModel
from train_retrosynthesis import (
    default_eval_amp_dtype,
    get_eval_autocast_context,
    load_init_checkpoint,
    move_batch,
    save_json,
)
from train_retrosynthesis_repa import RepaCollator, RepaReactionDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--results-path", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-product-len", type=int, default=128)
    parser.add_argument("--max-reactants-len", type=int, default=128)
    parser.add_argument("--max-teacher-len", type=int, default=128)
    parser.add_argument("--eos-weight", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--amp-dtype", choices=["fp32", "fp16", "bf16"], default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def masked_weighted_ce(logits, targets, loss_weights, pad_token_id):
    flat_targets = targets.reshape(-1)
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        flat_targets,
        ignore_index=pad_token_id,
        reduction="none",
    )
    valid = flat_targets != pad_token_id
    flat_weights = loss_weights.reshape(-1).to(token_loss.dtype)
    norm = flat_weights[valid].sum().clamp_min(1.0)
    return (token_loss[valid] * flat_weights[valid]).sum() / norm


def masked_mean(values, mask):
    mask = mask.to(values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


class NormHookCollector:
    def __init__(self):
        self.decoder_mask = None
        self.sums = defaultdict(float)
        self.counts = defaultdict(float)

    def set_mask(self, decoder_mask):
        self.decoder_mask = decoder_mask

    def _masked_rms(self, tensor):
        if self.decoder_mask is None:
            return float(tensor.pow(2).mean().sqrt().item())
        token_rms = tensor.pow(2).mean(dim=-1).sqrt()
        mask = self.decoder_mask.to(token_rms.dtype)
        denom = mask.sum().clamp_min(1.0)
        return float(((token_rms * mask).sum() / denom).item())

    def make_self_hook(self, layer_idx):
        name = f"layer_{layer_idx:02d}_self_rms"

        def hook(_, __, output):
            tensor = output[0]
            self.sums[name] += self._masked_rms(tensor)
            self.counts[name] += 1.0

        return hook

    def make_cross_hook(self, layer_idx):
        name = f"layer_{layer_idx:02d}_cross_rms"

        def hook(_, __, output):
            tensor = output
            self.sums[name] += self._masked_rms(tensor)
            self.counts[name] += 1.0

        return hook

    def summary(self):
        out = {}
        for key, value in self.sums.items():
            out[key] = value / max(self.counts[key], 1.0)
        return out


def analyze_checkpoint(args):
    device = torch.device(args.device)
    model = RetrosynthesisModel().to(device)
    load_init_checkpoint(model, args.checkpoint)
    model.eval()

    dataset = RepaReactionDataset(args.csv, args.limit)
    collator = RepaCollator(
        model.encoder_tokenizer,
        model.decoder_tokenizer,
        args.max_product_len,
        args.max_reactants_len,
        args.max_teacher_len,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    collector = NormHookCollector()
    hooks = []
    for layer_idx, block in enumerate(model.decoder.blocks):
        hooks.append(block.attn.register_forward_hook(collector.make_self_hook(layer_idx)))
        if block.cross_attn is not None:
            hooks.append(block.cross_attn.register_forward_hook(collector.make_cross_hook(layer_idx)))

    amp_dtype = default_eval_amp_dtype(device) if args.amp_dtype is None else args.amp_dtype
    pad_id = model.decoder_tokenizer.pad_token_id
    eos_id = model.decoder_tokenizer.eos_token_id
    c_id = model.decoder_tokenizer.convert_tokens_to_ids("C")

    totals = defaultdict(float)
    total_batches = 0

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if args.max_batches is not None and batch_idx >= args.max_batches:
                    break
                batch = move_batch(batch, device)
                decoder_mask = batch["reactant_input_ids"].ne(pad_id)
                target_mask = batch["targets"].ne(pad_id)
                loss_weights = model.build_token_loss_weights(batch["targets"], args.eos_weight)

                with get_eval_autocast_context(device, amp_dtype):
                    memory = model.encode_product(batch["product_input_ids"], batch["product_attention_mask"])
                    shuffled_memory = memory.roll(shifts=1, dims=0) if memory.size(0) > 1 else memory
                    shuffled_mask = (
                        batch["product_attention_mask"].roll(shifts=1, dims=0)
                        if batch["product_attention_mask"].size(0) > 1
                        else batch["product_attention_mask"]
                    )

                    collector.set_mask(decoder_mask)
                    logits_cond, _, _, _ = model.decoder(
                        batch["reactant_input_ids"],
                        model.decoder_tokenizer,
                        targets=batch["targets"],
                        encoder_hidden_states=memory,
                        encoder_attention_mask=batch["product_attention_mask"],
                        loss_weights=loss_weights,
                        return_hidden_states=True,
                    )
                    collector.set_mask(None)

                    logits_none, _, _ = model.decoder(
                        batch["reactant_input_ids"],
                        model.decoder_tokenizer,
                        targets=batch["targets"],
                        encoder_hidden_states=None,
                        encoder_attention_mask=None,
                        loss_weights=loss_weights,
                    )
                    logits_shuf, _, _ = model.decoder(
                        batch["reactant_input_ids"],
                        model.decoder_tokenizer,
                        targets=batch["targets"],
                        encoder_hidden_states=shuffled_memory,
                        encoder_attention_mask=shuffled_mask,
                        loss_weights=loss_weights,
                    )

                ce_cond = masked_weighted_ce(logits_cond, batch["targets"], loss_weights, pad_id)
                ce_none = masked_weighted_ce(logits_none, batch["targets"], loss_weights, pad_id)
                ce_shuf = masked_weighted_ce(logits_shuf, batch["targets"], loss_weights, pad_id)

                logp_cond = F.log_softmax(logits_cond.float(), dim=-1)
                logp_none = F.log_softmax(logits_none.float(), dim=-1)
                logp_shuf = F.log_softmax(logits_shuf.float(), dim=-1)
                p_cond = logp_cond.exp()

                kl_cond_none = (p_cond * (logp_cond - logp_none)).sum(dim=-1)
                kl_cond_shuf = (p_cond * (logp_cond - logp_shuf)).sum(dim=-1)

                pred_cond = torch.argmax(logits_cond, dim=-1)
                pred_none = torch.argmax(logits_none, dim=-1)
                pred_shuf = torch.argmax(logits_shuf, dim=-1)

                totals["ce_cond"] += float(ce_cond.item())
                totals["ce_none"] += float(ce_none.item())
                totals["ce_shuf"] += float(ce_shuf.item())
                totals["kl_cond_none"] += float(masked_mean(kl_cond_none, target_mask).item())
                totals["kl_cond_shuf"] += float(masked_mean(kl_cond_shuf, target_mask).item())
                totals["argmax_agree_none"] += float(masked_mean((pred_cond == pred_none).float(), target_mask).item())
                totals["argmax_agree_shuf"] += float(masked_mean((pred_cond == pred_shuf).float(), target_mask).item())
                totals["token_acc_cond"] += float(masked_mean((pred_cond == batch["targets"]).float(), target_mask).item())
                totals["token_acc_none"] += float(masked_mean((pred_none == batch["targets"]).float(), target_mask).item())
                totals["token_acc_shuf"] += float(masked_mean((pred_shuf == batch["targets"]).float(), target_mask).item())

                first_valid = decoder_mask[:, 0]
                totals["first_token_acc_cond"] += float(masked_mean((pred_cond[:, 0] == batch["targets"][:, 0]).float(), first_valid).item())
                totals["first_token_acc_none"] += float(masked_mean((pred_none[:, 0] == batch["targets"][:, 0]).float(), first_valid).item())
                totals["first_token_acc_shuf"] += float(masked_mean((pred_shuf[:, 0] == batch["targets"][:, 0]).float(), first_valid).item())
                totals["first_token_top1_c_cond"] += float(masked_mean((pred_cond[:, 0] == c_id).float(), first_valid).item())
                totals["first_token_top1_c_none"] += float(masked_mean((pred_none[:, 0] == c_id).float(), first_valid).item())
                totals["first_token_top1_c_shuf"] += float(masked_mean((pred_shuf[:, 0] == c_id).float(), first_valid).item())

                if batch["targets"].size(1) > 1:
                    second_valid = target_mask[:, 1]
                    totals["second_token_acc_cond"] += float(masked_mean((pred_cond[:, 1] == batch["targets"][:, 1]).float(), second_valid).item())
                    totals["second_token_acc_none"] += float(masked_mean((pred_none[:, 1] == batch["targets"][:, 1]).float(), second_valid).item())
                    totals["second_token_acc_shuf"] += float(masked_mean((pred_shuf[:, 1] == batch["targets"][:, 1]).float(), second_valid).item())
                    totals["second_token_top1_c_cond"] += float(masked_mean((pred_cond[:, 1] == c_id).float(), second_valid).item())
                    totals["second_token_top1_c_none"] += float(masked_mean((pred_none[:, 1] == c_id).float(), second_valid).item())
                    totals["second_token_top1_c_shuf"] += float(masked_mean((pred_shuf[:, 1] == c_id).float(), second_valid).item())
                    totals["second_token_top1_eos_cond"] += float(masked_mean((pred_cond[:, 1] == eos_id).float(), second_valid).item())
                    totals["second_token_top1_eos_none"] += float(masked_mean((pred_none[:, 1] == eos_id).float(), second_valid).item())
                    totals["second_token_top1_eos_shuf"] += float(masked_mean((pred_shuf[:, 1] == eos_id).float(), second_valid).item())

                total_batches += 1
    finally:
        for hook in hooks:
            hook.remove()

    denom = max(total_batches, 1)
    metrics = {key: value / denom for key, value in totals.items()}
    cross_stats = collector.summary()
    layer_stats = {}
    for layer_idx in range(len(model.decoder.blocks)):
        self_key = f"layer_{layer_idx:02d}_self_rms"
        cross_key = f"layer_{layer_idx:02d}_cross_rms"
        self_rms = cross_stats.get(self_key, 0.0)
        cross_rms = cross_stats.get(cross_key, 0.0)
        layer_stats[f"layer_{layer_idx:02d}"] = {
            "self_rms": self_rms,
            "cross_rms": cross_rms,
            "cross_to_self_ratio": cross_rms / max(self_rms, 1e-8),
        }

    summary = {
        "checkpoint": args.checkpoint,
        "csv": args.csv,
        "device": args.device,
        "amp_dtype": amp_dtype,
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
        "metrics": metrics,
        "derived": {
            "ce_gain_vs_none": metrics["ce_none"] - metrics["ce_cond"],
            "ce_gain_vs_shuf": metrics["ce_shuf"] - metrics["ce_cond"],
            "token_acc_gain_vs_none": metrics["token_acc_cond"] - metrics["token_acc_none"],
            "token_acc_gain_vs_shuf": metrics["token_acc_cond"] - metrics["token_acc_shuf"],
            "first_token_acc_gain_vs_none": metrics["first_token_acc_cond"] - metrics["first_token_acc_none"],
            "first_token_acc_gain_vs_shuf": metrics["first_token_acc_cond"] - metrics["first_token_acc_shuf"],
            "second_token_acc_gain_vs_none": metrics.get("second_token_acc_cond", 0.0) - metrics.get("second_token_acc_none", 0.0),
            "second_token_acc_gain_vs_shuf": metrics.get("second_token_acc_cond", 0.0) - metrics.get("second_token_acc_shuf", 0.0),
        },
        "layer_stats": layer_stats,
    }
    return summary


def main():
    args = parse_args()
    summary = analyze_checkpoint(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.results_path:
        save_json(Path(args.results_path), summary)


if __name__ == "__main__":
    main()
