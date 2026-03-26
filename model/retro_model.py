from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "encoder"))
sys.path.insert(0, str(ROOT / "decoder"))

from encoders import LocalBertEncoder
from model import GPTConfig, GPT
from tokenizer import SmilesTokenizer


DEFAULT_ENCODER_PATH = ROOT / "encoder" / "MolEncoder-SMILES-Drug-1.2B"
DEFAULT_DECODER_VOCAB_PATH = ROOT / "decoder" / "vocabs" / "vocab.txt"
DEFAULT_DECODER_WEIGHT_PATH = ROOT / "decoder" / "weights" / "SMILES-650M-3B-Epoch1.pt"
DEFAULT_DECODER_CONFIG = dict(n_layer=13, n_head=32, n_embd=2048)


def build_decoder_tokenizer(vocab_path=DEFAULT_DECODER_VOCAB_PATH):
    tokenizer = SmilesTokenizer(str(vocab_path))
    tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    tokenizer.bos_token = "[BOS]"
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    tokenizer.eos_token = "[EOS]"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")
    tokenizer.sep_token = "[SEP]"
    tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids("[SEP]")
    return tokenizer


class Aligner(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.ln(x))))


class SequenceProjector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.ln(x))))


class RetrosynthesisModel(nn.Module):
    def __init__(self,
                 encoder_path=DEFAULT_ENCODER_PATH,
                 decoder_vocab_path=DEFAULT_DECODER_VOCAB_PATH,
                 decoder_weight_path=DEFAULT_DECODER_WEIGHT_PATH):
        super().__init__()
        self.encoder = LocalBertEncoder(str(encoder_path))
        self.encoder_tokenizer = self.encoder.tokenizer
        self.decoder_tokenizer = build_decoder_tokenizer(decoder_vocab_path)
        self.aligner = Aligner(self.encoder.dim)
        self.sequence_projector = SequenceProjector(self.encoder.dim)
        self.token_projector = SequenceProjector(self.encoder.dim)
        self.alignment_layers = 4

        decoder_config = GPTConfig(
            vocab_size=self.decoder_tokenizer.vocab_size,
            cross_attn=True,
            **DEFAULT_DECODER_CONFIG,
        )
        self.decoder = GPT(decoder_config)

        state_dict = torch.load(str(decoder_weight_path), map_location="cpu", weights_only=False)
        load_result = self.decoder.load_state_dict(state_dict, strict=False)
        if load_result.unexpected_keys:
            raise ValueError(f"unexpected decoder keys: {load_result.unexpected_keys}")

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def encode_product(self, product_input_ids, product_attention_mask):
        with torch.no_grad():
            hidden = self.encoder(product_input_ids, product_attention_mask)
        return self.aligner(hidden)

    def encode_teacher_reactants(self, teacher_input_ids, teacher_attention_mask):
        with torch.no_grad():
            return self.encoder(teacher_input_ids, teacher_attention_mask)

    def build_token_loss_weights(self, targets, eos_weight):
        weights = torch.ones_like(targets, dtype=torch.float)
        weights[targets == self.decoder_tokenizer.eos_token_id] = eos_weight
        weights[targets == self.decoder_tokenizer.pad_token_id] = 0.0
        return weights

    def compute_weighted_ce_from_logits(self, logits, targets, loss_weights, reduction="mean"):
        flat_targets = targets.reshape(-1)
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            flat_targets,
            ignore_index=self.decoder_tokenizer.pad_token_id,
            reduction="none",
        ).view_as(targets)

        valid = targets.ne(self.decoder_tokenizer.pad_token_id)
        weights = loss_weights.to(token_loss.dtype)

        if reduction == "mean":
            norm = weights[valid].sum().clamp_min(1.0)
            return (token_loss[valid] * weights[valid]).sum() / norm

        if reduction == "per_sample":
            weighted_loss = token_loss * weights
            numer = weighted_loss.sum(dim=1)
            denom = weights.masked_fill(~valid, 0.0).sum(dim=1).clamp_min(1.0)
            return numer / denom

        raise ValueError(f"unsupported CE reduction: {reduction}")

    def pool_encoder_states(self, hidden_states, input_ids, attention_mask):
        mask = attention_mask.bool()
        cls_id = self.encoder_tokenizer.cls_token_id
        sep_id = self.encoder_tokenizer.sep_token_id
        pad_id = self.encoder_tokenizer.pad_token_id
        mask = mask & input_ids.ne(cls_id) & input_ids.ne(sep_id) & input_ids.ne(pad_id)
        return masked_mean_pool(hidden_states, mask)

    def pool_decoder_states(self, hidden_states, targets):
        mask = targets.ne(self.decoder_tokenizer.pad_token_id) & targets.ne(self.decoder_tokenizer.eos_token_id)
        return masked_mean_pool(hidden_states, mask)

    def build_teacher_token_mask(self, teacher_input_ids, teacher_attention_mask):
        cls_id = self.encoder_tokenizer.cls_token_id
        sep_id = self.encoder_tokenizer.sep_token_id
        pad_id = self.encoder_tokenizer.pad_token_id
        return (
            teacher_attention_mask.bool()
            & teacher_input_ids.ne(cls_id)
            & teacher_input_ids.ne(sep_id)
            & teacher_input_ids.ne(pad_id)
        )

    def build_decoder_token_mask(self, targets):
        return targets.ne(self.decoder_tokenizer.pad_token_id) & targets.ne(self.decoder_tokenizer.eos_token_id)

    def pool_teacher_token_groups(self, teacher_hidden, teacher_token_mask, teacher_group_lengths):
        grouped_states = []
        for sample_hidden, sample_mask, sample_lengths in zip(teacher_hidden, teacher_token_mask, teacher_group_lengths):
            valid_hidden = sample_hidden[sample_mask]
            valid_lengths = sample_lengths[sample_lengths > 0]
            if int(valid_lengths.sum().item()) != valid_hidden.size(0):
                raise ValueError(
                    f"teacher subtoken counts do not match grouped lengths: "
                    f"{valid_hidden.size(0)} vs {int(valid_lengths.sum().item())}"
                )
            offset = 0
            pooled = []
            for length in valid_lengths.tolist():
                pooled.append(valid_hidden[offset:offset + length].mean(dim=0))
                offset += length
            grouped_states.append(torch.stack(pooled, dim=0))
        return grouped_states

    def forward(self, product_input_ids, product_attention_mask, reactant_input_ids, targets=None):
        memory = self.encode_product(product_input_ids, product_attention_mask)
        return self.decoder(
            reactant_input_ids,
            self.decoder_tokenizer,
            targets=targets,
            encoder_hidden_states=memory,
            encoder_attention_mask=product_attention_mask,
        )

    def forward_repa(
        self,
        product_input_ids,
        product_attention_mask,
        reactant_input_ids,
        targets,
        teacher_input_ids,
        teacher_attention_mask,
        teacher_group_lengths,
        seq_align_weight=0.1,
        tok_align_weight=0.2,
        eos_weight=3.0,
        wrong_product_weight=0.0,
        wrong_product_margin=0.2,
    ):
        memory = self.encode_product(product_input_ids, product_attention_mask)
        loss_weights = self.build_token_loss_weights(targets, eos_weight)
        logits, ce_loss, attn_maps, hidden_states = self.decoder(
            reactant_input_ids,
            self.decoder_tokenizer,
            targets=targets,
            encoder_hidden_states=memory,
            encoder_attention_mask=product_attention_mask,
            loss_weights=loss_weights,
            return_hidden_states=True,
        )

        teacher_hidden = self.encode_teacher_reactants(teacher_input_ids, teacher_attention_mask)
        teacher_pooled = self.pool_encoder_states(teacher_hidden, teacher_input_ids, teacher_attention_mask)

        decoder_layers = hidden_states[-self.alignment_layers:]
        decoder_hidden = torch.stack(decoder_layers, dim=0).mean(dim=0)
        decoder_pooled = self.pool_decoder_states(decoder_hidden, targets)
        projected_decoder = self.sequence_projector(decoder_pooled)
        seq_align_loss = 1.0 - F.cosine_similarity(projected_decoder, teacher_pooled.detach(), dim=-1).mean()

        teacher_token_mask = self.build_teacher_token_mask(teacher_input_ids, teacher_attention_mask)
        decoder_token_mask = self.build_decoder_token_mask(targets)
        teacher_token_groups = self.pool_teacher_token_groups(teacher_hidden, teacher_token_mask, teacher_group_lengths)
        decoder_token_groups = [sample_hidden[sample_mask] for sample_hidden, sample_mask in zip(decoder_hidden, decoder_token_mask)]
        teacher_tokens = torch.cat(teacher_token_groups, dim=0)
        decoder_tokens = torch.cat(decoder_token_groups, dim=0)
        if teacher_tokens.size(0) != decoder_tokens.size(0):
            raise ValueError(
                f"teacher/decoder grouped token counts do not match: "
                f"{teacher_tokens.size(0)} vs {decoder_tokens.size(0)}"
            )
        projected_tokens = self.token_projector(decoder_tokens)
        tok_align_loss = 1.0 - F.cosine_similarity(projected_tokens, teacher_tokens.detach(), dim=-1).mean()

        align_loss = seq_align_weight * seq_align_loss + tok_align_weight * tok_align_loss
        contrastive_loss = logits.new_zeros(())
        wrong_ce_loss = logits.new_zeros(())
        wrong_ce_gap = logits.new_zeros(())

        if wrong_product_weight > 0.0 and product_input_ids.size(0) > 1:
            # Use an in-batch rolled product as a cheap negative condition.
            wrong_memory = memory.roll(shifts=1, dims=0)
            wrong_attention_mask = product_attention_mask.roll(shifts=1, dims=0)
            wrong_logits, _, _ = self.decoder(
                reactant_input_ids,
                self.decoder_tokenizer,
                encoder_hidden_states=wrong_memory,
                encoder_attention_mask=wrong_attention_mask,
            )
            correct_ce_per_sample = self.compute_weighted_ce_from_logits(
                logits,
                targets,
                loss_weights,
                reduction="per_sample",
            )
            wrong_ce_per_sample = self.compute_weighted_ce_from_logits(
                wrong_logits,
                targets,
                loss_weights,
                reduction="per_sample",
            )
            wrong_ce_loss = wrong_ce_per_sample.mean()
            wrong_ce_gap = (wrong_ce_per_sample - correct_ce_per_sample).mean()
            contrastive_loss = (wrong_product_margin + correct_ce_per_sample - wrong_ce_per_sample).clamp_min(0.0).mean()

        total_loss = ce_loss + align_loss + wrong_product_weight * contrastive_loss

        return {
            "logits": logits,
            "loss": total_loss,
            "ce_loss": ce_loss,
            "align_loss": align_loss,
            "seq_align_loss": seq_align_loss,
            "tok_align_loss": tok_align_loss,
            "contrastive_loss": contrastive_loss,
            "wrong_ce_loss": wrong_ce_loss,
            "wrong_ce_gap": wrong_ce_gap,
            "attn_maps": attn_maps,
        }

    @torch.inference_mode()
    def generate(self, product_input_ids, product_attention_mask, decoder_input_ids=None,
                 max_new_tokens=128, temperature=0.0, top_k=None, beam_width=1,
                 return_all_beams=False, length_penalty=0.0, length_norm_alpha=1.0):
        memory = self.encode_product(product_input_ids, product_attention_mask)
        if decoder_input_ids is None:
            decoder_input_ids = torch.full(
                (product_input_ids.size(0), 1),
                self.decoder_tokenizer.bos_token_id,
                dtype=torch.long,
                device=product_input_ids.device,
            )

        if beam_width > 1:
            return next(
                self.decoder.beam_search_generate(
                    decoder_input_ids,
                    self.decoder_tokenizer,
                    max_new_tokens=max_new_tokens,
                    beam_width=beam_width,
                    temperature=temperature,
                    top_k=top_k,
                    stream=False,
                    encoder_hidden_states=memory,
                    encoder_attention_mask=product_attention_mask,
                    return_all=return_all_beams,
                    length_penalty=length_penalty,
                    length_norm_alpha=length_norm_alpha,
                )
            )

        output = next(
            self.decoder.generate(
                decoder_input_ids,
                self.decoder_tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                stream=False,
                encoder_hidden_states=memory,
                encoder_attention_mask=product_attention_mask,
            )
        )
        if return_all_beams:
            return [output]
        return output


def masked_mean_pool(hidden_states, mask):
    weights = mask.to(hidden_states.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (hidden_states * weights).sum(dim=1) / denom
