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

    def pool_encoder_states(self, hidden_states, input_ids, attention_mask):
        mask = attention_mask.bool()
        cls_id = self.encoder_tokenizer.cls_token_id
        sep_id = self.encoder_tokenizer.sep_token_id
        pad_id = self.encoder_tokenizer.pad_token_id
        mask = mask & input_ids.ne(cls_id) & input_ids.ne(sep_id) & input_ids.ne(pad_id)
        return masked_mean_pool(hidden_states, mask)

    def pool_decoder_states(self, hidden_states, targets):
        mask = targets.ne(self.decoder_tokenizer.pad_token_id)
        return masked_mean_pool(hidden_states, mask)

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
        align_weight=0.2,
        eos_weight=3.0,
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
        align_loss = 1.0 - F.cosine_similarity(projected_decoder, teacher_pooled.detach(), dim=-1).mean()
        total_loss = ce_loss + align_weight * align_loss

        return {
            "logits": logits,
            "loss": total_loss,
            "ce_loss": ce_loss,
            "align_loss": align_loss,
            "attn_maps": attn_maps,
        }

    @torch.inference_mode()
    def generate(self, product_input_ids, product_attention_mask, decoder_input_ids=None,
                 max_new_tokens=128, temperature=0.0, top_k=None, beam_width=1):
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
                )
            )

        return next(
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


def masked_mean_pool(hidden_states, mask):
    weights = mask.to(hidden_states.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (hidden_states * weights).sum(dim=1) / denom
