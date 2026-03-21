import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from local_bert import SmilesTokenizer, load_yaml, BertConfig, BERT


def _set_smiles_special(tok):
    tok.pad_token = "[PAD]"
    tok.pad_token_id = tok.convert_tokens_to_ids("[PAD]")
    tok.unk_token = "[UNK]"
    tok.unk_token_id = tok.convert_tokens_to_ids("[UNK]")
    tok.cls_token = "[CLS]"
    tok.cls_token_id = tok.convert_tokens_to_ids("[CLS]")
    tok.sep_token = "[SEP]"
    tok.sep_token_id = tok.convert_tokens_to_ids("[SEP]")
    tok.mask_token = "[MASK]"
    tok.mask_token_id = tok.convert_tokens_to_ids("[MASK]")
    return tok


class HFEncoder(nn.Module):
    def __init__(self, path: str, trust_remote_code: bool):
        super().__init__()
        self.model = AutoModel.from_pretrained(path, trust_remote_code=trust_remote_code)
        self.dim = int(self.model.config.hidden_size)

        for attr in ["pooler", "lm_head", "cls", "classifier", "score"]:
            m = getattr(self.model, attr, None)
            if isinstance(m, nn.Module):
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state


class LocalBertEncoder(nn.Module):
    def __init__(self, base: str):
        super().__init__()
        tok = SmilesTokenizer(os.path.join(base, "vocab.txt"), do_lower_case=False)
        self.tokenizer = _set_smiles_special(tok)

        enc_cfg = load_yaml(os.path.join(base, "encoder.yaml"))
        ckpt = torch.load(os.path.join(base, "checkpoint.pt"), map_location="cpu", weights_only=False)

        cfg = BertConfig(
            vocab_size=self.tokenizer.vocab_size,
            n_layer=enc_cfg["n_layer"],
            n_head=enc_cfg["n_head"],
            n_embd=enc_cfg["n_embd"],
        )
        self.model = BERT(cfg)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.dim = int(enc_cfg["n_embd"])

        for p in self.model.pooler.parameters():
            p.requires_grad = False
        for p in self.model.mlm_head.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        seq, _, _, _ = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return seq


def build_encoder_bundle(spec: dict):
    name = spec["name"]
    kind = spec["kind"]
    max_len = int(spec.get("max_len", 128))

    if kind == "hf":
        tok = AutoTokenizer.from_pretrained(spec["path"], trust_remote_code=spec.get("trust_remote_code", False))
        enc = HFEncoder(spec["path"], trust_remote_code=spec.get("trust_remote_code", False))
    elif kind == "local_bert":
        enc = LocalBertEncoder(spec["path"])
        tok = enc.tokenizer
    else:
        raise ValueError(kind)

    if spec.get("freeze", False):
        for p in enc.parameters():
            p.requires_grad = False

    return {"name": name, "tokenizer": tok, "encoder": enc, "dim": enc.dim, "max_len": max_len}


def build_all_encoders(specs: list[dict]):
    return [build_encoder_bundle(s) for s in specs]
