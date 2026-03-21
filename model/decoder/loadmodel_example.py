import torch
from tokenizer import SmilesTokenizer
from model import GPTConfig, GPT

MODEL_SPECS = {
    "1M":   dict(n_layer=4,  n_head=4,  n_embd=160),
    "4M":   dict(n_layer=5,  n_head=4,  n_embd=256),
    "16M":  dict(n_layer=5,  n_head=4,  n_embd=512),
    "56M":  dict(n_layer=6,  n_head=16, n_embd=768),
    "85M":  dict(n_layer=12, n_head=12, n_embd=768),
    "152M": dict(n_layer=12, n_head=16, n_embd=1024),
    "278M": dict(n_layer=22, n_head=16, n_embd=1024),
    "650M": dict(n_layer=13, n_head=32, n_embd=2048),
}

def build_tokenizer(vocab_path="vocabs/vocab.txt"):
    tokenizer = SmilesTokenizer(vocab_path)
    tokenizer.bos_token = "[BOS]"
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    tokenizer.eos_token = "[EOS]"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")
    tokenizer.sep_token = "[SEP]"
    tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids("[SEP]")
    return tokenizer

def load_pretrained_model(weight_path, model_size, vocab_path="vocabs/vocab.txt", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = build_tokenizer(vocab_path)

    mconf = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        **MODEL_SPECS[model_size]
    )
    model = GPT(mconf).to(device)

    state_dict = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, tokenizer, device

@torch.no_grad()
def generate_smiles(model, tokenizer, device, max_seq_len=256, temperature=1.0, top_k=50):
    x = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)

    y = next(model.generate(
        x,
        tokenizer,
        max_new_tokens=max_seq_len,
        temperature=temperature,
        top_k=top_k,
        stream=False,
        rp=1.0,
        kv_cache=True,
        is_simulation=True,
    ))

    text = tokenizer.decode(y[0].tolist())
    text = text.replace(" ", "")
    text = text.replace("[BOS]", "")
    text = text.replace("[EOS]", "")
    text = text.replace("[SEP]", "")
    return text

@torch.no_grad()
def calc_logits(model, tokenizer, smiles, device):
    ids = [tokenizer.bos_token_id]
    ids += tokenizer.encode(smiles, add_special_tokens=False)
    ids += [tokenizer.eos_token_id]

    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    logits, _, _ = model(x, tokenizer)
    return logits

if __name__ == "__main__":
    weight_path = "weights/SMILES-650M-3B-Epoch1.pt"
    model_size = "650M"

    model, tokenizer, device = load_pretrained_model(
        weight_path=weight_path,
        model_size=model_size,
        vocab_path="vocabs/vocab.txt",
    )

    out = generate_smiles(model, tokenizer, device, max_seq_len=256, temperature=1.0, top_k=50)
    print(out)