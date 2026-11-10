"""
Everything about getting data ready for the model.

Two tokenizers (the codebook that turns text into numbers and back):
  - CharTokenizer: one number per character. Tiny vocabulary, good for small datasets like names.
  - GPT2Tokenizer: one number per word piece (50,257 of them), used for large datasets.

prepare_data.py uses these to turn raw text into token files (train.bin / val.bin + meta.json).
train.py then reads random chunks straight from those files on disk, so even a
20 GB dataset never has to fit in memory.
"""

import json
import os

import numpy as np
import torch

TOKEN_DTYPE = np.uint16   # each token id is stored in 2 bytes (enough for ids up to 65,535)


class CharTokenizer:
    """One number per character. The vocabulary is every distinct character in the text."""

    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}   # character -> number
        self.itos = {i: ch for i, ch in enumerate(self.chars)}   # number -> character

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    @property
    def eot_id(self):
        """The end-of-example marker: "\\n", since datasets have one example per line."""
        return self.stoi.get("\n")

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i in self.itos)

    def meta(self):
        return {"type": "char", "chars": self.chars}


class GPT2Tokenizer:
    """The word-piece tokenizer GPT-2 uses, e.g. " psychology" -> [22142]."""

    def __init__(self):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")

    @property
    def vocab_size(self):
        return self.enc.n_vocab        # 50,257

    @property
    def eot_id(self):
        """<|endoftext|>: marks the boundary between two documents."""
        return self.enc.eot_token

    def encode(self, s):
        return self.enc.encode_ordinary(s)

    def encode_batch(self, texts):
        return self.enc.encode_ordinary_batch(texts)   # many documents at once, using several CPU threads

    def decode(self, ids):
        # train.py pads the model's vocabulary past 50,257 for speed; skip those unused ids
        return self.enc.decode([i for i in ids if i < self.enc.n_vocab])

    def meta(self):
        return {"type": "gpt2"}


def load_tokenizer(meta):
    """Rebuild a tokenizer from its saved description (from meta.json or a checkpoint)."""
    if meta["type"] == "char":
        return CharTokenizer(meta["chars"])
    if meta["type"] == "gpt2":
        return GPT2Tokenizer()
    raise ValueError(f"unknown tokenizer type: {meta['type']}")


def load_meta(data_dir):
    """Read meta.json from a folder made by prepare_data.py."""
    path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Prepare the data first, e.g.: python prepare_data.py data/names.txt")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_batch(bin_path, block_size, batch_size, device):
    """Pick batch_size random chunks of block_size tokens from a token file.

    x is each chunk, y is the same chunk shifted one token to the right,
    so at every position the target is "the token that comes next".
    """
    # memmap reads only the parts of the file we touch, instead of loading all of it.
    # It's reopened every call because keeping one open makes memory use creep up over long runs.
    data = np.memmap(bin_path, dtype=TOKEN_DTYPE, mode="r")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,)).tolist()
    x = torch.from_numpy(np.stack([data[i : i + block_size] for i in ix]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[i + 1 : i + block_size + 1] for i in ix]).astype(np.int64))
    if device == "cuda":
        # pinned memory lets the copy to the GPU overlap with other work
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)
