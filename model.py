"""
A standard GPT-style transformer (decoder-only), the same design used by GPT-2
and the base of most modern language models.

Structure:
    tokens -> token embedding + position embedding
           -> N x [ LayerNorm -> Self-Attention -> add back ]
                  [ LayerNorm -> MLP            -> add back ]
           -> final LayerNorm -> output layer -> next-token scores

Also has save_checkpoint / load_checkpoint for storing a trained model.
Training lives in train.py, generating text in sample.py.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import asdict, dataclass


# ---------------------------------------------------------------------------
# 1. Settings: the "size knobs" of the model
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab_size: int = 65      # how many different tokens exist
    block_size: int = 64      # how many tokens the model can look back at (context length)
    n_layer: int = 4          # how many transformer blocks are stacked
    n_head: int = 4           # how many attention heads per block
    n_embd: int = 128         # size of each token's vector (must divide by n_head)
    dropout: float = 0.1      # randomly switch off some values during training to reduce overfitting


# ---------------------------------------------------------------------------
# 2. Self-attention: tokens look back at earlier tokens and gather information
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # One layer makes Query, Key and Value for every token at once
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Mixes the heads' results back together
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape   # Batch size, Time (sequence length), Channels (n_embd)

        # Query = "what am I looking for?", Key = "what do I contain?", Value = "what I pass on"
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)

        # Split into heads: (B, T, C) -> (B, n_head, T, head_size)
        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        # Attention: each token scores every earlier token, turns scores into weights,
        # and takes a weighted mix of their values.
        # is_causal=True stops tokens from looking at the future.
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # Join heads back together: (B, n_head, T, head_size) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


# ---------------------------------------------------------------------------
# 3. MLP: each token "thinks" on its own about what it just gathered
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)    # expand
        self.gelu = nn.GELU()                                    # non-linearity
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)  # shrink back
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.proj(self.gelu(self.fc(x))))


# ---------------------------------------------------------------------------
# 4. Transformer block: attention + MLP, each with a residual ("add back") connection
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # communicate with other tokens, add result back
        x = x + self.mlp(self.ln2(x))    # think individually, add result back
        return x


# ---------------------------------------------------------------------------
# 5. The full GPT model
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)   # token id -> vector
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)   # position -> vector
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # vector -> scores

        self.head.weight = self.tok_emb.weight   # weight tying: share input and output tables
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Start with small random weights (standard GPT-2 initialization)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, "sequence longer than block_size"

        positions = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(positions)   # what the token is + where it is
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)                             # (B, T, vocab_size) scores

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Predict one token at a time and append it, max_new_tokens times."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]      # only the last block_size tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature          # scores for the next token only
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")  # keep only the top_k choices
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # weighted dice roll
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ---------------------------------------------------------------------------
# 6. Saving and loading a trained model
# ---------------------------------------------------------------------------
def save_checkpoint(path, model, tokenizer, **info):
    """Save everything needed to use the model later without retraining.

    tokenizer is the tokenizer's description (e.g. {"type": "gpt2"}), so text can be decoded later.
    """
    tmp_path = path + ".tmp"
    torch.save({
        "model": model.state_dict(),       # the learned weights
        "config": asdict(model.config),    # the size knobs, to rebuild the same model
        "tokenizer": tokenizer,            # how to turn numbers back into text
        **info,                            # extras, e.g. step, val_loss, optimizer state
    }, tmp_path)
    # Write to a temporary file first, then swap it in: if the PC dies mid-save,
    # the previous checkpoint is still intact.
    os.replace(tmp_path, path)


def load_checkpoint(path, device="cpu"):
    """Rebuild a saved model, ready to generate. Returns (model, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt
