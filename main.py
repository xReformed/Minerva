"""
A standard GPT-style transformer (decoder-only), the same design used by GPT-2
and the base of most modern language models.
 
Structure:
    tokens -> token embedding + position embedding
           -> N x [ LayerNorm -> Self-Attention -> add back ]
                  [ LayerNorm -> MLP            -> add back ]
           -> final LayerNorm -> output layer -> next-token scores
 
At the bottom is a tiny training demo on names.txt (character level).
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
 
 
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
# 6. Tiny demo: train a character-level GPT on names.txt
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    # Load text: all names joined by newlines, so "\n" acts as the end-of-name marker
    text = open("names.txt", "r").read()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)
 
    data = torch.tensor(encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
 
    config = GPTConfig(vocab_size=len(chars), block_size=32, n_layer=4, n_head=4, n_embd=128)
    model = GPT(config).to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
 
    def get_batch(split_name, batch_size=64):
        d = train_data if split_name == "train" else val_data
        ix = torch.randint(len(d) - config.block_size - 1, (batch_size,))
        x = torch.stack([d[i : i + config.block_size] for i in ix])          # input
        y = torch.stack([d[i + 1 : i + config.block_size + 1] for i in ix])  # same, shifted by one
        return x.to(device), y.to(device)
 
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
 
    for step in range(3000):
        x, y = get_batch("train")
        _, loss = model(x, y)          # 1. guess + 2. measure
        optimizer.zero_grad()
        loss.backward()                # 3. find the direction
        optimizer.step()               # 4. nudge
 
        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                _, val_loss = model(*get_batch("val"))
            model.train()
            print(f"step {step}: train loss {loss.item():.3f} | val loss {val_loss.item():.3f}")

    model.eval()
    start = torch.tensor([[stoi["\n"]]], device=device)
    out = model.generate(start, max_new_tokens=200, temperature=0.8)
    print(decode(out[0].tolist()))