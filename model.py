"""
Byte-level next-character predictor + training pipeline.

A small GPT-style decoder that reads raw bytes (vocab = 256) and predicts the
next byte. Designed to fit comfortably in 4 GB VRAM (RTX 3050 Laptop) when
trained with mixed precision.

Run directly to train on training.txt and export lm.pth:
    python model.py
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler


# ============================================================================
# Model
# ============================================================================

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with a fixed lower-triangular mask."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(C, dim=2)

        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """Transformer block: LayerNorm -> Attention -> residual, then MLP -> residual."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(
            {
                "c_fc": nn.Linear(n_embd, 4 * n_embd),
                "c_proj": nn.Linear(4 * n_embd, n_embd),
                "dropout": nn.Dropout(dropout),
            }
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp["dropout"](
            self.mlp["c_proj"](F.gelu(self.mlp["c_fc"](self.ln2(x))))
        )
        return x


class ByteGPT(nn.Module):
    """
    Byte-level GPT decoder.

    Args:
        vocab_size: number of byte values (256)
        n_embd: embedding / hidden dimension
        n_layer: number of transformer blocks
        n_head: number of attention heads
        block_size: maximum context length (sequence length)
        dropout: dropout probability
    """

    def __init__(self, vocab_size=256, n_embd=384, n_layer=6, n_head=6,
                 block_size=512, dropout=0.1):
        super().__init__()
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: share the embedding matrix with the output head.
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        """
        Args:
            idx: (B, T) long tensor of byte indices
            targets: (B, T) long tensor of next-byte targets, or None

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar cross-entropy loss (None if targets is None)
        """
        B, T = idx.size()
        assert T <= self.block_size, (
            f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        )

        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb[:, :T, :]
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=0):
        """
        Autoregressively sample new bytes starting from `idx`.

        If top_k > 0, print the top-k most likely next bytes and their
        probabilities (from the sampling distribution) at each step.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            if top_k > 0:
                top_probs, top_idx = torch.topk(probs[0], min(top_k, probs.size(-1)))
                parts = []
                for p, b in zip(top_probs.tolist(), top_idx.tolist()):
                    label = repr(chr(b)) if 32 <= b < 127 else f"\\x{b:02x}"
                    parts.append(f"{label}: {p*100:.1f}%")
                print("  top bytes: " + " | ".join(parts))

            # Greedy: always pick the single most likely byte (deterministic)
            idx_next = probs.argmax(dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============================================================================
# Training configuration
# ============================================================================

config = {
    # Data
    "data_file": "training.txt",
    "save_file": "lm.pth",
    "train_split": 0.90,        # 90 MB train / 10 MB val

    # Model
    "vocab_size": 256,          # raw bytes
    "n_embd": 384,              # hidden dimension
    "n_layer": 6,               # transformer blocks
    "n_head": 6,                # attention heads
    "block_size": 512,          # context length (bytes the model can see back)
    "dropout": 0.1,

    # Training
    "batch_size": 16,           # sequences per micro-batch
    "gradient_accumulation_steps": 4,   # effective batch = 64
    "max_iters": 5000,          # training iterations
    "eval_interval": 500,       # validate every N iters
    "eval_iters": 200,          # batches used for validation estimate
    "learning_rate": 1e-3,
    "min_lr": 1e-4,             # LR floor (learning_rate / 10)
    "warmup_iters": 200,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,

    # System
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": "float16",         # mixed precision

    # Goals / milestones (bits-per-byte on validation)
    "goals": [2.0, 1.8, 1.6, 1.4, 1.2],
}

device_type = "cuda" if "cuda" in config["device"] else "cpu"
ptdtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[config["dtype"]]


# ============================================================================
# Training pipeline
# ============================================================================

def train():
    """Load training.txt, train ByteGPT, save lm.pth. Print milestones/goals."""
    # --- Step 1: Load data ---------------------------------------------------
    print(f"[1/6] Loading {config['data_file']} ...")
    with open(config["data_file"], "rb") as f:
        raw = f.read()

    data = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(dtype=torch.long)
    print(f"    Loaded {len(data):,} bytes ({len(data)/1e6:.1f} MB)")

    n = int(config["train_split"] * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"    Train: {len(train_data):,} bytes | Val: {len(val_data):,} bytes")

    # --- Step 2: Data sampler ------------------------------------------------
    def get_batch(split):
        """Sample a random batch of (input, target) byte windows."""
        source = train_data if split == "train" else val_data
        ix = torch.randint(len(source) - config["block_size"], (config["batch_size"],))
        x = torch.stack([source[i : i + config["block_size"]] for i in ix])
        y = torch.stack([source[i + 1 : i + config["block_size"] + 1] for i in ix])
        if device_type == "cuda":
            x = x.pin_memory().to(config["device"], non_blocking=True)
            y = y.pin_memory().to(config["device"], non_blocking=True)
        else:
            x = x.to(config["device"])
            y = y.to(config["device"])
        return x, y

    # --- Step 3: Build model -------------------------------------------------
    print("[2/6] Building ByteGPT ...")
    model = ByteGPT(
        vocab_size=config["vocab_size"],
        n_embd=config["n_embd"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        block_size=config["block_size"],
        dropout=config["dropout"],
    ).to(config["device"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params:,}  (~{n_params * 2 / 1e6:.1f} MB in FP16)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(config["beta1"], config["beta2"]),
    )
    scaler = GradScaler("cuda", enabled=(config["dtype"] == "float16"))

    # --- Step 4: Helpers -----------------------------------------------------
    def get_lr(it):
        """Cosine decay with linear warmup."""
        if it < config["warmup_iters"]:
            return config["learning_rate"] * it / config["warmup_iters"]
        if it > config["max_iters"]:
            return config["min_lr"]
        decay_ratio = (it - config["warmup_iters"]) / (
            config["max_iters"] - config["warmup_iters"]
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config["min_lr"] + coeff * (config["learning_rate"] - config["min_lr"])

    @torch.no_grad()
    def estimate_loss():
        """Run several val/train batches and report mean loss + bits-per-byte."""
        out = {}
        model.eval()
        for split in ("train", "val"):
            losses = torch.zeros(config["eval_iters"], device=config["device"])
            for k in range(config["eval_iters"]):
                X, Y = get_batch(split)
                with autocast("cuda", dtype=ptdtype,
                              enabled=(config["dtype"] == "float16")):
                    _, loss = model(X, Y)
                losses[k] = loss
            out[split] = losses.mean().item()
        model.train()
        return out

    # --- Step 5: Training loop -----------------------------------------------
    print(f"[3/6] Training on {config['device']} (mixed precision={config['dtype']}) ...")
    print(f"    Goals (val bpb): {config['goals']}")
    print(f"    {config['max_iters']} iters, eval every {config['eval_interval']}, "
          f"effective batch = {config['batch_size'] * config['gradient_accumulation_steps']}")
    print("-" * 78)

    t0 = time.time()
    running_loss = 0.0

    reached_goals = {g: False for g in sorted(config["goals"], reverse=True)}

    def check_goals(iter_num, val_bpb):
        """Print a message when a new bits-per-byte goal is reached."""
        for g in sorted(config["goals"], reverse=True):
            if not reached_goals[g] and val_bpb <= g:
                reached_goals[g] = True
                print(f"    *** GOAL REACHED at iter {iter_num}: val bpb {val_bpb:.4f} <= {g} ***")

    for iter_num in range(config["max_iters"] + 1):
        # --- Validation snapshot ---------------------------------------------
        if iter_num % config["eval_interval"] == 0 or iter_num == config["max_iters"]:
            stats = estimate_loss()
            dt = time.time() - t0
            train_bpb = stats["train"] / math.log(2)
            val_bpb = stats["val"] / math.log(2)
            est_mb = val_bpb * 1e8 / 8 / 1e6
            print(
                f"[iter {iter_num:5d}/{config['max_iters']}] "
                f"train loss {stats['train']:.4f} (bpb {train_bpb:.4f}) | "
                f"val loss {stats['val']:.4f} (bpb {val_bpb:.4f}) | "
                f"est. compressed {est_mb:.2f} MB | "
                f"{dt*1000:.0f} ms since last eval"
            )
            check_goals(iter_num, val_bpb)
            t0 = time.time()

        if iter_num == config["max_iters"]:
            break

        # --- Training micro-batches with gradient accumulation ---------------
        for micro in range(config["gradient_accumulation_steps"]):
            xb, yb = get_batch("train")
            lr = get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            with autocast("cuda", dtype=ptdtype,
                          enabled=(config["dtype"] == "float16")):
                _, loss = model(xb, yb)
                loss = loss / config["gradient_accumulation_steps"]

            scaler.scale(loss).backward()
            running_loss += loss.item()

        # Gradient clipping + optimizer step
        if config["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    # --- Step 6: Export ------------------------------------------------------
    final_stats = estimate_loss()
    final_bpb = final_stats["val"] / math.log(2)
    est_mb = final_bpb * 1e8 / 8 / 1e6

    print("-" * 78)
    print("[TRAINING COMPLETE]")
    print(f"    Val loss:     {final_stats['val']:.4f}")
    print(f"    Val bpb:      {final_bpb:.4f}")
    print(f"    Est. size:    {est_mb:.2f} MB on enwik8 (100 MB)")
    print("    Goals reached:")
    for g in sorted(config["goals"], reverse=True):
        flag = "YES" if reached_goals[g] else "no"
        print(f"      bpb <= {g}: {flag}")

    print(f"[4/6] Saving model to {config['save_file']} ...")
    torch.save(model.state_dict(), config["save_file"])
    print(f"[5/6] Model exported to {config['save_file']}")

    return model


if __name__ == "__main__":
    train()
