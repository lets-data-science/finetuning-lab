"""StoryByte in PyTorch - an exact port of the shipped 1.09M-param nano-GPT.

Architecture (from storybyte_config.json): 4 layers, 4 heads, d_model 128,
vocab 2048, context 256, GELU(tanh), pre-LN, learned absolute positions,
weight-tied embedding/unembedding, biases everywhere.

The shipped weights live in an .npz with GPT-2 convention: linear weights are
stored as (in_features, out_features); nn.Linear stores (out, in), so we
transpose on load/save. 00_env_check.py verifies this port reproduces the
course's published greedy sample exactly before anything is trained.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SBConfig:
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    vocab_size: int = 2048
    block_size: int = 256
    mlp_ratio: int = 4

    @classmethod
    def from_json(cls, cfg: dict) -> "SBConfig":
        return cls(
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            n_embd=cfg["n_embd"],
            vocab_size=cfg["vocab_size"],
            block_size=cfg["block_size"],
            mlp_ratio=cfg.get("mlp_ratio", 4),
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: SBConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=True)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: SBConfig):
        super().__init__()
        hidden = cfg.mlp_ratio * cfg.n_embd
        self.c_fc = nn.Linear(cfg.n_embd, hidden, bias=True)
        self.c_proj = nn.Linear(hidden, cfg.n_embd, bias=True)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: SBConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class StoryByte(nn.Module):
    def __init__(self, cfg: SBConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        # weight-tied unembedding: logits = h @ wte.T  (no separate lm_head)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block size"
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)[None, :, :]
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = x @ self.wte.weight.t()
        loss = None
        if targets is not None:
            per_tok = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="none",
            ).view(B, T)
            if loss_mask is not None:
                denom = loss_mask.sum().clamp(min=1)
                loss = (per_tok * loss_mask).sum() / denom
            else:
                loss = per_tok.mean()
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 40,
        eos_id: int = 0,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Batched autoregressive generation. Rows freeze once they emit eos_id."""
        self.eval()
        B = idx.size(0)
        done = torch.zeros(B, dtype=torch.bool, device=idx.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-6)
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator)
            nxt = torch.where(done[:, None], torch.full_like(nxt, eos_id), nxt)
            idx = torch.cat([idx, nxt], dim=1)
            done = done | (nxt.squeeze(1) == eos_id)
            if bool(done.all()):
                break
        return idx

    # ---------- vocab surgery (Module 2 mechanics) ----------

    def resize_vocab(self, new_size: int, init_std: float = 0.02, seed: int = 1337) -> None:
        """Grow the (tied) embedding table for new special tokens.

        New rows are small random vectors - they mean nothing until fine-tuning
        teaches them a job. This is the standard 'add special tokens' surgery.
        """
        old = self.wte.weight.data
        if new_size == old.size(0):
            return
        assert new_size > old.size(0)
        g = torch.Generator().manual_seed(seed)
        extra = torch.randn(new_size - old.size(0), old.size(1), generator=g) * init_std
        self.wte = nn.Embedding(new_size, old.size(1))
        self.wte.weight.data = torch.cat([old, extra], dim=0)
        self.cfg.vocab_size = new_size

    # ---------- npz IO (matches the shipped web-artifact layout) ----------

    def load_npz(self, path: str | Path) -> None:
        z = np.load(str(path))
        sd: dict[str, torch.Tensor] = {}
        sd["wte.weight"] = torch.from_numpy(z["wte"].copy())
        sd["wpe.weight"] = torch.from_numpy(z["wpe"].copy())
        sd["ln_f.weight"] = torch.from_numpy(z["ln_f.g"].copy())
        sd["ln_f.bias"] = torch.from_numpy(z["ln_f.b"].copy())
        for i in range(self.cfg.n_layer):
            p = f"h.{i}."
            sd[p + "ln_1.weight"] = torch.from_numpy(z[p + "ln_1.g"].copy())
            sd[p + "ln_1.bias"] = torch.from_numpy(z[p + "ln_1.b"].copy())
            sd[p + "ln_2.weight"] = torch.from_numpy(z[p + "ln_2.g"].copy())
            sd[p + "ln_2.bias"] = torch.from_numpy(z[p + "ln_2.b"].copy())
            sd[p + "attn.c_attn.weight"] = torch.from_numpy(z[p + "attn.c_attn.w"].copy()).t().contiguous()
            sd[p + "attn.c_attn.bias"] = torch.from_numpy(z[p + "attn.c_attn.b"].copy())
            sd[p + "attn.c_proj.weight"] = torch.from_numpy(z[p + "attn.c_proj.w"].copy()).t().contiguous()
            sd[p + "attn.c_proj.bias"] = torch.from_numpy(z[p + "attn.c_proj.b"].copy())
            sd[p + "mlp.c_fc.weight"] = torch.from_numpy(z[p + "mlp.c_fc.w"].copy()).t().contiguous()
            sd[p + "mlp.c_fc.bias"] = torch.from_numpy(z[p + "mlp.c_fc.b"].copy())
            sd[p + "mlp.c_proj.weight"] = torch.from_numpy(z[p + "mlp.c_proj.w"].copy()).t().contiguous()
            sd[p + "mlp.c_proj.bias"] = torch.from_numpy(z[p + "mlp.c_proj.b"].copy())
        missing, unexpected = self.load_state_dict(sd, strict=True)
        assert not missing and not unexpected

    def save_npz(self, path: str | Path, dtype=np.float32) -> None:
        sd = self.state_dict()
        out: dict[str, np.ndarray] = {}
        out["wte"] = sd["wte.weight"].numpy().astype(dtype)
        out["wpe"] = sd["wpe.weight"].numpy().astype(dtype)
        out["ln_f.g"] = sd["ln_f.weight"].numpy().astype(dtype)
        out["ln_f.b"] = sd["ln_f.bias"].numpy().astype(dtype)
        for i in range(self.cfg.n_layer):
            p = f"h.{i}."
            out[p + "ln_1.g"] = sd[p + "ln_1.weight"].numpy().astype(dtype)
            out[p + "ln_1.b"] = sd[p + "ln_1.bias"].numpy().astype(dtype)
            out[p + "ln_2.g"] = sd[p + "ln_2.weight"].numpy().astype(dtype)
            out[p + "ln_2.b"] = sd[p + "ln_2.bias"].numpy().astype(dtype)
            out[p + "attn.c_attn.w"] = sd[p + "attn.c_attn.weight"].numpy().T.astype(dtype)
            out[p + "attn.c_attn.b"] = sd[p + "attn.c_attn.bias"].numpy().astype(dtype)
            out[p + "attn.c_proj.w"] = sd[p + "attn.c_proj.weight"].numpy().T.astype(dtype)
            out[p + "attn.c_proj.b"] = sd[p + "attn.c_proj.bias"].numpy().astype(dtype)
            out[p + "mlp.c_fc.w"] = sd[p + "mlp.c_fc.weight"].numpy().T.astype(dtype)
            out[p + "mlp.c_fc.b"] = sd[p + "mlp.c_fc.bias"].numpy().astype(dtype)
            out[p + "mlp.c_proj.w"] = sd[p + "mlp.c_proj.weight"].numpy().T.astype(dtype)
            out[p + "mlp.c_proj.b"] = sd[p + "mlp.c_proj.bias"].numpy().astype(dtype)
        np.savez(str(path), **out)


def load_base_model(base_dir: str | Path, cfg_json: dict) -> StoryByte:
    cfg = SBConfig.from_json(cfg_json)
    model = StoryByte(cfg)
    model.load_npz(Path(base_dir) / "storybyte_weights.npz")
    model.eval()
    return model
