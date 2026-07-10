"""LoRA: fine-tune StoryByte on the request task with low-rank adapters (Module 4).

What's frozen and what trains:
  - ALL pretrained base values stay frozen. LoRA runs start from the
    SFT-extended tokenizer, re-initialize the four special-token rows, and train:
      (a) the special-token embedding rows (they're new - someone must teach them), and
      (b) rank-r adapters. In row-vector notation the update is
          x @ A @ B and the merged weight is W + (alpha/r) * A @ B on every attention projection
          (c_attn, c_proj) in all 4 blocks.
  This matches the honest LoRA story: base knowledge frozen, tiny task-specific
  addition trains.

Merging: 07's job normally; here we also export a merged full .npz per rank so
the eval suite and the browser can run it with the standard loader
(checkpoints/lora_r{r}_merged.npz) plus the adapter-only file
(checkpoints/lora_r{r}_adapter.npz - the thing whose SIZE the course brags about).

Resumable: --resume --max-seconds 30 (same pattern as 02).
Run: python3 03_train_lora.py --rank 4 [--steps 450]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn

from sb_common.paths import base_artifacts_dir, data_dir, ckpt_dir, out_dir
from sb_common.tokenizer import SBTokenizer, load_config, EOS_ID
from sb_common.model import StoryByte, SBConfig

# reuse the SFT machinery so data packing / probes can never drift
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("sft", Path(__file__).parent / "02_sft_full.py")
_sft = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sft)

SEED = 1337


class LoRAAdapter(nn.Module):
    """Low-rank update attached to a frozen ``nn.Linear``.

    The course uses row-vector notation: ``x @ A_row @ B_row`` and
    ``W_row + scale * A_row @ B_row``. PyTorch stores Linear weights as
    ``(out, in)``, and the shipped adapter format stores those transposes as
    ``A=(r,in)`` and ``B=(out,r)``. The forward pass below is the same algebra
    without changing the historical checkpoint layout.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, seed: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        g = torch.Generator().manual_seed(seed)
        self.A = nn.Parameter(torch.randn(r, base.in_features, generator=g) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.t() @ self.B.t())

    def delta(self) -> torch.Tensor:
        # PyTorch storage orientation: (A_row @ B_row).T == B @ A.
        return self.scale * (self.B @ self.A)


def attach_lora(model: StoryByte, r: int, alpha: float) -> list[tuple[str, LoRAAdapter]]:
    adapters = []
    for i, block in enumerate(model.h):
        for name in ("c_attn", "c_proj"):
            base = getattr(block.attn, name)
            ad = LoRAAdapter(base, r, alpha, seed=SEED + 31 * i + len(name))
            setattr(block.attn, name, ad)
            adapters.append((f"h.{i}.attn.{name}", ad))
    return adapters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--alpha", type=float, default=None, help="default 2*r")
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=30.0)
    args = ap.parse_args()
    alpha = args.alpha if args.alpha is not None else 2.0 * args.rank

    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    base_dir = base_artifacts_dir()
    cfg_json = load_config(base_dir)
    tk = SBTokenizer(ckpt_dir() / "tokenizer_ext.json")

    cfg = SBConfig.from_json(cfg_json)
    model = StoryByte(cfg)
    model.load_npz(base_dir / "storybyte_weights.npz")  # BASE weights, not SFT
    model.resize_vocab(tk.vocab_size)
    extended_base_params = sum(p.numel() for p in model.parameters())

    # freeze everything...
    for p in model.parameters():
        p.requires_grad_(False)
    # ...except the brand-new special-token embedding rows (masked-gradient trick)
    n_new = tk.vocab_size - cfg_json["vocab_size"]
    model.wte.weight.requires_grad_(True)
    new_row_mask = torch.zeros_like(model.wte.weight)
    new_row_mask[-n_new:] = 1.0

    adapters = attach_lora(model, args.rank, alpha)
    lora_params = [p for _, ad in adapters for p in (ad.A, ad.B)]
    adapter_matrix_params = sum(p.numel() for p in lora_params)
    special_token_embedding_params = n_new * cfg.n_embd
    trainable = adapter_matrix_params + special_token_embedding_params

    train = [json.loads(l) for l in open(data_dir() / "requests_train.jsonl")]
    val = [json.loads(l) for l in open(data_dir() / "requests_val.jsonl")]
    packed_train = [p for e in train if (p := _sft.pack_example(tk, e)) is not None]
    packed_val = [p for e in val if (p := _sft.pack_example(tk, e)) is not None]
    dev_probe = _sft.build_dev_probe(val)

    opt = torch.optim.AdamW(
        [{"params": lora_params}, {"params": [model.wte.weight]}],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0,
    )

    state_path = ckpt_dir() / f"lora_r{args.rank}_state.pt"
    trace = {
        "rank": args.rank, "alpha": alpha, "lr": args.lr, "steps": args.steps,
        "seed": SEED, "trainable_params": int(trainable),
        "adapter_matrix_params": int(adapter_matrix_params),
        "special_token_embedding_params": int(special_token_embedding_params),
        "base_params": int(extended_base_params),
        "train_loss": [], "val_loss": [], "probe_curve": [],
        "checkpoint_policy": "best dev-probe compliance (same probe as SFT)",
    }
    step, prior_wall = 0, 0.0
    best_probe, best_probe_step, best_state = -1.0, -1, None
    if args.resume and state_path.exists():
        snap = torch.load(state_path, weights_only=False)
        model.load_state_dict(snap["model"])
        opt.load_state_dict(snap["opt"])
        step, trace, prior_wall = snap["step"], snap["trace"], snap.get("wall", 0.0)
        best_probe = snap.get("best_probe", -1.0)
        best_probe_step = snap.get("best_probe_step", -1)
        best_state = snap.get("best_state")
        rng = random.Random(SEED + step)
        print(f"resumed r{args.rank} at step {step}")

    def lr_at(s):
        warm = 40
        if s < warm:
            return args.lr * (s + 1) / warm
        t = (s - warm) / max(args.steps - warm, 1)
        return 0.1 * args.lr + 0.45 * args.lr * (1 + math.cos(math.pi * t))

    model.train()
    t0 = time.perf_counter()
    while step < args.steps:
        for x, y, m in _sft.make_batches(packed_train, args.batch, rng):
            for grp in opt.param_groups:
                grp["lr"] = lr_at(step)
            _, loss = model(x, targets=y, loss_mask=m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if model.wte.weight.grad is not None:
                model.wte.weight.grad *= new_row_mask  # only new rows may move
            torch.nn.utils.clip_grad_norm_(
                lora_params + [model.wte.weight], 1.0)
            opt.step()
            step += 1
            trace["train_loss"].append(round(float(loss.detach()), 4))
            if step % args.eval_every == 0 or step == args.steps:
                vl = _sft.val_loss(model, packed_val)
                pc = _sft.probe_compliance(model, tk, dev_probe)
                trace["val_loss"].append({"step": step, "loss": round(vl, 4)})
                trace["probe_curve"].append({"step": step, "compliance": round(pc, 4)})
                print(f"r{args.rank} step {step} train {float(loss.detach()):.4f} "
                      f"val {vl:.4f} probe {pc:.2f} ({time.perf_counter()-t0:.0f}s)")
                if pc > best_probe:
                    best_probe, best_probe_step = pc, step
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wall = time.perf_counter() - t0
            if wall > args.max_seconds and step < args.steps:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step, "trace": trace, "wall": prior_wall + wall,
                            "best_probe": best_probe, "best_probe_step": best_probe_step,
                            "best_state": best_state}, state_path)
                print(json.dumps({"paused_at_step": step, "resume": True}))
                return
            if step >= args.steps:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    trace["best_probe_compliance"] = round(best_probe, 4)
    trace["best_probe_step"] = best_probe_step
    trace["wall_seconds"] = round(prior_wall + time.perf_counter() - t0, 1)

    # ---- export: adapter-only npz + merged full npz ----
    adapter_out = {}
    for name, ad in adapters:
        adapter_out[name + ".A"] = ad.A.detach().numpy().astype(np.float32)
        adapter_out[name + ".B"] = ad.B.detach().numpy().astype(np.float32)
        adapter_out[name + ".scale"] = np.array([ad.scale], dtype=np.float32)
    adapter_out["wte_new_rows"] = model.wte.weight.detach()[-n_new:].numpy().astype(np.float32)
    np.savez(ckpt_dir() / f"lora_r{args.rank}_adapter.npz", **adapter_out)

    # Course row-vector notation: W_row' = W_row + scale * A_row @ B_row.
    # nn.Linear stores W_row.T, so ad.delta() returns the stored transpose.
    for name, ad in adapters:
        with torch.no_grad():
            ad.base.weight += ad.delta()
        i = int(name.split(".")[1])
        attr = name.split(".")[-1]
        setattr(model.h[i].attn, attr, ad.base)
    model.save_npz(ckpt_dir() / f"lora_r{args.rank}_merged.npz")

    trace["adapter_bytes"] = int(Path(ckpt_dir() / f"lora_r{args.rank}_adapter.npz").stat().st_size)
    with open(out_dir() / f"lora_r{args.rank}_trace.json", "w") as f:
        json.dump(trace, f, indent=2)
    print(json.dumps({
        "rank": args.rank, "trainable_params": trace["trainable_params"],
        "best_probe_compliance": trace["best_probe_compliance"],
        "best_probe_step": best_probe_step,
        "adapter_bytes": trace["adapter_bytes"],
        "wall_seconds": trace["wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
