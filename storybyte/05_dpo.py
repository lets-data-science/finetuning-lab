"""DPO: teach StoryByte-Instruct to PREFER dialogue-rich stories (Module 5).

The DPO loss (Rafailov et al. 2023), computed on story tokens only:
  margin = beta * [ (logp_pol(chosen) - logp_ref(chosen))
                  - (logp_pol(rejected) - logp_ref(rejected)) ]
  loss   = -logsigmoid(margin)

Policy initializes from the SFT checkpoint; the frozen reference IS the SFT
checkpoint (the leash). Success metric (S2 gate): dialogue rate on gold
generations rises measurably vs SFT, without wrecking compliance or fluency.

Resumable: --resume / --max-seconds (same chunked-CPU pattern as 02).
Outputs: checkpoints/dpo.npz, results/dpo_train_trace.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from sb_common.paths import base_artifacts_dir, data_dir, ckpt_dir, out_dir
from sb_common.tokenizer import SBTokenizer, load_config, REQ_TOKEN, STORY_TOKEN, EOS_ID
from sb_common.model import StoryByte, SBConfig

SEED = 1337
BLOCK = 256


def load_pair_tensors(tk, pairs):
    """Tokenize pairs once: (ids, story_start) per side, truncated to BLOCK."""
    out = []
    for p in pairs:
        req_ids = tk.encode(f"{REQ_TOKEN} {p['request']} {STORY_TOKEN}")
        rows = {}
        ok = True
        for side in ("chosen", "rejected"):
            sids = tk.encode(" " + p[side].strip()) + [EOS_ID]
            ids = (req_ids + sids)[:BLOCK]
            if len(req_ids) + 8 > len(ids):
                ok = False
                break
            rows[side] = (ids, len(req_ids))
        if ok:
            out.append(rows)
    return out


def seq_logprob(model, ids: list[int], story_start: int) -> torch.Tensor:
    """Sum of log P(token) over story tokens (differentiable for the policy)."""
    x = torch.tensor([ids[:-1]], dtype=torch.long)
    y = torch.tensor([ids[1:]], dtype=torch.long)
    logits, _ = model(x)
    logp = F.log_softmax(logits, dim=-1)
    tok_lp = logp[0].gather(1, y[0][:, None]).squeeze(1)
    return tok_lp[story_start - 1 :].sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8, help="pairs per step")
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=30.0)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    cfg_json = load_config(base_artifacts_dir())
    tk = SBTokenizer(ckpt_dir() / "tokenizer_ext.json")
    cfg = SBConfig.from_json(cfg_json)
    cfg.vocab_size = tk.vocab_size

    policy = StoryByte(cfg)
    policy.load_npz(ckpt_dir() / "sft_full.npz")
    ref = StoryByte(cfg)
    ref.load_npz(ckpt_dir() / "sft_full.npz")
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    pairs = [json.loads(l) for l in open(data_dir() / "preference_pairs.jsonl")]
    tensors = load_pair_tensors(tk, pairs)

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)

    state_path = ckpt_dir() / "dpo_state.pt"
    trace = {
        "seed": SEED, "steps": args.steps, "batch": args.batch,
        "beta": args.beta, "lr": args.lr, "n_pairs": len(tensors),
        "loss": [], "margin": [], "accuracy": [],
    }
    step, prior_wall = 0, 0.0
    if args.resume and state_path.exists():
        snap = torch.load(state_path, weights_only=False)
        policy.load_state_dict(snap["model"])
        opt.load_state_dict(snap["opt"])
        step, trace, prior_wall = snap["step"], snap["trace"], snap.get("wall", 0.0)
        rng = random.Random(SEED + step)
        print(f"resumed at step {step}")

    policy.train()
    t0 = time.perf_counter()
    while step < args.steps:
        batch = rng.sample(tensors, min(args.batch, len(tensors)))
        losses, margins, correct = [], [], 0
        opt.zero_grad(set_to_none=True)
        for rows in batch:
            c_ids, c_start = rows["chosen"]
            r_ids, r_start = rows["rejected"]
            pol_c = seq_logprob(policy, c_ids, c_start)
            pol_r = seq_logprob(policy, r_ids, r_start)
            with torch.no_grad():
                ref_c = seq_logprob(ref, c_ids, c_start)
                ref_r = seq_logprob(ref, r_ids, r_start)
            margin = args.beta * ((pol_c - ref_c) - (pol_r - ref_r))
            loss = -F.logsigmoid(margin) / len(batch)
            loss.backward()
            losses.append(float(loss) * len(batch))
            margins.append(float(margin))
            correct += margin.item() > 0
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        step += 1
        trace["loss"].append(round(sum(losses) / len(losses), 4))
        trace["margin"].append(round(sum(margins) / len(margins), 4))
        trace["accuracy"].append(round(correct / len(batch), 3))
        if step % 25 == 0 or step == args.steps:
            print(f"step {step} loss {trace['loss'][-1]:.4f} "
                  f"margin {trace['margin'][-1]:.3f} acc {trace['accuracy'][-1]:.2f} "
                  f"({time.perf_counter()-t0:.0f}s)")
        wall = time.perf_counter() - t0
        if wall > args.max_seconds and step < args.steps:
            torch.save({"model": policy.state_dict(), "opt": opt.state_dict(),
                        "step": step, "trace": trace, "wall": prior_wall + wall},
                       state_path)
            print(json.dumps({"paused_at_step": step, "resume": True}))
            return

    trace["wall_seconds"] = round(prior_wall + time.perf_counter() - t0, 1)
    policy.save_npz(ckpt_dir() / "dpo.npz")
    with open(out_dir() / "dpo_train_trace.json", "w") as f:
        json.dump(trace, f, indent=2)
    print(json.dumps({"done": True, "steps": step,
                      "final_margin": trace["margin"][-1],
                      "wall_seconds": trace["wall_seconds"]}))


if __name__ == "__main__":
    main()
