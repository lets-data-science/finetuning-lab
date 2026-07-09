"""Distillation: StoryByte-Instruct (1.09M teacher) -> StoryByte-Nano student (Module 6).

Classic logit distillation (Hinton, Vinyals & Dean 2015):
  loss = alpha * T^2 * KL(soft_teacher(T) || soft_student(T)) + (1-alpha) * CE(hard)
computed on story tokens only (same masking as SFT).

The module's centerpiece experiment (S3 gate): at EQUAL steps, the KD student
must beat an identical student trained from scratch with plain CE on the same
data. Both runs use identical seeds/architecture/data ordering.

  python3 06_distill_kd.py --mode kd       # student learns with the teacher
  python3 06_distill_kd.py --mode scratch  # control: same student, no teacher

Nano config: 2 layers, 4 heads, d_model 96 (~0.41M params incl. embeddings).
Outputs: checkpoints/nano_kd.npz / nano_scratch.npz + nano_config.json
         results/kd_train_trace_{mode}.json
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

NANO = {"n_layer": 2, "n_head": 4, "n_embd": 96, "block_size": 256, "mlp_ratio": 4}


def pack(tk, e):
    req_ids = tk.encode(f"{REQ_TOKEN} {e['request']} {STORY_TOKEN}")
    story_ids = tk.encode(" " + e["story"].strip()) + [EOS_ID]
    ids = req_ids + story_ids
    if len(ids) > BLOCK:
        return None
    mask = [0] * len(req_ids) + [1] * len(story_ids)
    pad = BLOCK - len(ids)
    return ids + [EOS_ID] * pad, mask + [0] * pad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["kd", "scratch"], required=True)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=30.0)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    cfg_json = load_config(base_artifacts_dir())
    tk = SBTokenizer(ckpt_dir() / "tokenizer_ext.json")

    nano_cfg = SBConfig(**NANO, vocab_size=tk.vocab_size)
    student = StoryByte(nano_cfg)
    with open(ckpt_dir() / "nano_config.json", "w") as f:
        json.dump({**NANO, "vocab_size": tk.vocab_size,
                   "n_params": student.num_params()}, f, indent=2)

    teacher = None
    if args.mode == "kd":
        tcfg = SBConfig.from_json(cfg_json)
        tcfg.vocab_size = tk.vocab_size
        teacher = StoryByte(tcfg)
        teacher.load_npz(ckpt_dir() / "sft_full.npz")
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    train = [json.loads(l) for l in open(data_dir() / "requests_train.jsonl")]
    packed = [p for e in train if (p := pack(tk, e)) is not None]

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    state_path = ckpt_dir() / f"kd_state_{args.mode}.pt"
    trace = {"mode": args.mode, "seed": SEED, "steps": args.steps,
             "kd_temp": args.kd_temp, "alpha": args.alpha, "lr": args.lr,
             "student_params": student.num_params(), "loss": []}
    step, prior_wall = 0, 0.0
    if args.resume and state_path.exists():
        snap = torch.load(state_path, weights_only=False)
        student.load_state_dict(snap["model"])
        opt.load_state_dict(snap["opt"])
        step, trace, prior_wall = snap["step"], snap["trace"], snap.get("wall", 0.0)
        rng = random.Random(SEED + step)
        print(f"resumed {args.mode} at step {step}")

    student.train()
    t0 = time.perf_counter()
    idx = list(range(len(packed)))
    while step < args.steps:
        rng.shuffle(idx)
        for i in range(0, len(idx) - args.batch + 1, args.batch):
            chunk = [packed[j] for j in idx[i : i + args.batch]]
            x = torch.tensor([c[0] for c in chunk], dtype=torch.long)
            m = torch.tensor([c[1] for c in chunk], dtype=torch.float32)
            xin, y, mm = x[:, :-1], x[:, 1:], m[:, 1:]
            s_logits, _ = student(xin)
            ce = F.cross_entropy(s_logits.reshape(-1, s_logits.size(-1)),
                                 y.reshape(-1), reduction="none").view_as(y.float())
            ce = (ce * mm).sum() / mm.sum().clamp(min=1)
            if teacher is not None:
                with torch.no_grad():
                    t_logits, _ = teacher(xin)
                T = args.kd_temp
                kl = F.kl_div(
                    F.log_softmax(s_logits / T, dim=-1),
                    F.softmax(t_logits / T, dim=-1),
                    reduction="none",
                ).sum(-1)
                kl = (kl * mm).sum() / mm.sum().clamp(min=1)
                loss = args.alpha * (T * T) * kl + (1 - args.alpha) * ce
            else:
                loss = ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            step += 1
            trace["loss"].append(round(float(loss.detach()), 4))
            if step % 100 == 0 or step == args.steps:
                print(f"{args.mode} step {step} loss {float(loss.detach()):.4f} "
                      f"({time.perf_counter()-t0:.0f}s)")
            wall = time.perf_counter() - t0
            if wall > args.max_seconds and step < args.steps:
                torch.save({"model": student.state_dict(), "opt": opt.state_dict(),
                            "step": step, "trace": trace, "wall": prior_wall + wall},
                           state_path)
                print(json.dumps({"paused_at_step": step, "resume": True}))
                return
            if step >= args.steps:
                break

    trace["wall_seconds"] = round(prior_wall + time.perf_counter() - t0, 1)
    student.save_npz(ckpt_dir() / f"nano_{args.mode}.npz")
    with open(out_dir() / f"kd_train_trace_{args.mode}.json", "w") as f:
        json.dump(trace, f, indent=2)
    print(json.dumps({"done": True, "mode": args.mode,
                      "student_params": trace["student_params"],
                      "wall_seconds": trace["wall_seconds"]}))


if __name__ == "__main__":
    main()
