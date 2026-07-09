"""Full-parameter SFT: teach StoryByte to follow requests (Module 3, for real).

Mechanics on display:
  - tokenizer extension: <|req|> / <|story|> appended (ids 2048, 2049) and the
    tied embedding matrix resized to match (Module 2's surgery).
  - loss masking: cross-entropy counts ONLY story tokens + the terminating
    <|endoftext|>; the request tokens are context, not curriculum.
  - low LR + val-curve checkpointing; every step logged for the course's
    Training-Run Scrubber animation.

Outputs:
  checkpoints/sft_full.npz          — best-val checkpoint (browser-export layout)
  checkpoints/tokenizer_ext.json    — the extended tokenizer
  results/sft_train_trace.json      — full loss curves + sample generations per eval
Run: python3 02_sft_full.py [--steps 1200] [--quick]
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

import torch

from sb_common.paths import base_artifacts_dir, data_dir, out_dir, ckpt_dir
from sb_common.tokenizer import (
    load_base_tokenizer, load_config,
    REQ_TOKEN, STORY_TOKEN, TALK_TOKEN, NOTALK_TOKEN, EOS_ID,
)
from sb_common.model import load_base_model

SEED = 1337
BLOCK = 256


def load_jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(line) for line in f]


def pack_example(tk, e: dict) -> tuple[list[int], list[int]] | None:
    """Return (ids, mask) padded to BLOCK; mask=1 where loss counts (story+eos)."""
    req_part = f"{REQ_TOKEN} {e['request']} {STORY_TOKEN}"
    req_ids = tk.encode(req_part)
    story_ids = tk.encode(" " + e["story"].strip()) + [EOS_ID]
    ids = req_ids + story_ids
    if len(ids) > BLOCK:
        return None  # drop over-long examples; stats reported by caller
    mask = [0] * len(req_ids) + [1] * len(story_ids)
    pad = BLOCK - len(ids)
    ids = ids + [EOS_ID] * pad
    mask = mask + [0] * pad
    return ids, mask


def make_batches(packed: list[tuple[list[int], list[int]]], batch_size: int, rng: random.Random):
    idx = list(range(len(packed)))
    rng.shuffle(idx)
    for i in range(0, len(idx) - batch_size + 1, batch_size):
        chunk = [packed[j] for j in idx[i : i + batch_size]]
        x = torch.tensor([c[0] for c in chunk], dtype=torch.long)
        m = torch.tensor([c[1] for c in chunk], dtype=torch.float32)
        # next-token prediction: input = ids[:-1], target = ids[1:]
        yield x[:, :-1], x[:, 1:], m[:, 1:]


@torch.no_grad()
def val_loss(model, packed_val, batch_size=32) -> float:
    model.eval()
    losses = []
    for i in range(0, len(packed_val), batch_size):
        chunk = packed_val[i : i + batch_size]
        x = torch.tensor([c[0] for c in chunk], dtype=torch.long)
        m = torch.tensor([c[1] for c in chunk], dtype=torch.float32)
        _, loss = model(x[:, :-1], targets=x[:, 1:], loss_mask=m[:, 1:])
        losses.append(float(loss))
    model.train()
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def sample_generation(model, tk, request: str, seed: int = SEED) -> str:
    model.eval()
    prompt = f"{REQ_TOKEN} {request} {STORY_TOKEN}"
    ids = tk.encode(prompt)
    g = torch.Generator().manual_seed(seed)
    y = model.generate(
        torch.tensor([ids], dtype=torch.long),
        max_new_tokens=180, temperature=0.8, top_k=40, eos_id=EOS_ID, generator=g,
    )
    model.train()
    return tk.decode(y[0].tolist()[len(ids):], skip_special_tokens=True).strip()


def build_dev_probe(val_examples: list[dict]) -> list[dict]:
    """12 requests for compliance-based checkpoint selection.

    NOT the gold set (that would be test-set leakage — the course teaches this):
    6 seen-name requests taken from val + 6 cross-paired unseen-name requests
    that do NOT appear in gold (gold pairs UNSEEN_NAMES[i] with animal i; the
    probe shifts the pairing by 3).
    """
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "dsb", Path(__file__).parent / "01_build_requests_dataset.py")
    dsb = ilu.module_from_spec(spec)
    spec.loader.exec_module(dsb)
    probe = []
    for e in val_examples[:6]:
        probe.append({"request": e["request"], "animal": e["animal"], "name": e["name"]})
    for i, animal in enumerate(dsb.TRAIN_ANIMALS[:6]):
        name = dsb.UNSEEN_NAMES[(i + 3) % len(dsb.UNSEEN_NAMES)]
        probe.append({
            "request": dsb.request_for(animal, name, i % 3),
            "animal": animal, "name": name,
        })
    return probe


@torch.no_grad()
def probe_compliance(model, tk, probe: list[dict]) -> float:
    import re
    model.eval()
    ok = 0
    for i, p in enumerate(probe):
        prompt = f"{REQ_TOKEN} {p['request']} {STORY_TOKEN}"
        ids = tk.encode(prompt)
        g = torch.Generator().manual_seed(4242 + i)
        y = model.generate(torch.tensor([ids], dtype=torch.long),
                           max_new_tokens=170, temperature=0.8, top_k=40,
                           eos_id=EOS_ID, generator=g)
        s = tk.decode(y[0].tolist()[len(ids):], skip_special_tokens=True)
        a_ok = len(re.findall(rf"\b{re.escape(p['animal'])}\b", s.lower())) >= 1
        n_ok = len(re.findall(rf"\b{re.escape(p['name'])}\b", s)) >= 2
        ok += a_ok and n_ok
    model.train()
    return ok / len(probe)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=36.0,
                    help="checkpoint state and exit after this many seconds (chunked CPU runs)")
    args = ap.parse_args()
    if args.quick:
        args.steps, args.eval_every = 60, 30

    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    base = base_artifacts_dir()
    cfg = load_config(base)
    tk = load_base_tokenizer(base)
    new_ids = tk.extend_with_specials()
    model = load_base_model(base, cfg)
    old_vocab = model.cfg.vocab_size
    model.resize_vocab(tk.vocab_size)

    train = load_jsonl(data_dir() / "requests_train.jsonl")
    val = load_jsonl(data_dir() / "requests_val.jsonl")
    packed_train, dropped = [], 0
    for e in train:
        p = pack_example(tk, e)
        if p is None:
            dropped += 1
        else:
            packed_train.append(p)
    packed_val = [p for e in val if (p := pack_example(tk, e)) is not None]

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))

    probe_request = "Tell me a story about a dog named Rex."  # the course's running request (canon)
    dev_probe = build_dev_probe(val)
    state_path = ckpt_dir() / "sft_state.pt"
    trace = {
        "seed": SEED, "steps": args.steps, "batch": args.batch,
        "lr": args.lr, "min_lr": args.min_lr, "warmup": args.warmup,
        "weight_decay": args.weight_decay,
        "special_token_ids": new_ids, "old_vocab": old_vocab, "new_vocab": tk.vocab_size,
        "train_examples": len(packed_train), "val_examples": len(packed_val),
        "dropped_too_long": dropped,
        "checkpoint_policy": "best dev-probe compliance (12 requests, NOT gold); tie -> earlier step",
        "train_loss": [], "val_loss": [], "lr_trace": [], "samples": [], "probe_curve": [],
    }

    model.train()
    best_val = float("inf")
    best_probe, best_probe_step, best_state = -1.0, -1, None
    step, prior_wall = 0, 0.0
    if args.resume and state_path.exists():
        snap = torch.load(state_path, weights_only=False)
        model.load_state_dict(snap["model"])
        opt.load_state_dict(snap["opt"])
        step = snap["step"]
        best_val = snap["best_val"]
        best_probe = snap.get("best_probe", -1.0)
        best_probe_step = snap.get("best_probe_step", -1)
        best_state = snap.get("best_state")
        trace = snap["trace"]
        prior_wall = snap.get("wall", 0.0)
        rng = random.Random(SEED + step)  # fresh shuffle stream per chunk
        print(f"resumed at step {step} (best probe {best_probe:.2f} @ {best_probe_step})")

    def snapshot(wall: float) -> None:
        torch.save({
            "model": model.state_dict(), "opt": opt.state_dict(),
            "step": step, "best_val": best_val, "best_probe": best_probe,
            "best_probe_step": best_probe_step, "best_state": best_state,
            "trace": trace, "wall": wall,
        }, state_path)

    t0 = time.perf_counter()
    while step < args.steps:
        for x, y, m in make_batches(packed_train, args.batch, rng):
            lr = lr_at(step)
            for grp in opt.param_groups:
                grp["lr"] = lr
            _, loss = model(x, targets=y, loss_mask=m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            trace["train_loss"].append(round(float(loss), 4))
            trace["lr_trace"].append(round(lr, 6))
            step += 1
            if step % args.eval_every == 0 or step == args.steps:
                vl = val_loss(model, packed_val)
                pc = probe_compliance(model, tk, dev_probe)
                trace["val_loss"].append({"step": step, "loss": round(vl, 4)})
                trace["probe_curve"].append({"step": step, "compliance": round(pc, 4)})
                sample = sample_generation(model, tk, probe_request)
                trace["samples"].append({"step": step, "request": probe_request, "story": sample[:400]})
                print(f"step {step} train {float(loss):.4f} val {vl:.4f} probe {pc:.2f} ({time.perf_counter()-t0:.0f}s)")
                best_val = min(best_val, vl)
                if pc > best_probe:
                    best_probe, best_probe_step = pc, step
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wall = time.perf_counter() - t0
            if wall > args.max_seconds and step < args.steps:
                snapshot(prior_wall + wall)
                print(json.dumps({"paused_at_step": step, "resume": True,
                                  "wall_so_far": round(prior_wall + wall, 1)}))
                return
            if step >= args.steps:
                break

    model.save_npz(ckpt_dir() / "sft_final.npz")  # last step, for the overfit demo
    if best_state is not None:
        model.load_state_dict(best_state)
    trace["best_val_loss"] = round(best_val, 4)
    trace["best_probe_compliance"] = round(best_probe, 4)
    trace["best_probe_step"] = best_probe_step
    trace["wall_seconds"] = round(prior_wall + time.perf_counter() - t0, 1)

    model.save_npz(ckpt_dir() / "sft_full.npz")  # the shipped checkpoint (probe-best)
    tk.export_extended_json(ckpt_dir() / "tokenizer_ext.json")
    with open(out_dir() / "sft_train_trace.json", "w") as f:
        json.dump(trace, f, indent=2)
    print(json.dumps({
        "best_val_loss": trace["best_val_loss"],
        "best_probe_compliance": trace["best_probe_compliance"],
        "best_probe_step": trace["best_probe_step"],
        "wall_seconds": trace["wall_seconds"],
        "train_examples": trace["train_examples"],
        "final_sample": trace["samples"][-1] if trace["samples"] else None,
    }, indent=2))


if __name__ == "__main__":
    main()
