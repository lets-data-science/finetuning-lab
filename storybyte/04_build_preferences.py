"""Build DPO preference pairs: users prefer stories WITH dialogue (Module 5).

Why dialogue is the PREFERENCE target and not an SFT request field: per-request
dialogue switching proved unlearnable at 1.09M params (the v1-v3 ablation in
results/eval_ladder.json). A GLOBAL style preference needs no per-request
conditioning - exactly what preference tuning is for.

Pipeline: for each of N training requests, sample K completions from the SFT
model; if both a quoted and an unquoted completion appear, form (chosen=quoted,
rejected=unquoted). Preference labels are a RULE (quote-mark presence), stated
honestly - the course explains that real labels come from humans/rubrics.

Sharded: --shard i:n  then  --assemble --parts n
Output: data/preference_pairs.jsonl  {"request","chosen","rejected"}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from sb_common.paths import base_artifacts_dir, data_dir, ckpt_dir, out_dir
from sb_common.tokenizer import SBTokenizer, load_config, REQ_TOKEN, STORY_TOKEN, EOS_ID
from sb_common.model import StoryByte, SBConfig

SEED = 2024
K = 4
MAX_NEW = 190
N_REQUESTS = 168  # multiple of shards; ~1/3 of train requests


def load_sft(cfg_json):
    tk = SBTokenizer(ckpt_dir() / "tokenizer_ext.json")
    cfg = SBConfig.from_json(cfg_json)
    cfg.vocab_size = tk.vocab_size
    m = StoryByte(cfg)
    m.load_npz(ckpt_dir() / "sft_full.npz")
    m.eval()
    return m, tk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=str, default=None, help="i:n")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--parts", type=int, default=None,
                    help="exact shard count to assemble; required with --assemble")
    args = ap.parse_args()

    dd = data_dir()
    if args.assemble:
        if not args.parts or args.parts < 1:
            ap.error("--assemble requires --parts N")
        paths = [dd / f"pref_shard_{i}of{args.parts}.json"
                 for i in range(1, args.parts + 1)]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            ap.error(f"missing preference shards: {', '.join(missing)}")
        pairs = []
        for p in paths:
            pairs.extend(json.load(open(p)))
        with open(dd / "preference_pairs.jsonl", "w") as f:
            for pr in pairs:
                f.write(json.dumps(pr) + "\n")
        stats = {
            "pairs": len(pairs),
            "sampler": {"K": K, "temperature": 0.9, "top_k": 40, "seed": SEED},
            "rule": "chosen = contains a double-quote character; rejected = does not",
        }
        with open(out_dir() / "preference_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(json.dumps(stats))
        return

    if not args.shard:
        ap.error("pass --shard i:n or --assemble --parts N")
    i, n = (int(x) for x in args.shard.split(":"))
    if n < 1 or i < 1 or i > n:
        ap.error("--shard must be i:n with 1 <= i <= n")

    cfg_json = load_config(base_artifacts_dir())
    model, tk = load_sft(cfg_json)

    reqs = [json.loads(l)["request"] for l in open(dd / "requests_train.jsonl")]
    reqs = reqs[:N_REQUESTS][i - 1 :: n]

    pairs = []
    t0 = time.perf_counter()
    gen = torch.Generator().manual_seed(SEED + i)
    for req in reqs:
        prompt = f"{REQ_TOKEN} {req} {STORY_TOKEN}"
        ids = tk.encode(prompt)
        x = torch.tensor([ids] * K, dtype=torch.long)
        y = model.generate(x, max_new_tokens=MAX_NEW, temperature=0.9, top_k=40,
                           eos_id=EOS_ID, generator=gen)
        quoted, unquoted = [], []
        for row in y:
            out_ids = row.tolist()[len(ids):]
            if EOS_ID not in out_ids:
                continue  # only naturally-ended stories become pairs
            s = tk.decode(out_ids, skip_special_tokens=True).strip()
            (quoted if '"' in s else unquoted).append(s)
        if quoted and unquoted:
            pairs.append({"request": req, "chosen": quoted[0], "rejected": unquoted[0]})
    with open(dd / f"pref_shard_{i}of{n}.json", "w") as f:
        json.dump(pairs, f)
    print(json.dumps({"shard": args.shard, "requests": len(reqs), "pairs": len(pairs),
                      "wall_seconds": round(time.perf_counter() - t0, 1)}))


if __name__ == "__main__":
    main()
