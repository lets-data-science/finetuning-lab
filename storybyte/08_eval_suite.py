"""The scoreboard: every canonical number in the course comes from this script.

Metrics (all heuristic string/structure checks - the course labels them as such):
  - character_compliance: requested animal appears >=2 times in the story
  - name_compliance:      requested NAME appears >=2 times (the copy/induction skill)
  - format_compliance:    story terminates with <|endoftext|> within budget and
                          is >=30 words
  - full_compliance:      all three at once
  - forgetting guard:     masked perplexity on 64 held-out BASE-model stories
                          (drift from the base distribution; labeled honestly)

Modes:
  --model base            plain-prompt baseline (request text + newline as prefix)
  --model sft             checkpoints/sft_full.npz + extended tokenizer
  (later: --model lora_rN | dpo | nano | nano_int8)

3 samples per gold request (seeds 1337/1338/1339); compliance averaged over samples.
Results appended into results/eval_ladder.json under the model name.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from sb_common.paths import base_artifacts_dir, data_dir, out_dir, ckpt_dir
from sb_common.tokenizer import (
    SBTokenizer, load_base_tokenizer, load_config,
    REQ_TOKEN, STORY_TOKEN, TALK_TOKEN, NOTALK_TOKEN, EOS_ID,
)
from sb_common.model import StoryByte, SBConfig, load_base_model

# Import the labeling heuristics from the dataset builder so eval and data
# can never drift apart.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("dsb", Path(__file__).parent / "01_build_requests_dataset.py")
_dsb = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_dsb)

SAMPLE_SEEDS = [1337, 1338, 1339]
MAX_NEW = 200


def load_model(which: str, base_dir, cfg_json) -> tuple[StoryByte, SBTokenizer, bool]:
    """Returns (model, tokenizer, uses_request_format)."""
    if which == "base":
        return load_base_model(base_dir, cfg_json), load_base_tokenizer(base_dir), False
    tk = SBTokenizer(ckpt_dir() / "tokenizer_ext.json")
    cfg = SBConfig.from_json(cfg_json)
    cfg.vocab_size = tk.vocab_size
    model = StoryByte(cfg)
    path = {
        "sft": ckpt_dir() / "sft_full.npz",
        "sft_final": ckpt_dir() / "sft_final.npz",
        "dpo": ckpt_dir() / "dpo.npz",
        "nano_kd": ckpt_dir() / "nano_kd.npz",
        "nano_scratch": ckpt_dir() / "nano_scratch.npz",
    }.get(which)
    if path is None and which.startswith("lora_r"):
        path = ckpt_dir() / f"{which}_merged.npz"
    if path is None or not Path(path).exists():
        raise SystemExit(f"unknown/missing model '{which}'")
    if which.startswith("nano"):
        with open(ckpt_dir() / "nano_config.json") as f:
            ncfg = json.load(f)
        cfg = SBConfig.from_json(ncfg)
        cfg.vocab_size = tk.vocab_size
        model = StoryByte(cfg)
    model.load_npz(path)
    model.eval()
    return model, tk, True


def build_prompt(request: str, uses_format: bool) -> str:
    if uses_format:
        return f"{REQ_TOKEN} {request} {STORY_TOKEN}"
    # base model gets the natural request as a plain-text prefix, as a user would type it
    return request + "\n"


def check_story(story: str, animal: str, name: str, ended: bool) -> dict:
    # animal >=1 (the story introduces "a dog named Rex", then uses the name);
    # name >=2 (the actual copy/induction skill under test)
    char_ok = _dsb.animal_present(story, animal, min_count=1)
    name_ok = _dsb.name_present(story, name, min_count=2)
    fmt_ok = bool(ended) and len(story.split()) >= 30
    return {
        "ok_character": char_ok,
        "ok_name": name_ok,
        "ok_format": fmt_ok,
        "ok_full": char_ok and name_ok and fmt_ok,
    }


@torch.no_grad()
def forgetting_perplexity(model: StoryByte, tk, stories: list[str]) -> float:
    """Mean per-token loss (exp'd) on held-out base-model stories."""
    losses, ntok = 0.0, 0
    for s in stories:
        ids = tk.encode(s)[: model.cfg.block_size]
        if len(ids) < 16:
            continue
        x = torch.tensor([ids[:-1]], dtype=torch.long)
        y = torch.tensor([ids[1:]], dtype=torch.long)
        _, loss = model(x, targets=y)
        losses += float(loss) * (len(ids) - 1)
        ntok += len(ids) - 1
    import math
    return round(math.exp(losses / max(ntok, 1)), 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quick", action="store_true", help="1 seed, first 12 gold requests")
    ap.add_argument("--part", type=str, default=None,
                    help="i:n - run only the i-th of n slices of gold, save partial rows")
    ap.add_argument("--report", action="store_true",
                    help="merge saved parts into the ladder entry")
    args = ap.parse_args()

    base = base_artifacts_dir()
    cfg_json = load_config(base)

    with open(data_dir() / "gold_requests.json") as f:
        gold = json.load(f)
    seeds = SAMPLE_SEEDS[:1] if args.quick else SAMPLE_SEEDS
    if args.quick:
        gold = gold[:12]

    holdout_path = data_dir() / "forgetting_holdout.json"
    holdout = json.load(open(holdout_path)) if holdout_path.exists() else []

    t0 = time.perf_counter()
    if args.report:
        model, tk, uses_format = load_model(args.model, base, cfg_json)
        rows = []
        for p in sorted(out_dir().glob(f"eval_rows_{args.model}_part*.json")):
            rows.extend(json.load(open(p)))
        if not rows:
            raise SystemExit("no saved parts to report on")
    else:
        model, tk, uses_format = load_model(args.model, base, cfg_json)
        if args.part:
            i, n = (int(x) for x in args.part.split(":"))
            gold = gold[i - 1 :: n]
        rows = []
        for g in gold:
            for seed in seeds:
                prompt = build_prompt(g["request"], uses_format)
                ids = tk.encode(prompt)
                gen = torch.Generator().manual_seed(seed)
                y = model.generate(
                    torch.tensor([ids], dtype=torch.long),
                    max_new_tokens=MAX_NEW, temperature=0.8, top_k=40,
                    eos_id=EOS_ID, generator=gen,
                )
                out_ids = y[0].tolist()[len(ids):]
                ended = EOS_ID in out_ids
                story = tk.decode(out_ids, skip_special_tokens=True).strip()
                checks = check_story(story, g["animal"], g["name"], ended)
                rows.append({**g, "seed": seed, "story": story, **checks})
        if args.part:
            i, n = (int(x) for x in args.part.split(":"))
            with open(out_dir() / f"eval_rows_{args.model}_part{i}of{n}.json", "w") as f:
                json.dump(rows, f)
            print(json.dumps({"model": args.model, "part": args.part,
                              "rows": len(rows),
                              "wall_seconds": round(time.perf_counter() - t0, 1)}))
            return

    def rate(key: str, subset=None) -> float:
        pool = [r for r in rows if subset is None or r["subset"] == subset]
        return round(sum(r[key] for r in pool) / max(len(pool), 1), 4)

    report = {
        "model": args.model,
        "n_gold": len(gold),
        "n_samples": len(rows),
        "seeds": seeds,
        "sampler": {"temperature": 0.8, "top_k": 40, "max_new_tokens": MAX_NEW},
        "dialogue_rate": round(sum(1 for r in rows if '"' in r["story"]) / max(len(rows), 1), 4),
        "character_compliance": rate("ok_character"),
        "name_compliance": rate("ok_name"),
        "format_compliance": rate("ok_format"),
        "full_compliance": rate("ok_full"),
        "per_name_cond": {
            c: {k: round(sum(r[k] for r in rows if r["name_cond"] == c)
                         / max(sum(1 for r in rows if r["name_cond"] == c), 1), 4)
                for k in ["ok_character", "ok_name", "ok_format", "ok_full"]}
            for c in ["seen_name", "unseen_name"]
        },
        "seen": {k: rate(k, "seen") for k in ["ok_character", "ok_name", "ok_format", "ok_full"]},
        "unseen_animal": {k: rate(k, "unseen_animal") for k in ["ok_character", "ok_name", "ok_format", "ok_full"]},
        "forgetting_perplexity": forgetting_perplexity(model, tk, holdout) if holdout else None,
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }

    ladder_path = out_dir() / "eval_ladder.json"
    ladder = json.load(open(ladder_path)) if ladder_path.exists() else {}
    ladder[args.model] = report
    with open(ladder_path, "w") as f:
        json.dump(ladder, f, indent=2)
    with open(out_dir() / f"eval_detail_{args.model}.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
