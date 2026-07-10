"""Build the request->story dataset (Module 2's pipeline, for real).

The control schema (decided by measurement - see results/sentiment_probe.json):
  - CHARACTER: which animal the story is about (8 train animals + 2 eval-only)
  - DIALOGUE:  "with talking" vs "with no talking" (natural base rates ~56%/44%,
               so BOTH classes have abundant real data; checked via quote marks)

Why not "happy vs sad endings"? We tried (results/sentiment_probe.json): the base
model ends sad only ~4-8% of the time even when seeded sad, and marker-lexicon
labels misfire on negations ("not sad anymore"). You cannot SFT what your data
distribution barely contains - that failed attempt is taught honestly in Module 2.
Length control fails the same way (natural spread is only ~82-164 words).

Strategy (synthetic data, honestly labeled):
  1. Seed base StoryByte with "Once upon a time, there was a little {animal}".
  2. Keep stories that end naturally (eos), are >=30 words, feature the animal.
  3. Label DIALOGUE by quote-mark presence (a string check, not judgment).
  4. Wrap each story in a request from the canon templates.

Sharded execution (CPU-friendly, resumable):
  python3 01_build_requests_dataset.py --animal dog          # one shard (~30s)
  ... (one call per animal) ...
  python3 01_build_requests_dataset.py --assemble            # merge + split + gold

Outputs: data/raw_shard_{animal}.json, then requests_train.jsonl / requests_val.jsonl /
gold_requests.json / forgetting_holdout.json + results/dataset_stats.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SEED = 1337

# ---- entity-registry canon (course-wide; do not drift) ----
TRAIN_ANIMALS = ["dog", "cat", "bird", "bunny", "bear", "fish", "duck", "frog"]
UNSEEN_ANIMALS = ["fox", "mouse"]  # appear ONLY in the gold eval set
# v4 task (measured - the one that works): CHARACTER + NAME control.
# Copying a requested name into a generated story is an induction-style skill
# transformers have natively even at 1M params; per-request STYLE switching
# (dialogue on/off) proved unlearnable at this scale across v1-v3 - see
# results/eval_ladder.json - and that failure is taught honestly in the course.
# Dialogue returns in Module 5 as a GLOBAL preference target for DPO.
TEMPLATES = [
    "Tell me a story about a {animal} named {name}.",
    "Write a story about a {animal} named {name}.",
    "I want a story about a {animal} named {name}.",
]

# canonical seen-name per animal (top extracted names, entity registry canon)
SEEN_NAME = {
    "dog": "Max", "cat": "Kitty", "bird": "Tom", "bunny": "Ben",
    "bear": "Bob", "fish": "Fin", "duck": "Ducky", "frog": "Fred",
    "fox": "Spot", "mouse": "Tim",
}
# Names reserved for the gold-set requested-name condition. Future dataset builds
# reject a row when any of these names appears anywhere in its story, including as
# a secondary character. The frozen shipped data predates this whole-row guard;
# its documented Bella exception is intentionally preserved for reproducibility.
UNSEEN_NAMES = ["Rex", "Bella", "Milo", "Suzy", "Toby", "Nina", "Gus", "Lola", "Zack", "Pia"]
PROTECTED_NAME_PATTERNS = {
    name: re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
    for name in UNSEEN_NAMES
}

NAME_PATTERN = re.compile(r"named ([A-Z][a-z]+)")

HAPPY_MARKERS = [
    "happy", "happily", "smiled", "smile", "laughed", "glad",
    "hugged", "hug", "friends", "fun", "loved", "love", "joy",
]
SAD_MARKERS = [
    "sad", "sadly", "cried", "cry", "crying", "tears", "sorry",
    "hurt", "lost", "alone", "scared",
]

SEED_PROMPTS = [
    "Once upon a time, there was a little {animal}",
    "Once upon a time, there was a {animal}",
]


def request_for(animal: str, name: str, template_id: int) -> str:
    return TEMPLATES[template_id].format(animal=animal, name=name)


def extract_name(story: str) -> str | None:
    m = NAME_PATTERN.search(story)
    return m.group(1) if m else None


def name_present(story: str, name: str, min_count: int = 2) -> bool:
    return len(re.findall(rf"\b{re.escape(name)}\b", story)) >= min_count


def protected_name_counts(text: str) -> dict[str, int]:
    """Return every protected-name occurrence, matched as a whole word."""
    return {
        name: len(pattern.findall(text))
        for name, pattern in PROTECTED_NAME_PATTERNS.items()
        if pattern.search(text)
    }


def assert_no_protected_names(rows: list[dict]) -> None:
    """Fail closed if a future train/validation row contains a protected name."""
    leaks = []
    for i, row in enumerate(rows):
        for field in ("request", "story", "name"):
            hits = protected_name_counts(str(row.get(field, "")))
            if hits:
                leaks.append({"row": i, "field": field, "hits": hits})
    if leaks:
        raise AssertionError(f"protected-name leak in assembled rows: {leaks[:10]}")


def has_dialogue(story: str) -> bool:
    return '"' in story


def has_moral(story: str) -> bool:
    return "moral of the story" in story.lower()


def last_sentences(text: str, n: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[-n:]).lower()


def count_markers(text: str, markers: list[str]) -> int:
    return sum(len(re.findall(rf"\b{re.escape(m)}\b", text)) for m in markers)


def label_ending(story: str) -> str:
    """Heuristic sentiment label - kept for the Module 2 honest-lesson stats."""
    tail = last_sentences(story, 2)
    h = count_markers(tail, HAPPY_MARKERS)
    s = count_markers(tail, SAD_MARKERS)
    if h > 0 and s == 0:
        return "happy"
    if s > 0 and h == 0:
        return "sad"
    return "mixed"


def animal_present(story: str, animal: str, min_count: int = 2) -> bool:
    return len(re.findall(rf"\b{re.escape(animal)}\b", story.lower())) >= min_count


def generate_shard(animal: str, per_animal: int, batch: int, max_new: int, round_id: int = 1) -> None:
    import torch
    from sb_common.paths import base_artifacts_dir, data_dir
    from sb_common.tokenizer import load_base_tokenizer, load_config
    from sb_common.model import load_base_model

    torch.manual_seed(SEED)
    base = base_artifacts_dir()
    tk = load_base_tokenizer(base)
    model = load_base_model(base, load_config(base))
    # per-animal, per-round generator seed => shards are order-independent and reproducible
    gen = torch.Generator().manual_seed(SEED + sum(ord(c) for c in animal) + 7919 * (round_id - 1))

    stories: list[dict] = []
    t0 = time.perf_counter()
    per_seed = per_animal // len(SEED_PROMPTS)
    for seed_tpl in SEED_PROMPTS:
        prompt = seed_tpl.format(animal=animal)
        ids = tk.encode(prompt)
        done = 0
        while done < per_seed:
            b = min(batch, per_seed - done)
            x = torch.tensor([ids] * b, dtype=torch.long)
            y = model.generate(x, max_new_tokens=max_new, temperature=0.8,
                               top_k=40, eos_id=0, generator=gen)
            for row in y:
                toks = row.tolist()
                ended = 0 in toks[len(ids):]
                text = tk.decode(toks, skip_special_tokens=True).strip()
                stories.append({"animal": animal, "text": text,
                                "ended_naturally": bool(ended)})
            done += b
    suffix = "" if round_id == 1 else f"_r{round_id}"
    out = data_dir() / f"raw_shard_{animal}{suffix}.json"
    with open(out, "w") as f:
        json.dump({"animal": animal, "seconds": round(time.perf_counter() - t0, 1),
                   "stories": stories}, f)
    print(json.dumps({"animal": animal, "round": round_id, "generated": len(stories),
                      "seconds": round(time.perf_counter() - t0, 1)}))


def assemble() -> None:
    from sb_common.paths import data_dir, out_dir

    rng = random.Random(SEED)
    dd = data_dir()
    all_stories: list[dict] = []
    gen_seconds = 0.0
    shard_files = sorted(dd.glob("raw_shard_*.json"))
    have_animals = set()
    for p in shard_files:
        shard = json.load(open(p))
        gen_seconds += shard["seconds"]
        all_stories.extend(shard["stories"])
        have_animals.add(shard["animal"])
    missing = set(TRAIN_ANIMALS) - have_animals
    if missing:
        raise SystemExit(f"missing shards for: {sorted(missing)}")

    kept: list[dict] = []
    no_name_pool: list[str] = []  # fluent stories with no named character -> forgetting holdout
    drop = {"no_eos": 0, "too_short": 0, "no_animal": 0, "no_name": 0, "unseen_name_leak": 0}
    for s in all_stories:
        if not s["ended_naturally"]:
            drop["no_eos"] += 1
            continue
        words = len(s["text"].split())
        if words < 30:
            drop["too_short"] += 1
            continue
        # >=1: once a story names its character, it refers to it by NAME.
        # demanding repeated animal mentions wrongly rejects compliant stories.
        if not animal_present(s["text"], s["animal"], min_count=1):
            drop["no_animal"] += 1
            continue
        # Keep the historical stats key for downstream compatibility, but make
        # its future meaning strict: any protected name anywhere in the story.
        if protected_name_counts(s["text"]):
            drop["unseen_name_leak"] += 1
            continue
        name = extract_name(s["text"])
        if name is None or not name_present(s["text"], name):
            drop["no_name"] += 1
            no_name_pool.append(s["text"])
            continue
        kept.append({
            "animal": s["animal"],
            "name": name,
            "story": s["text"],
            "word_count": words,
            "has_dialogue": has_dialogue(s["text"]),
            "has_moral": has_moral(s["text"]),
            "ending_label": label_ending(s["text"]),
        })

    rng.shuffle(kept)
    balanced = kept  # the name task needs no class balancing

    # v4 task (measured redesign): character + NAME control. Copying a requested
    # name is an induction-style skill the model has natively; per-request STYLE
    # switching (dialogue on/off) proved unlearnable at 1.09M params across
    # v1-v3 (see results/eval_ladder.json). All template phrasings train.
    examples = []
    for i, k in enumerate(balanced):
        tpl_id = i % len(TEMPLATES)
        examples.append({
            "request": request_for(k["animal"], k["name"], tpl_id),
            "story": k["story"],
            "animal": k["animal"],
            "name": k["name"],
            "template_id": tpl_id,
            "word_count": k["word_count"],
            "has_dialogue": k["has_dialogue"],
            "has_moral": k["has_moral"],
            "ending_label": k["ending_label"],
        })

    n_val = max(16, len(examples) // 10)
    val, train = examples[:n_val], examples[n_val:]
    assert_no_protected_names(train)
    assert_no_protected_names(val)
    with open(dd / "requests_train.jsonl", "w") as f:
        for e in train:
            f.write(json.dumps(e) + "\n")
    with open(dd / "requests_val.jsonl", "w") as f:
        for e in val:
            f.write(json.dumps(e) + "\n")

    # ---- forgetting holdout: 64 fluent base-model stories that are NOT in
    # training (they have no extractable named character, so they can't be) ----
    rng.shuffle(no_name_pool)
    holdout = no_name_pool[:64]
    with open(dd / "forgetting_holdout.json", "w") as f:
        json.dump(holdout, f)

    # ---- gold eval set: 10 animals x {seen,unseen} name x 3 templates = 60 ----
    train_names = {e["name"] for e in examples}
    gold = []
    for ai, animal in enumerate(TRAIN_ANIMALS + UNSEEN_ANIMALS):
        for cond in ["seen_name", "unseen_name"]:
            name = SEEN_NAME[animal] if cond == "seen_name" else UNSEEN_NAMES[ai]
            assert cond == "seen_name" or name not in train_names, f"gold leak: {name}"
            for tpl_id in range(len(TEMPLATES)):
                gold.append({
                    "request": request_for(animal, name, tpl_id),
                    "animal": animal,
                    "name": name,
                    "name_cond": cond,
                    "template_id": tpl_id,
                    "subset": "unseen_animal" if animal in UNSEEN_ANIMALS else "seen",
                })
    with open(dd / "gold_requests.json", "w") as f:
        json.dump(gold, f, indent=2)

    wc = sorted(k["word_count"] for k in kept)
    stats = {
        "seed": SEED,
        "generated_stories": len(all_stories),
        "generation_seconds": round(gen_seconds, 1),
        "dropped": drop,
        "kept_after_filter": len(kept),
        "after_balance": len(balanced),
        "train": len(train),
        "val": len(val),
        "gold_requests": len(gold),
        "forgetting_holdout": len(holdout),
        "dialogue_base_rate": round(
            sum(1 for k in kept if k["has_dialogue"]) / max(len(kept), 1), 3),
        "moral_base_rate": round(
            sum(1 for k in kept if k["has_moral"]) / max(len(kept), 1), 3),
        "ending_label_dist": {
            lab: sum(1 for k in kept if k["ending_label"] == lab)
            for lab in ["happy", "sad", "mixed"]
        },
        "word_count": {"min": wc[0], "p25": wc[len(wc) // 4],
                       "median": wc[len(wc) // 2],
                       "p75": wc[3 * len(wc) // 4], "max": wc[-1]},
        "distinct_names": len({k["name"] for k in kept}),
        "top_names": sorted(
            {n: sum(1 for k in kept if k["name"] == n) for n in {k["name"] for k in kept}}.items(),
            key=lambda kv: -kv[1])[:10],
        "per_animal_counts": {
            a: sum(1 for e in examples if e["animal"] == a) for a in TRAIN_ANIMALS
        },
    }
    with open(out_dir() / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--animal", type=str, default=None)
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--per-animal", type=int, default=80)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--max-new", type=int, default=210)
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()

    if args.assemble:
        assemble()
    elif args.animal:
        assert args.animal in TRAIN_ANIMALS, f"unknown animal {args.animal}"
        generate_shard(args.animal, args.per_animal, args.batch, args.max_new, args.round)
    else:
        raise SystemExit("pass --animal <name> for a shard, or --assemble")


if __name__ == "__main__":
    main()
