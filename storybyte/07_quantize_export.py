"""Quantize (Module 7) + export every browser artifact for the course.

int8 quantization: per-tensor symmetric (scale = max|w| / 127), embeddings and
LayerNorm params kept float32 (standard practice at this scale - stated in the
course). Round-trip error and size measured and saved.

Exports to the website's public directory in the monorepo, or `web_artifacts/`
when this companion is cloned alone. Set FTLAB_WEB_DIR to override either path:
  sft_weights.npz          float32 full SFT (probe-best) model
  dpo_weights.npz          float16 DPO steps-25 (Goldilocks)
  dpo_overcooked.npz       float16 DPO steps-300 (the cautionary dial position)
  lora_adapters.npz        all rank adapters r1/r2/r4/r8 (+ wte_new_rows each)
  nano_kd.npz              float32 student
  nano_int8.npz            int8 student + scales
  nano_config.json         student architecture
  tokenizer_ext.json       extended tokenizer (authoritative encoder)
  eval_results.json        the full measured ladder (source: results/eval_ladder.json)
  train_traces.json        SFT/LoRA/DPO/KD curves for the course animations
  preference_pairs.json    20 pairs WITH policy/ref story logprobs (M5 margin widget)
  sample_generations.json  canned per-model outputs for the running request (fallbacks)

Run: python3 07_quantize_export.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from sb_common.paths import base_artifacts_dir, data_dir, ckpt_dir, out_dir, LDS_ROOT, REPO_ROOT
from sb_common.tokenizer import SBTokenizer, load_config, REQ_TOKEN, STORY_TOKEN, EOS_ID
from sb_common.model import StoryByte, SBConfig

RUNNING_REQUEST = "Tell me a story about a dog named Rex."  # canon


def resolve_web_dir() -> Path:
    override = os.environ.get("FTLAB_WEB_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if (LDS_ROOT / "src" / "app").exists():
        return LDS_ROOT / "public" / "learn" / "fine-tuning-llms"
    return REPO_ROOT / "web_artifacts"


WEB_DIR = resolve_web_dir()


def load_model(npz_path, cfg_json, tk, nano=False):
    if nano:
        cfg = SBConfig.from_json(json.load(open(ckpt_dir() / "nano_config.json")))
    else:
        cfg = SBConfig.from_json(cfg_json)
    cfg.vocab_size = tk.vocab_size
    m = StoryByte(cfg)
    m.load_npz(npz_path)
    m.eval()
    return m


def quantize_int8(npz_path: Path, out_path: Path) -> dict:
    z = np.load(str(npz_path))
    out, meta = {}, {}
    for k in z.files:
        w = z[k]
        if w.ndim >= 2 and "wte" not in k and "wpe" not in k:
            scale = float(np.abs(w).max()) / 127.0 or 1.0
            out[k + ".q"] = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
            out[k + ".scale"] = np.array([scale], dtype=np.float32)
            meta[k] = "int8"
        else:
            out[k] = w.astype(np.float32)
            meta[k] = "float32"
    np.savez(str(out_path), **out)
    return meta


def dequantize_to_model(int8_path: Path, cfg_json, tk) -> StoryByte:
    z = np.load(str(int8_path))
    tmp = {}
    for k in z.files:
        if k.endswith(".q"):
            base = k[:-2]
            tmp[base] = (z[k].astype(np.float32) * z[base + ".scale"][0])
        elif k.endswith(".scale"):
            continue
        else:
            tmp[k] = z[k]
    tmp_path = ckpt_dir() / "_deq_tmp.npz"
    np.savez(str(tmp_path), **tmp)
    return load_model(tmp_path, cfg_json, tk, nano=True)


@torch.no_grad()
def gen(model, tk, request, seed=1337, greedy=False):
    ids = tk.encode(f"{REQ_TOKEN} {request} {STORY_TOKEN}")
    g = torch.Generator().manual_seed(seed)
    y = model.generate(torch.tensor([ids]), max_new_tokens=190, temperature=0.8,
                       top_k=40, eos_id=EOS_ID, greedy=greedy, generator=g)
    return tk.decode(y[0].tolist()[len(ids):], skip_special_tokens=True).strip()


@torch.no_grad()
def story_logprob(model, tk, request, story):
    req_ids = tk.encode(f"{REQ_TOKEN} {request} {STORY_TOKEN}")
    ids = (req_ids + tk.encode(" " + story.strip()) + [EOS_ID])[:256]
    x = torch.tensor([ids[:-1]]); y = torch.tensor([ids[1:]])
    logits, _ = model(x)
    lp = torch.log_softmax(logits, -1)[0].gather(1, y[0][:, None]).squeeze(1)
    return float(lp[len(req_ids) - 1:].sum())


def main():
    t0 = time.perf_counter()
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    base_dir = base_artifacts_dir()
    cfg_json = load_config(base_dir)
    tk = SBTokenizer(ckpt_dir() / "tokenizer_ext.json")

    # ---- 1. int8 quantization of the student ----
    quantize_int8(ckpt_dir() / "nano_kd.npz", WEB_DIR / "nano_int8.npz")
    nano = load_model(ckpt_dir() / "nano_kd.npz", cfg_json, tk, nano=True)
    nano_q = dequantize_to_model(WEB_DIR / "nano_int8.npz", cfg_json, tk)
    # measure: max logit diff on the running request prompt + greedy agreement
    ids = tk.encode(f"{REQ_TOKEN} {RUNNING_REQUEST} {STORY_TOKEN}")
    x = torch.tensor([ids])
    l1, _ = nano(x); l2, _ = nano_q(x)
    q_report = {
        "scope": "offline PyTorch fp32 nano vs dequantized int8 nano on the running request",
        "method": "per-tensor symmetric int8 (matmul weights); embeddings+LN float32",
        "max_logit_diff": round(float((l1 - l2).abs().max()), 5),
        "greedy_50tok_agree": gen(nano, tk, RUNNING_REQUEST, greedy=True)[:200]
                              == gen(nano_q, tk, RUNNING_REQUEST, greedy=True)[:200],
        "fp32_bytes": (ckpt_dir() / "nano_kd.npz").stat().st_size,
        "int8_bytes": (WEB_DIR / "nano_int8.npz").stat().st_size,
    }

    # ---- 2. copy model artifacts ----
    import shutil
    shutil.copy(ckpt_dir() / "sft_full.npz", WEB_DIR / "sft_weights.npz")
    shutil.copy(ckpt_dir() / "nano_kd.npz", WEB_DIR / "nano_kd.npz")
    shutil.copy(ckpt_dir() / "nano_config.json", WEB_DIR / "nano_config.json")
    shutil.copy(ckpt_dir() / "tokenizer_ext.json", WEB_DIR / "tokenizer_ext.json")
    for name, src in [("dpo_weights.npz", "dpo.npz"), ("dpo_overcooked.npz", "dpo_overcooked.npz")]:
        z = np.load(str(ckpt_dir() / src))
        np.savez(str(WEB_DIR / name), **{k: z[k].astype(np.float16) for k in z.files})

    # all LoRA adapters into ONE file, namespaced by rank
    lora_all = {}
    for r in (1, 2, 4, 8):
        z = np.load(str(ckpt_dir() / f"lora_r{r}_adapter.npz"))
        for k in z.files:
            lora_all[f"r{r}.{k}"] = z[k]
    np.savez(str(WEB_DIR / "lora_adapters.npz"), **lora_all)

    # ---- 3. eval ladder + traces ----
    shutil.copy(out_dir() / "eval_ladder.json", WEB_DIR / "eval_results.json")
    traces = {
        "sft": json.load(open(out_dir() / "sft_train_trace.json")),
        "dpo": json.load(open(out_dir() / "dpo_train_trace.json")),
        "kd": json.load(open(out_dir() / "kd_train_trace_kd.json")),
        "kd_scratch": json.load(open(out_dir() / "kd_train_trace_scratch.json")),
        "lora": {f"r{r}": json.load(open(out_dir() / f"lora_r{r}_trace.json"))
                 for r in (1, 2, 4, 8)},
    }
    # slim the big arrays (course scrubbers need every 5th point at most)
    for key in ("sft", "dpo", "kd", "kd_scratch"):
        tr = traces[key]
        for arr in ("train_loss", "loss", "margin", "accuracy", "lr_trace"):
            if arr in tr and isinstance(tr[arr], list) and len(tr[arr]) > 400:
                tr[arr] = tr[arr][::3]
    for r in traces["lora"].values():
        if len(r.get("train_loss", [])) > 400:
            r["train_loss"] = r["train_loss"][::3]
    with open(WEB_DIR / "train_traces.json", "w") as f:
        json.dump(traces, f)

    # ---- 4. preference pairs with logprobs for the M5 margin widget ----
    sft = load_model(ckpt_dir() / "sft_full.npz", cfg_json, tk)
    dpo = load_model(ckpt_dir() / "dpo.npz", cfg_json, tk)
    pairs = [json.loads(l) for l in open(data_dir() / "preference_pairs.jsonl")][:20]
    enriched = []
    for p in pairs:
        enriched.append({
            **p,
            "logp": {
                "policy_chosen": round(story_logprob(dpo, tk, p["request"], p["chosen"]), 3),
                "policy_rejected": round(story_logprob(dpo, tk, p["request"], p["rejected"]), 3),
                "ref_chosen": round(story_logprob(sft, tk, p["request"], p["chosen"]), 3),
                "ref_rejected": round(story_logprob(sft, tk, p["request"], p["rejected"]), 3),
            },
        })
    with open(WEB_DIR / "preference_pairs.json", "w") as f:
        json.dump({"note": "logp = summed story-token logprobs; policy=dpo(steps25), ref=sft",
                   "pairs": enriched}, f)

    # ---- 5. canned sample generations (honest fallbacks + before/after demos) ----
    base_model = load_model(base_dir / "storybyte_weights.npz", cfg_json,
                            SBTokenizer(base_dir / "storybyte_tokenizer_hf.json"))
    base_tk = SBTokenizer(base_dir / "storybyte_tokenizer_hf.json")

    @torch.no_grad()
    def gen_plain(model, tkz, prompt, seed=1337):
        ids = tkz.encode(prompt)
        g = torch.Generator().manual_seed(seed)
        y = model.generate(torch.tensor([ids]), max_new_tokens=190, temperature=0.8,
                           top_k=40, eos_id=EOS_ID, generator=g)
        return tkz.decode(y[0].tolist()[len(ids):], skip_special_tokens=True).strip()

    lora4 = load_model(ckpt_dir() / "lora_r4_merged.npz", cfg_json, tk)
    samples = {
        "running_request": RUNNING_REQUEST,
        "sampler": {"temperature": 0.8, "top_k": 40, "seed": 1337},
        "base_plain_prompt": gen_plain(base_model, base_tk, RUNNING_REQUEST + "\n"),
        "sft": gen(sft, tk, RUNNING_REQUEST),
        "lora_r4": gen(lora4, tk, RUNNING_REQUEST),
        "dpo": gen(dpo, tk, RUNNING_REQUEST),
        "dpo_overcooked": gen(load_model(ckpt_dir() / "dpo_overcooked.npz", cfg_json, tk),
                              tk, RUNNING_REQUEST),
        "nano_kd": gen(nano, tk, RUNNING_REQUEST),
        "nano_int8": gen(nano_q, tk, RUNNING_REQUEST),
    }
    with open(WEB_DIR / "sample_generations.json", "w") as f:
        json.dump(samples, f, indent=2)

    report = {
        "quantization": q_report,
        "web_dir_files": {p.name: p.stat().st_size for p in sorted(WEB_DIR.iterdir())},
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }
    with open(out_dir() / "export_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
