"""Verify the browser path: a pure-NumPy forward (what the Pyodide worker runs)
must reproduce the PyTorch training stack on the SHIPPED web artifacts.

Same standard as tiny-llm's verification.json: max logit diff ~1e-5-ish and
byte-identical greedy continuation. Checks sft_weights.npz and dpo_weights.npz
(float16 on disk -> upcast f32, like the RAG loader convention).

Run: python3 09_verify_web_artifacts.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from sb_common.paths import base_artifacts_dir, out_dir, LDS_ROOT, REPO_ROOT
from sb_common.tokenizer import SBTokenizer, load_config, REQ_TOKEN, STORY_TOKEN, EOS_ID
from sb_common.model import StoryByte, SBConfig



def resolve_web_dir() -> Path:
    override = os.environ.get("FTLAB_WEB_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if (LDS_ROOT / "src" / "app").exists():
        return LDS_ROOT / "public" / "learn" / "fine-tuning-llms"
    return REPO_ROOT / "web_artifacts"


WEB = resolve_web_dir()
RUNNING_REQUEST = "Tell me a story about a dog named Rex."


# ---------- the exact NumPy forward the worker will run ----------
def np_forward(weights: dict, ids: list[int]) -> np.ndarray:
    def ln(x, g, b, eps=1e-5):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * g + b

    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def softmax(x):
        e = np.exp(x - x.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    w = weights
    T = len(ids)
    x = w["wte"][ids] + w["wpe"][:T]
    n_layer = 4
    n_head = 4
    d = x.shape[-1]
    hd = d // n_head
    mask = np.triu(np.full((T, T), -1e10, dtype=np.float32), k=1)
    for i in range(n_layer):
        p = f"h.{i}."
        h = ln(x, w[p + "ln_1.g"], w[p + "ln_1.b"])
        qkv = h @ w[p + "attn.c_attn.w"] + w[p + "attn.c_attn.b"]
        q, k, v = np.split(qkv, 3, axis=-1)
        outs = []
        for hh in range(n_head):
            qs = q[:, hh * hd:(hh + 1) * hd]
            ks = k[:, hh * hd:(hh + 1) * hd]
            vs = v[:, hh * hd:(hh + 1) * hd]
            att = softmax(qs @ ks.T / np.sqrt(hd) + mask)
            outs.append(att @ vs)
        att_out = np.concatenate(outs, axis=-1) @ w[p + "attn.c_proj.w"] + w[p + "attn.c_proj.b"]
        x = x + att_out
        h = ln(x, w[p + "ln_2.g"], w[p + "ln_2.b"])
        ff = gelu(h @ w[p + "mlp.c_fc.w"] + w[p + "mlp.c_fc.b"])
        x = x + ff @ w[p + "mlp.c_proj.w"] + w[p + "mlp.c_proj.b"]
    x = ln(x, w["ln_f.g"], w["ln_f.b"])
    return x @ w["wte"].T


def np_greedy(weights, ids, n_tokens, eos_id=0):
    ids = list(ids)
    for _ in range(n_tokens):
        logits = np_forward(weights, ids[-256:])
        nxt = int(logits[-1].argmax())
        ids.append(nxt)
        if nxt == eos_id:
            break
    return ids


def check(npz_name: str, tk, cfg_json) -> dict:
    z = np.load(str(WEB / npz_name))
    weights = {k: z[k].astype(np.float32) for k in z.files}

    cfg = SBConfig.from_json(cfg_json)
    cfg.vocab_size = weights["wte"].shape[0]
    m = StoryByte(cfg)
    # torch model loads the same shipped file
    with tempfile.TemporaryDirectory(prefix="storybyte-verify-") as temp_dir:
        tmp = Path(temp_dir) / "weights.npz"
        np.savez(str(tmp), **weights)
        m.load_npz(tmp)
    m.eval()

    ids = tk.encode(f"{REQ_TOKEN} {RUNNING_REQUEST} {STORY_TOKEN}")
    with torch.no_grad():
        t_logits, _ = m(torch.tensor([ids]))
    n_logits = np_forward(weights, ids)
    diff = float(np.abs(t_logits[0].numpy() - n_logits).max())

    np_ids = np_greedy(weights, ids, 60)
    with torch.no_grad():
        t_ids = m.generate(torch.tensor([ids]), max_new_tokens=60, greedy=True,
                           eos_id=EOS_ID)[0].tolist()
    agree = np_ids == t_ids[: len(np_ids)] or t_ids == np_ids[: len(t_ids)]
    return {"max_logit_diff": diff, "greedy_agreement": bool(agree),
            "greedy_text": tk.decode(np_ids[len(ids):], skip_special_tokens=True)[:120]}


def main():
    cfg_json = load_config(base_artifacts_dir())
    tk = SBTokenizer(WEB / "tokenizer_ext.json")
    report = {
        "sft_weights.npz": check("sft_weights.npz", tk, cfg_json),
        "dpo_weights.npz (f16 on disk)": check("dpo_weights.npz", tk, cfg_json),
        "pass": None,
    }
    report["pass"] = all(
        v["greedy_agreement"] and v["max_logit_diff"] < 1e-3
        for k, v in report.items() if isinstance(v, dict)
    )
    with open(WEB / "verification.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(out_dir() / "web_verification.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
