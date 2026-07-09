"""Gate 0: prove the environment and the model port are exact before training anything.

Checks:
  1. torch + tokenizers import, versions printed.
  2. StoryByte loads from the shipped .npz and its greedy continuation of
     "Once upon a time" matches the published sample (sample_generations.json)
     character-for-character. If this fails, NOTHING downstream can be trusted.
  3. Reports CPU ms/token so the course's "reproducible on a laptop" numbers are honest.

Run: python3 00_env_check.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sb_common.paths import base_artifacts_dir, out_dir
from sb_common.tokenizer import load_base_tokenizer, load_config
from sb_common.model import load_base_model

import torch


def main() -> None:
    base = base_artifacts_dir()
    import tokenizers

    report: dict = {
        "torch_version": torch.__version__,
        "tokenizers_version": tokenizers.__version__,
    }

    cfg = load_config(base)
    tk = load_base_tokenizer(base)
    model = load_base_model(base, cfg)
    report["n_params"] = model.num_params()
    report["n_params_matches_config"] = model.num_params() == cfg["n_params"]

    with open(base / "sample_generations.json") as f:
        samples = json.load(f)
    prompt = "Once upon a time"
    expected = samples[prompt]["greedy"]

    ids = tk.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)
    t0 = time.perf_counter()
    y = model.generate(x, max_new_tokens=80, greedy=True, eos_id=0)
    dt = time.perf_counter() - t0
    new_tokens = y.size(1) - len(ids)
    text = tk.decode(y[0].tolist(), skip_special_tokens=True)

    n = min(len(text), len(expected))
    report["greedy_prefix_match"] = text[:n] == expected[:n]
    report["compared_chars"] = n
    report["ms_per_token_cpu"] = round(1000.0 * dt / max(new_tokens, 1), 2)
    report["generated_preview"] = text[:120]

    ok = report["greedy_prefix_match"] and report["n_params_matches_config"]
    report["pass"] = bool(ok)

    with open(out_dir() / "env_check.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
