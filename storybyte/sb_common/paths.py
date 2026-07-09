"""Path resolution for the finetuning-lab repo.

Layout assumption (inside the lets-data-science project during development):
  <LDS_ROOT>/scripts/finetuning-lab/storybyte/sb_common/paths.py

When this directory is extracted into its own GitHub repo, set the env var
FTLAB_ARTIFACTS to a directory containing the base StoryByte artifacts
(storybyte_weights.npz, storybyte_tokenizer_hf.json, storybyte_config.json).
"""
import os
from pathlib import Path

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[2]  # scripts/finetuning-lab/
LDS_ROOT = _HERE.parents[4]   # project root (dev layout)

_DEFAULT_BASE = LDS_ROOT / "public" / "learn" / "build-a-tiny-llm"


def base_artifacts_dir() -> Path:
    env = os.environ.get("FTLAB_ARTIFACTS")
    if env:
        return Path(env)
    return _DEFAULT_BASE


def out_dir() -> Path:
    d = REPO_ROOT / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir() -> Path:
    d = REPO_ROOT / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ckpt_dir() -> Path:
    d = REPO_ROOT / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d
