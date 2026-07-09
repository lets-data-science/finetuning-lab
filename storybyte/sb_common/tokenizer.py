"""StoryByte tokenizer - a thin wrapper around the authoritative HF tokenizer file.

The tiny-llm course ships two tokenizer files; storybyte_tokenizer_hf.json is the
authoritative encoder (GPT-2 byte-level BPE, use_regex=True, add_prefix_space=False).
We load it with the `tokenizers` library so encoding here is byte-identical to the
shipped model's training tokenizer.

Fine-tuning extension: `extend_with_specials` registers the request-format special
tokens. New ids are appended after the base vocab (2048, 2049, ...), which is why
the model's embedding matrix must be resized to match (see model.resize_vocab).
"""
from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer

EOS_TOKEN = "<|endoftext|>"
EOS_ID = 0

# The course's request-format special tokens (entity registry canon).
# v3 decision (measured): the dialogue control rides on dedicated CONTROL TOKENS,
# CTRL-style, because the natural-phrase contrast ("with talking" vs "with no
# talking") was NOT learnable at 1.09M params (dialogue compliance stayed ~chance
# in v1/v2 - see results/eval_ladder.json). The UI shows the request in natural
# language and compiles it to this format transparently.
REQ_TOKEN = "<|req|>"
STORY_TOKEN = "<|story|>"
TALK_TOKEN = "<|talk|>"
NOTALK_TOKEN = "<|notalk|>"
SPECIAL_TOKENS = [REQ_TOKEN, STORY_TOKEN, TALK_TOKEN, NOTALK_TOKEN]


class SBTokenizer:
    def __init__(self, hf_json_path: str | Path):
        self.path = str(hf_json_path)
        self.tok = Tokenizer.from_file(self.path)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size(with_added_tokens=True)

    def extend_with_specials(self) -> dict[str, int]:
        """Add <|req|> and <|story|> as special tokens; returns their ids."""
        self.tok.add_special_tokens(SPECIAL_TOKENS)
        vocab = self.tok.get_vocab(with_added_tokens=True)
        return {t: vocab[t] for t in SPECIAL_TOKENS}

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def token_strs(self, ids: list[int]) -> list[str]:
        return [self.tok.id_to_token(i) for i in ids]

    def export_extended_json(self, out_path: str | Path) -> None:
        """Persist the extended tokenizer (base + specials) for reuse/browser export."""
        self.tok.save(str(out_path))


def special_ids(tk: SBTokenizer) -> dict[str, int]:
    vocab = tk.tok.get_vocab(with_added_tokens=True)
    out = {EOS_TOKEN: vocab[EOS_TOKEN]}
    for t in SPECIAL_TOKENS:
        if t in vocab:
            out[t] = vocab[t]
    return out


def load_base_tokenizer(base_dir: str | Path) -> SBTokenizer:
    return SBTokenizer(Path(base_dir) / "storybyte_tokenizer_hf.json")


def load_config(base_dir: str | Path) -> dict:
    with open(Path(base_dir) / "storybyte_config.json") as f:
        return json.load(f)
