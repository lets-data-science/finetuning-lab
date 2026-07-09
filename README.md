# finetuning-lab 🎛️🤖 — take a tiny LLM and make it yours

**This repo fine-tunes [StoryByte](https://github.com/lets-data-science/storybyte) — the real ~1.09M-parameter GPT trained from scratch in [Build a Tiny LLM](https://letsdatascience.com/learn/build-a-tiny-llm) — five different ways: full SFT, LoRA, DPO, knowledge distillation, and int8 quantization.** Every training run completes on a laptop CPU in seconds to minutes, and every number the course shows regenerates from the numbered scripts here.

It's the offline half of the interactive course **["Fine-Tuning LLMs: Make the Model Yours"](https://letsdatascience.com/learn/fine-tuning-llms)** — the post-training sequel to Build a Tiny LLM. The course teaches the decisions and failure modes of post-training; this repo is where those numbers are actually measured.

> **Honesty first:** every metric here is a **heuristic string check** (did the story mention the animal? repeat the name? terminate cleanly?), not a human judgment, and the preference labels are a **rule** ("the reply contains dialogue"), not human ranking. All results are at **1.09M-parameter scale on a toddler-vocabulary story model**. The mechanics and the failure modes transfer to 7B+ models; the *constants* (how many examples, which rank, how many DPO steps) **do not**. The course says this on every module, and so does this repo.

---

## What's in here

```
finetuning-lab/
├── storybyte/
│   ├── sb_common/              # model.py, tokenizer.py, paths.py (the StoryByte port)
│   ├── 00_env_check.py         # prove the environment + base model port are EXACT
│   ├── 01_build_requests_dataset.py  # synthesize request->story data, rule-labeled
│   ├── 02_sft_full.py          # full SFT with loss masking + probe-based checkpointing
│   ├── 03_train_lora.py        # LoRA adapters (attention), merge + adapter export
│   ├── 04_build_preferences.py # mine DPO preference pairs (chosen = has dialogue)
│   ├── 05_dpo.py               # DPO: policy vs frozen SFT reference
│   ├── 06_distill_kd.py        # logit distillation vs a scratch control
│   ├── 07_quantize_export.py   # int8 quantize + export the web artifacts
│   ├── 08_eval_suite.py        # the eval scoreboard — the single source of every number
│   └── 09_verify_web_artifacts.py  # NumPy-vs-PyTorch parity on the shipped files
├── data/                       # synthesized dataset + gold/holdout sets (committed, frozen)
├── checkpoints/                # trained weights (heavy states ship via a Release — see below)
├── results/                    # every eval + training trace as JSON (the audit trail)
├── Makefile                    # `make all` = the whole pipeline, 00 -> 09
├── requirements.txt            # pinned to results/env_check.json
├── README.md
├── REPRODUCIBILITY.md          # seeds, versions, expected numbers, browser parity
└── HARDWARE.md                 # CPU-only; wall-clock times
```

**Heavy checkpoints ship via a GitHub Release, not git.** Optimizer states (`*_state.pt`),
merged LoRA weights (`*_merged.npz`), and scratch temp files are `.gitignore`d because they're
regenerable and large (~110 MB). The small artifacts the course actually consumes live in
[`public/learn/fine-tuning-llms/`](#the-contract-the-course-consumes) in the website repo; the
provenance copies are the light `.npz` files kept in `checkpoints/`.

## Quickstart — reproduce it yourself

```bash
pip install -r requirements.txt

# Running outside the lets-data-science monorepo? Point this at the StoryByte
# base artifacts (storybyte_weights.npz / _tokenizer_hf.json / _config.json):
export FTLAB_ARTIFACTS=/path/to/build-a-tiny-llm

make all            # env -> data -> sft -> lora -> prefs -> dpo -> distill -> export -> eval -> verify
```

Or step by step:

```bash
cd storybyte
python3 00_env_check.py                                   # gate: base model port is EXACT
python3 01_build_requests_dataset.py --animal dog --round 1   # ... per animal, rounds 1 & 2
python3 01_build_requests_dataset.py --assemble
python3 02_sft_full.py --steps 450                        # full SFT + honest overfit curves
python3 03_train_lora.py --rank 4 --steps 450             # LoRA (repeat for r=1,2,8)
python3 04_build_preferences.py --shard 1:6               # ... shards 1..6, then --assemble
python3 05_dpo.py --steps 25 --beta 0.2 --lr 1e-5         # DPO — 25 steps, ~7s
python3 06_distill_kd.py --mode kd --steps 1500           # KD student (and --mode scratch)
python3 07_quantize_export.py                             # int8 + web export
python3 08_eval_suite.py --model sft --part 1:3           # (2:3, 3:3), then:
python3 08_eval_suite.py --model sft --report
python3 09_verify_web_artifacts.py                        # NumPy == PyTorch on shipped files
```

To reproduce the **exact** shipped numbers, keep the committed `data/` and `results/` and skip
`make data` — dataset synthesis samples the base model, so a fresh run drifts by a little noise.
Long runs accept `--resume --max-seconds N` for time-boxed environments.

## The base model + the task

| | |
|---|---|
| Base model | StoryByte — decoder-only GPT, 4 layers / 4 heads / d_model 128, context 256 |
| Parameters | **1,088,256** (verified in `00_env_check.py`) |
| Vocab | byte-level BPE 2,048, extended to 2,052 with 4 special tokens (`<\|req\|>`, `<\|story\|>`, ...) |
| The task | `Tell me a story about a {animal} named {Name}.` — the model must obey character + **copy the name** |
| Running request | `Tell me a story about a dog named Rex.` (Rex is an **unseen** name — copying, not memorization) |
| Compliance | character (animal appears), name (name appears >=2x), format (ends `<\|endoftext\|>`, >=30 words); **full** = all three |
| Eval sampler | temp 0.8, top-k 40, seeds {1337, 1338, 1339}; 3 samples x 60 gold requests = **180 generations/model** |

| Method | Script | What it teaches |
|---|---|---|
| Full SFT | `02_sft_full.py` | loss masking, overfitting, checkpoint selection by behavior not val-loss |
| LoRA | `03_train_lora.py` | low-rank adapters; learns-less-forgets-less; the rank/quality knee |
| DPO | `05_dpo.py` | preference tuning as a dial, and over-optimization |
| Distillation | `06_distill_kd.py` | soft targets ("dark knowledge") vs training from scratch |
| int8 quant | `07_quantize_export.py` | quantization as controlled approximation + serving-size math |

## Results

All numbers below are **measured**, live in `results/eval_ladder.json`, and regenerate from
`08_eval_suite.py`. Full compliance is on the 60-request gold set (180 generations/model).

**The ladder — the course's spine:**

| Model | Trainable params | Dialogue | Character | Name | Format | **FULL** | Forgetting ppl |
|---|---|---|---|---|---|---|---|
| base (plain prompt) | 0 | 80.6% | 40.0% | 85.0% | 91.7% | **35.6%** | 2.889 |
| SFT full (step 150) | 1,088,640 | 59.4% | 93.9% | 80.0% | 92.8% | **71.7%** | 3.484 |
| LoRA r=1 | 3,584 | 77.8% | 81.7% | 70.0% | 83.3% | **50.0%** | 2.968 |
| LoRA r=2 | 6,656 | 72.2% | 83.3% | 75.6% | 91.1% | **60.0%** | 2.951 |
| LoRA r=4 | 12,800 | 74.4% | 86.1% | 80.6% | 88.9% | **61.1%** | 2.948 |
| LoRA r=8 | 25,088 | 74.4% | 85.0% | 81.1% | 88.3% | **62.2%** | 2.934 |
| DPO 25 steps (ships) | (all, from SFT) | **94.4%** | 93.9% | 80.0% | 88.3% | **67.8%** | 3.639 |
| DPO 300 steps (overcooked) | (all) | **100%** | 92.8% | 79.4% | **57.8%** | **45.0%** | 4.211 |
| Nano scratch (1500) | 445,440 | 75.6% | 65.0% | 48.9% | 97.8% | **36.7%** | 96.47 |
| Nano KD (1500, ships) | 445,440 | 73.3% | 76.7% | 50.6% | 100% | **39.4%** | **43.79** |

**How to read it (the honest version):**

- **SFT doubles full compliance** (35.6% -> 71.7%) and nearly solves character (40% -> 94%) — but
  *taxes* novel copying (unseen-name copying drops 82.2% -> 60.0%). Specialization costs
  generalization. Measured.
- **LoRA learns less AND forgets less** — Biderman et al. (2405.09673) reproduced at 1M scale.
  Best LoRA (r=8) reaches **62.2%** vs full SFT's 71.7% (learns less), but forgetting perplexity
  stays at base level ~**2.93** vs SFT's 3.48 (forgets less). The rank ladder 50 -> 60 -> 61 -> 62%
  is a textbook diminishing-returns curve; the knee is r ≈ 2-4. The **r=4 adapter is ~75x smaller**
  than the full model (57,958 B vs 4,368,538 B).
- **DPO is a dial, not a switch.** SFT suppressed dialogue to 59.4%; **25 DPO steps (~7s CPU)** lift
  it to **94.4%** at a modest cost (-3.9 pt full). 300 steps hit 100% dialogue but **collapse format**
  (92.8% -> 57.8%) — real, measured over-optimization.
- **Distillation's gift is distributional.** KD's headline compliance edge is small (+2.8 pt), but
  forgetting perplexity is **43.79 vs 96.47** for the scratch control — a 2.2x smaller distribution
  drift at the same 1500-step budget.
- **int8** shrinks the nano model **1,788,986 -> 1,127,706 bytes (~37% smaller** — the fp32 embedding
  table dominates a tiny model, taught honestly) and stays **greedy byte-stable** on the running
  request (max logit diff 0.169).

**Training is cheap (CPU wall-clock, measured):**

| Run | Config | CPU time |
|---|---|---|
| SFT | 450 steps | **~231 s** (~0.45 s/step) |
| LoRA | 450 steps, per rank | **~214-229 s** |
| DPO (ships) | 25 steps | **~7 s** |
| KD student | 1500 steps | ~586 s (teacher forward roughly doubles cost) |
| Nano scratch | 1500 steps | ~268 s |
| int8 quantize + export | — | ~5.5 s |

**Browser parity:** the course runs these models in a NumPy web worker. Against PyTorch on the
shipped artifacts, max logit diff is **3.3e-5** (SFT) and **2.7e-5** (DPO, f16), and greedy decoding
is **byte-identical** (`results/web_verification.json` -> `verification.json`).

<a name="the-contract-the-course-consumes"></a>
## `public/learn/fine-tuning-llms/` — the contract the course consumes

The browser course never bundles these; it fetches them on demand. This table is the contract —
`07_quantize_export.py` writes it and `09_verify_web_artifacts.py` checks it.

| File | Bytes | What |
|---|---|---|
| `sft_weights.npz` | 4,368,538 | full SFT weights (f32), GPT-2 naming |
| `dpo_weights.npz` | 2,191,002 | DPO weights (f16) — the shipped 25-step checkpoint |
| `dpo_overcooked.npz` | 2,191,002 | 300-step DPO (f16) — the over-optimization demo |
| `lora_adapters.npz` | 220,078 | all four LoRA ranks (r=1,2,4,8) in one file |
| `nano_kd.npz` | 1,788,986 | distilled nano student (the shipped KD model) |
| `nano_int8.npz` | 1,127,706 | int8-quantized nano (the serving-size demo) |
| `nano_config.json` | 132 | nano student architecture |
| `tokenizer_ext.json` | 124,168 | byte-level BPE + the 4 added special tokens |
| `eval_results.json` | 16,472 | the full ladder (mirrors `results/eval_ladder.json`) |
| `train_traces.json` | 25,648 | loss / LR / probe curves for the training animations |
| `preference_pairs.json` | 26,801 | 20 DPO pairs with policy + reference logprobs |
| `sample_generations.json` | 3,737 | canned fallback generations (seed 1337) |
| `verification.json` | 522 | proof the NumPy worker == PyTorch on the shipped files |

## Credits & sources

The methods here follow (quote each with its scope — the course does):

- **InstructGPT** — Ouyang et al., 2022, [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) (a 1.3B model, fine-tuned, preferred over 175B GPT-3).
- **LoRA** — Hu et al., 2021, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) (low-rank adapters; no added inference latency after merge).
- **QLoRA** — Dettmers et al., 2023, [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) (NF4 + double quantization + paged optimizers).
- **LIMA** — Zhou et al., 2023, [arXiv:2305.11206](https://arxiv.org/abs/2305.11206) (1,000 curated pairs, no RL).
- **LoRA Learns Less and Forgets Less** — Biderman et al., 2024, [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) (TMLR) — reproduced in direction here at 1M scale.
- **The False Promise of Imitating Proprietary LLMs** — Gudibande et al., 2023, [arXiv:2305.15717](https://arxiv.org/abs/2305.15717) (style ≠ capability).
- **DPO** — Rafailov et al., 2023, [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) (RLHF as a classification loss; no reward model).
- **Distilling the Knowledge in a Neural Network** — Hinton, Vinyals & Dean, 2015, [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) (soft targets / dark knowledge).

Base model, tokenizer, and the pure-NumPy inference style come from
[StoryByte](https://github.com/lets-data-science/storybyte).

MIT licensed. Built by [Let's Data Science](https://letsdatascience.com).
