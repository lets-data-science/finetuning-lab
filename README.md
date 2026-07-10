# finetuning-lab

Companion code for the Let's Data Science course
["Fine-Tuning LLMs: Make the Model Yours"](https://letsdatascience.com/learn/fine-tuning-llms).

This repo fine-tunes [StoryByte](https://github.com/lets-data-science/storybyte), the
~1.09M-parameter GPT from
[Build a Tiny LLM](https://letsdatascience.com/learn/build-a-tiny-llm). It runs five
post-training experiments: full SFT, LoRA, DPO, knowledge distillation, and int8
quantization. Every training run finishes on a laptop CPU in seconds to minutes, and
the numbers shown in the course come from the scripts here.

The important caveat: the metrics are heuristic string checks, not human ratings. They
ask whether the story mentions the requested animal, repeats the requested name, and
terminates cleanly. The DPO preference label is also a rule: "contains dialogue." The
mechanics transfer to larger models; the constants do not.

## Repository layout

```text
finetuning-lab/
  storybyte/
    sb_common/                 model.py, tokenizer.py, paths.py
    00_env_check.py            verify the environment and StoryByte port
    01_build_requests_dataset.py
    02_sft_full.py             full SFT with loss masking
    03_train_lora.py           LoRA adapters, merge, and adapter export
    04_build_preferences.py    DPO pair mining
    05_dpo.py                  policy vs frozen SFT reference
    06_distill_kd.py           logit distillation vs scratch control
    07_quantize_export.py      int8 quantization and web export
    08_eval_suite.py           eval scoreboard
    09_verify_web_artifacts.py NumPy vs PyTorch parity checks
  data/                        synthesized dataset and frozen gold sets
  checkpoints/                 trained weights kept small enough for git
  results/                     eval rows and training traces
  web_artifacts/               standalone export target for browser files
  Makefile
  requirements.txt
  REPRODUCIBILITY.md
  HARDWARE.md
```

Optimizer states, merged LoRA weights, and scratch temp files are ignored because they
are regenerable and large. The browser course consumes the exported files under
`public/learn/fine-tuning-llms/` in the website repo.

## Quickstart

```bash
pip install -r requirements.txt

# Running outside the lets-data-science monorepo? Point this at the StoryByte
# base artifacts: storybyte_weights.npz, _tokenizer_hf.json, and _config.json.
export FTLAB_ARTIFACTS=/path/to/build-a-tiny-llm

# Optional: choose an explicit browser-artifact output directory.
export FTLAB_WEB_DIR=/path/to/website/public/learn/fine-tuning-llms

# Reproduce the course ladder from the committed frozen data.
make reproduce

# Optional: regenerate sampled data first. Metrics can drift slightly.
make all
```

Step by step, using the same complete targets as `make reproduce`:

```bash
make env       # 00: environment and base-model parity
make sft       # 02: 450 steps, uninterrupted
make lora      # 03: ranks 1, 2, 4, and 8
make prefs     # 04: all six preference-mining shards + exact assembly
make dpo       # 05: separate 25-step and 300-step checkpoints
make distill   # 06: scratch control + KD, 1500 steps each
make eval      # 08: all 10 models, all 3 slices, exact part assembly
make export    # 07: int8 evaluation and web-artifact export
make verify    # 09: NumPy-browser-path vs PyTorch parity

# Show every underlying Python command and required argument without running it.
make commands
```

`make reproduce` keeps the committed `data/` inputs. `make all` rebuilds them first;
dataset synthesis samples the base model, so a fresh run can drift slightly.
Inside the website checkout, export writes to `public/learn/fine-tuning-llms/`. In a
standalone clone it writes to `web_artifacts/` unless `FTLAB_WEB_DIR` is set.

## Base model and task

| Item | Value |
|---|---|
| Base model | StoryByte, decoder-only GPT, 4 layers, 4 heads, d_model 128, context 256 |
| Parameters | 1,088,256, verified by `00_env_check.py` |
| Vocab | Byte-level BPE 2,048, extended to 2,052 with `<\|req\|>`, `<\|story\|>`, `<\|talk\|>`, `<\|notalk\|>` |
| Task | `Tell me a story about a {animal} named {Name}.` |
| Running request | `Tell me a story about a dog named Rex.` Rex is unseen, so success means copying from context. |
| Compliance | Character, name, and format checks. Full compliance means all three pass. |
| Eval sampler | Temperature 0.8, top-k 40, seeds 1337-1339, 60 requests, 180 generations per model. |

## Results

All results below are measured from `results/eval_ladder.json` by
`storybyte/08_eval_suite.py`.

| Model | Trainable params, total | Dialogue | Character | Name | Format | Full | Forgetting ppl |
|---|---:|---:|---:|---:|---:|---:|---:|
| base, plain prompt | 0 | 80.6% | 40.0% | 85.0% | 91.7% | 35.6% | 2.889 |
| SFT full, step 150 | 1,088,768 | 59.4% | 93.9% | 80.0% | 92.8% | 71.7% | 3.484 |
| LoRA r=1 | 3,584 | 77.8% | 81.7% | 70.0% | 83.3% | 50.0% | 2.968 |
| LoRA r=2 | 6,656 | 72.2% | 83.3% | 75.6% | 91.1% | 60.0% | 2.951 |
| LoRA r=4 | 12,800 | 74.4% | 86.1% | 80.6% | 88.9% | 61.1% | 2.948 |
| LoRA r=8 | 25,088 | 74.4% | 85.0% | 81.1% | 88.3% | 62.2% | 2.934 |
| DPO 25 steps | all, from SFT | 94.4% | 93.9% | 80.0% | 88.3% | 67.8% | 3.639 |
| DPO 300 steps | all, from SFT | 100% | 92.8% | 79.4% | 57.8% | 45.0% | 4.211 |
| Nano scratch, 1500 | 445,440 | 75.6% | 65.0% | 48.9% | 97.8% | 36.7% | 96.47 |
| Nano KD, 1500 | 445,440 | 73.3% | 76.7% | 50.6% | 100% | 39.4% | 43.79 |

The short read:

- SFT moves full compliance from 35.6% to 71.7%. On the frozen held-out-requested-name
  condition, copying drops from 82.2% to 60.0%; the strict lexical-unseen slice described
  below is 80.2% to 55.6%.
- LoRA learns less and forgets less. The r=8 run reaches 62.2% full compliance while
  staying close to base forgetting perplexity. The r=4 adapter is about 75x smaller
  than the full model.
- DPO is a dial. At 25 steps, dialogue rises from 59.4% to 94.4%. At 300 steps,
  dialogue hits 100%, but format falls to 57.8%.
- Distillation gives the small model a better distribution. KD reaches 43.79
  forgetting perplexity; the scratch control is 96.47.
- int8 quantization shrinks the nano model from 1,788,986 to 1,127,706 bytes, about
  37% smaller, while greedy decoding stays byte-stable on the running request.

### Protected-name audit of the frozen dataset

All ten gold names are absent from training requests and primary-name labels. `Bella`
nevertheless appears as a secondary character in 5 of 571 frozen training stories (23
whole-word occurrences); validation is clean, and no other protected name occurs in a
training story. Existing artifacts are preserved, so the historical condition is labeled
**held-out requested name**, not strict lexical unseen.

For a strict comparison, exclude Bella's 9 gold generations (3 templates x 3 seeds) and
re-aggregate the persisted eval rows: base name compliance is 65/81 (80.2%), and SFT is
45/81 (55.6%). This is not a training or generation rerun. The running name Rex is absent
from every train/validation story and remains strictly unseen. Future dataset builds scan
the entire story and reject every protected-name occurrence.

Parameter accounting is separate: full SFT trains 1,088,256 base values plus 4 x 128 new
embedding values, or 1,088,768 total. LoRA counts already include those 512 embedding
values. Adapter-only counts are 3,072 / 6,144 / 12,288 / 24,576; adding 512 gives the
table totals 3,584 / 6,656 / 12,800 / 25,088.

## Measured CPU time

| Run | Config | CPU time |
|---|---|---:|
| SFT | 450 steps | ~231 s |
| LoRA | 450 steps, per rank | ~214-229 s |
| DPO | 25 steps | ~7 s |
| DPO overcooked control | 300 steps | ~72.3 s |
| KD student | 1500 steps | ~586 s |
| Nano scratch | 1500 steps | ~268 s |
| int8 quantize and export | one pass | ~5.5 s |

The recorded SFT, four LoRA, both DPO, scratch, KD, and export runs sum to about
**34.4 CPU minutes**. That is a derived subtotal, not a full-pipeline runtime. Fresh
data synthesis adds **9.2 measured minutes**. Preference mining, the full ten-model
eval, and parity checks add hardware-dependent time; `make reproduce` and `make all`
print actual end-to-end wall seconds.

Two checks must stay separate. The offline int8 evaluation compares fp32 nano against
dequantized int8 nano: max logit diff **0.169**, with greedy output byte-stable on the
running request. Browser-path parity compares the NumPy worker against PyTorch on the
shipped SFT and DPO artifacts: max differences **3.3e-5** and **9.62e-5**, with
byte-identical greedy decoding.

## Web artifact contract

`storybyte/07_quantize_export.py` writes the files the course loads on demand, and
`storybyte/09_verify_web_artifacts.py` checks them.

| File | Bytes | Purpose |
|---|---:|---|
| `sft_weights.npz` | 4,368,538 | full SFT weights, f32 |
| `dpo_weights.npz` | 2,191,002 | DPO 25-step checkpoint, f16 |
| `dpo_overcooked.npz` | 2,191,002 | DPO 300-step checkpoint, f16 |
| `lora_adapters.npz` | 220,078 | all four LoRA ranks in one file |
| `nano_kd.npz` | 1,788,986 | distilled nano student |
| `nano_int8.npz` | 1,127,706 | int8-quantized nano |
| `nano_config.json` | 132 | nano architecture |
| `tokenizer_ext.json` | 124,168 | BPE tokenizer plus four special tokens |
| `eval_results.json` | 16,472 | full ladder mirrored from `results/eval_ladder.json` |
| `train_traces.json` | 25,648 | loss, LR, and probe curves |
| `preference_pairs.json` | 26,801 | 20 DPO pairs with policy/reference logprobs |
| `sample_generations.json` | 3,737 | recorded fallback generations |
| `verification.json` | 522 | NumPy/PyTorch parity proof |

## Sources

- InstructGPT: Ouyang et al., 2022, [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- LoRA: Hu et al., 2021, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- QLoRA: Dettmers et al., 2023, [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- LIMA: Zhou et al., 2023, [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)
- LoRA Learns Less and Forgets Less: Biderman et al., 2024, [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)
- The False Promise of Imitating Proprietary LLMs: Gudibande et al., 2023, [arXiv:2305.15717](https://arxiv.org/abs/2305.15717)
- DPO: Rafailov et al., 2023, [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Distilling the Knowledge in a Neural Network: Hinton, Vinyals, and Dean, 2015, [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)

MIT licensed. Built by [Let's Data Science](https://letsdatascience.com).
