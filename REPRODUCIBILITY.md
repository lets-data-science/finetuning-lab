# Reproducibility

Everything the course shows was measured by the numbered scripts in `storybyte/` and
persisted as JSON in `results/`. This file records the seeds, versions, and expected
numbers so you can confirm a fresh run matches.

## Versions

Captured in `results/env_check.json` (regenerate with `python3 storybyte/00_env_check.py`):

| Component | Version |
|---|---|
| torch | **2.12.1+cpu** |
| tokenizers | **0.23.1** |
| numpy | not captured in env_check.json - any modern NumPy (1.26+ / 2.x) reproduces results |
| Python | 3.10+ |

Base model sanity, also from `env_check.json`: **1,088,256 params** (matches config),
greedy prefix match against the published sample = true, ~1.83 ms/token on CPU.

## Seeds & sampler

- **Eval sampler seeds: `{1337, 1338, 1339}`.** Every ladder number is 3 samples x 60 gold
  requests = **180 generations per model**, at **temperature 0.8, top-k 40**.
- Canned fallback generations (`sample_generations.json`) are frozen at **seed 1337**.
- Dataset synthesis (`01_...`) samples the base model, so regenerating `data/` drifts by a
  little sampling noise. Use `make reproduce` for the committed frozen inputs; `make all`
  intentionally regenerates data first.

## Complete commands

`make reproduce` runs environment checks, SFT, all four LoRA ranks, all six preference-mining
shards plus exact assembly, both DPO checkpoints, scratch and KD students, all three eval parts
for all ten models, export, and browser-path parity. `make commands` prints every Python
command and required argument without starting the long run. `make all` adds fresh data synthesis.
Both timed targets print their actual end-to-end wall seconds.
Standalone clones export to `web_artifacts/`; set `FTLAB_WEB_DIR` to target a website checkout.

## Frozen protected-name caveat

The shipped 571-row training artifact has no protected name in any request or primary-name
field. A later whole-row audit found `Bella` as a secondary character in exactly 5 training
stories (23 whole-word occurrences); validation is clean and the other nine protected names
do not occur in training stories. The current builder rejects every protected name anywhere
in future story text, but the frozen data and checkpoints were not regenerated.

For the frozen eval, `name_cond=unseen_name` therefore means **held-out requested name**.
The original base/SFT name rates over all 90 rows in that condition remain 82.2%/60.0%.
A strict lexical-unseen re-aggregation excludes Bella's 9 generations and gives 65/81
(80.2%) for base and 45/81 (55.6%) for SFT. These values come from the existing
`eval_detail_*.json` rows; no model was rerun. Rex has zero train/validation story
occurrences, so the running request remains strictly unseen.

## Expected numbers (the ladder)

Full compliance on the 60-request gold set (180 generations/model). Source of truth:
`results/eval_ladder.json`.

| Model | Trainable params, total | Dialogue | Character | Name | Format | FULL | Forgetting ppl |
|---|---|---|---|---|---|---|---|
| base | 0 | 80.6% | 40.0% | 85.0% | 91.7% | 35.6% | 2.889 |
| SFT full (step 150) | 1,088,768 | 59.4% | 93.9% | 80.0% | 92.8% | 71.7% | 3.484 |
| LoRA r=1 | 3,584 | 77.8% | 81.7% | 70.0% | 83.3% | 50.0% | 2.968 |
| LoRA r=2 | 6,656 | 72.2% | 83.3% | 75.6% | 91.1% | 60.0% | 2.951 |
| LoRA r=4 | 12,800 | 74.4% | 86.1% | 80.6% | 88.9% | 61.1% | 2.948 |
| LoRA r=8 | 25,088 | 74.4% | 85.0% | 81.1% | 88.3% | 62.2% | 2.934 |
| DPO 25 steps (ships) | (all, from SFT) | 94.4% | 93.9% | 80.0% | 88.3% | 67.8% | 3.639 |
| DPO 300 steps (overcooked) | (all) | 100% | 92.8% | 79.4% | 57.8% | 45.0% | 4.211 |
| Nano scratch (1500) | 445,440 | 75.6% | 65.0% | 48.9% | 97.8% | 36.7% | 96.47 |
| Nano KD (1500, ships) | 445,440 | 73.3% | 76.7% | 50.6% | 100% | 39.4% | 43.79 |

LoRA trainable counts include the 512 new-embedding-row values (4 rows x 128) it trains.
For example, r=4 is 12,288 adapter-matrix values + 512 embedding values = 12,800 total.
Adapter file sizes: r1 21,094 B, r2 33,382 B, r4 57,958 B, r8 107,110 B (vs the 4,368,538 B
full model - the r4 adapter is ~75x smaller).

## Training recipes

- **SFT:** AdamW beta=(0.9, 0.95), lr 3e-4 -> 3e-5 cosine, warmup 60, wd 0.1, clip 1.0, batch 32,
  block 256, loss on story tokens only. 450 steps. Checkpoint policy = **best dev-probe
  compliance (12 requests, not gold)** -> step 150. (Val-loss minimum was step 50 - val loss
  picks fluency, the probe picks behavior. That gap is the lesson.)
- **LoRA:** adapters on attention `c_attn` + `c_proj` (all 4 blocks), alpha = 2r, lr 1e-3 cosine,
  base frozen except the 4 new token-embedding rows. Row-vector notation is `x @ A @ B`;
  merging uses `W + (alpha/r) * A @ B`. 450 steps.
- **DPO:** beta 0.2, lr 1e-5, batch 8 pairs, policy + reference init from SFT; higher beta means
  stronger reference regularization and less policy deviation at the optimum; 113 preference pairs
  (chosen = has dialogue), K=4 sampled at temp 0.9. Ships at 25 steps; margin 2.88,
  pair-accuracy 1.00.
- **KD:** student 2L / 4H / d96 = 445,440 params (41% of teacher); loss = 0.5*T^2*KL(T=2) + 0.5*CE
  on story tokens; lr 6e-4, 1500 steps.

## Measured CPU time

The recorded SFT, four LoRA, 25-step and 300-step DPO, scratch, KD, and export runs sum to a
derived **34.4 CPU minutes**. Fresh data synthesis adds **9.2 measured minutes**. Preference
mining, the full eval, and parity checks add hardware-dependent time, so this is not presented as
a full-pipeline estimate. The timed Make targets report the actual wall time.

## int8 evaluation vs browser parity

The offline int8 evaluation in `07_quantize_export.py` compares fp32 nano with dequantized int8
nano on the running request. It measured max logit diff **0.169** and byte-stable greedy output.
That is a quantization-degradation check, not browser-worker parity.

The course runs these models in a NumPy web worker. Verified against PyTorch on the shipped
artifacts (`results/web_verification.json`, mirrored to `verification.json`):

| Artifact | Max logit diff | Greedy decode |
|---|---|---|
| `sft_weights.npz` | **3.3e-5** | byte-identical |
| `dpo_weights.npz` (f16) | **9.62e-5** | byte-identical |

Run it yourself: `python3 storybyte/09_verify_web_artifacts.py`.

## A note on determinism

Runs are CPU float32. Tiny numerical differences across BLAS builds or hardware will not move the
compliance rates materially, and greedy decoding on the running request is byte-stable. The
audit trail for every model is the per-model `results/eval_detail_*.json` and `eval_rows_*` files.
