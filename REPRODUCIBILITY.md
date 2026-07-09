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
  little sampling noise. To reproduce the **exact** frozen numbers, keep the committed `data/`
  and `results/` and skip `make data`.

## Expected numbers (the ladder)

Full compliance on the 60-request gold set (180 generations/model). Source of truth:
`results/eval_ladder.json`.

| Model | Trainable params | Dialogue | Character | Name | Format | FULL | Forgetting ppl |
|---|---|---|---|---|---|---|---|
| base | 0 | 80.6% | 40.0% | 85.0% | 91.7% | 35.6% | 2.889 |
| SFT full (step 150) | 1,088,640 | 59.4% | 93.9% | 80.0% | 92.8% | 71.7% | 3.484 |
| LoRA r=1 | 3,584 | 77.8% | 81.7% | 70.0% | 83.3% | 50.0% | 2.968 |
| LoRA r=2 | 6,656 | 72.2% | 83.3% | 75.6% | 91.1% | 60.0% | 2.951 |
| LoRA r=4 | 12,800 | 74.4% | 86.1% | 80.6% | 88.9% | 61.1% | 2.948 |
| LoRA r=8 | 25,088 | 74.4% | 85.0% | 81.1% | 88.3% | 62.2% | 2.934 |
| DPO 25 steps (ships) | (all, from SFT) | 94.4% | 93.9% | 80.0% | 88.3% | 67.8% | 3.639 |
| DPO 300 steps (overcooked) | (all) | 100% | 92.8% | 79.4% | 57.8% | 45.0% | 4.211 |
| Nano scratch (1500) | 445,440 | 75.6% | 65.0% | 48.9% | 97.8% | 36.7% | 96.47 |
| Nano KD (1500, ships) | 445,440 | 73.3% | 76.7% | 50.6% | 100% | 39.4% | 43.79 |

LoRA trainable counts exclude the 512 new-embedding-row params (4 rows x 128) it also trains.
Adapter file sizes: r1 21,094 B, r2 33,382 B, r4 57,958 B, r8 107,110 B (vs the 4,368,538 B
full model - the r4 adapter is ~75x smaller).

## Training recipes

- **SFT:** AdamW beta=(0.9, 0.95), lr 3e-4 -> 3e-5 cosine, warmup 60, wd 0.1, clip 1.0, batch 32,
  block 256, loss on story tokens only. 450 steps. Checkpoint policy = **best dev-probe
  compliance (12 requests, not gold)** -> step 150. (Val-loss minimum was step 50 - val loss
  picks fluency, the probe picks behavior. That gap is the lesson.)
- **LoRA:** adapters on attention `c_attn` + `c_proj` (all 4 blocks), alpha = 2r, lr 1e-3 cosine,
  base frozen except the 4 new token-embedding rows. 450 steps.
- **DPO:** beta 0.2, lr 1e-5, batch 8 pairs, policy + reference init from SFT; 113 preference pairs
  (chosen = has dialogue), K=4 sampled at temp 0.9. Ships at 25 steps; margin 2.88,
  pair-accuracy 1.00.
- **KD:** student 2L / 4H / d96 = 445,440 params (41% of teacher); loss = 0.5*T^2*KL(T=2) + 0.5*CE
  on story tokens; lr 6e-4, 1500 steps.

## Browser parity

The course runs these models in a NumPy web worker. Verified against PyTorch on the shipped
artifacts (`results/web_verification.json`, mirrored to `verification.json`):

| Artifact | Max logit diff | Greedy decode |
|---|---|---|
| `sft_weights.npz` | **3.3e-5** | byte-identical |
| `dpo_weights.npz` (f16) | **2.7e-5** | byte-identical |
| `nano_int8.npz` | 0.169 | byte-stable on the running request |

Run it yourself: `python3 storybyte/09_verify_web_artifacts.py`.

## A note on determinism

Runs are CPU float32. Tiny numerical differences across BLAS builds or hardware will not move the
compliance rates materially, and greedy decoding on the running request is byte-stable. The
audit trail for every model is the per-model `results/eval_detail_*.json` and `eval_rows_*` files.
