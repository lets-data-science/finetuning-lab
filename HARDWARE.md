# Hardware

**No GPU required. No MPS, no CUDA, no cloud.** Every training run, eval, and export in this repo
runs on a laptop CPU. That is the point: post-training a 1.09M-parameter model is cheap enough that
the whole course is reproducible on the machine you're reading this on.

## Reference machine

- A modest **arm64 CPU** (Apple-Silicon class). Your machine is probably faster.
- **torch 2.12.1+cpu**, tokenizers 0.23.1 (see `results/env_check.json`).
- Measured base-model throughput: **~1.83 ms/token** on CPU.
- Memory: negligible. The model is 1,088,256 params (~4.3 MB as fp32); everything fits in well
  under 1 GB of RAM.

## Wall-clock times (measured, CPU)

| Stage | Config | Time |
|---|---|---|
| `00` env check | — | a few seconds |
| `01` dataset synthesis | 8 animals x 2 rounds | ~33-43 s per shard |
| `02` SFT | 450 steps | **~231 s** (~0.45 s/step) |
| `03` LoRA | 450 steps, per rank (x4 ranks) | **~214-229 s** each |
| `05` DPO (ships) | 25 steps | **~7 s** |
| `05` DPO (overcooked demo) | 300 steps | ~85 s (estimate at ~0.28 s/step; only the 25-step run is in canon) |
| `06` KD student | 1500 steps | **~586 s** (teacher forward roughly doubles cost) |
| `06` nano scratch control | 1500 steps | **~268 s** |
| `07` int8 quantize + web export | — | ~5.5 s |
| `08` eval suite | 180 generations per model | minutes per model (nano models are faster) |

## Time-boxed environments

Long-running scripts (`02`, `03`, `05`, `06`) accept `--resume --max-seconds N`: run the same
command repeatedly until it prints the final JSON instead of a `paused_at_step`. This lets the
pipeline checkpoint through environments that cap process wall-time. On a normal laptop you never
need it — just run the scripts to completion (that's what `make all` does).
