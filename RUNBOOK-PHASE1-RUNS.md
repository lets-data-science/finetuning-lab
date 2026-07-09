# Advanced reproduction runs

`make all` reproduces the shipping ladder. These are the deeper experiments behind three of the
course's lessons - run them to regenerate the full ladder, including the entries that deliberately
teach a failure mode. All CPU, all from the `storybyte/` directory. Long runs accept
`--resume --max-seconds N` for time-boxed environments; on a laptop, just run them to completion.
Everything lands in `results/eval_ladder.json` - that file is the single source of truth.

## 1. The LoRA rank ladder (Module 4)

For each rank r in 1, 2, 4, 8:

```bash
python3 03_train_lora.py --rank <r> --steps 450
python3 08_eval_suite.py --model lora_r<r> --part 1:3   # then 2:3, 3:3
python3 08_eval_suite.py --model lora_r<r> --report
```

Record per rank: trainable params, adapter bytes, best dev-probe compliance, and the report metrics.
Expected shape: full compliance **50.0 -> 60.0 -> 61.1 -> 62.2%** (diminishing returns; knee at
r ~= 2-4), while forgetting perplexity stays near base (~2.93). LoRA learns less and forgets less.

## 2. DPO: the "just right" checkpoint vs over-optimization (Module 5)

The repo ships the 25-step DPO checkpoint and keeps a 300-step "overcooked" one to show the failure.

```bash
# ships (dpo_weights.npz):
python3 05_dpo.py --steps 25 --beta 0.2 --lr 1e-5
python3 08_eval_suite.py --model dpo --part 1:3        # 2:3, 3:3, then --report

# over-optimization demo (dpo_overcooked.npz):
python3 05_dpo.py --steps 300 --beta 0.2 --lr 1e-5
```

What to look for: dialogue climbs (59.4% -> **94.4%** at 25 steps -> 100% at 300), but by 300 steps
format compliance **collapses** (92.8% -> 57.8%) - stories stop terminating. DPO converges fast;
watch the downstream evals, not the DPO loss (pair-accuracy saturates by step 25).

## 3. KD vs scratch at equal budget (Module 6)

Run both arms at an identical step budget so the only difference is the teacher's soft targets:

```bash
python3 06_distill_kd.py --mode scratch --steps 1500
python3 06_distill_kd.py --mode kd --steps 1500
python3 08_eval_suite.py --model nano_scratch --part 1:3   # 2:3, 3:3, then --report
python3 08_eval_suite.py --model nano_kd --part 1:3        # 2:3, 3:3, then --report
```

The decisive number is forgetting perplexity: KD **43.79** vs scratch **96.47** (2.2x less drift)
at the same budget. The KD advantage widens with more steps while scratch plateaus.
