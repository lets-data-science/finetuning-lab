# finetuning-lab — reproduce every number in the "Fine-Tuning LLMs" course.
# CPU only. `make all` runs the whole pipeline (00 -> 09) in order.
# See README.md and REPRODUCIBILITY.md for the recipes and expected numbers.
#
# Running outside the lets-data-science monorepo? Point FTLAB_ARTIFACTS at the
# StoryByte base artifacts (storybyte_weights.npz / _tokenizer_hf.json /
# _config.json), e.g.:  make all FTLAB_ARTIFACTS=/path/to/build-a-tiny-llm

PY ?= python3
SB := storybyte
ANIMALS := dog cat bird bunny bear fish duck frog
LORA_RANKS := 1 2 4 8
EVAL_MODELS := base sft lora_r1 lora_r2 lora_r4 lora_r8 dpo nano_scratch nano_kd

.PHONY: all env data sft lora prefs dpo distill export eval verify clean help
.DEFAULT_GOAL := help

all: env data sft lora prefs dpo distill export eval verify ## Full pipeline, 00 -> 09

env: ## 00 — verify env + prove the base model port is exact
	cd $(SB) && $(PY) 00_env_check.py

data: ## 01 — synthesize the request->story dataset (2 rounds/animal, then assemble)
	cd $(SB) && for a in $(ANIMALS); do \
	  $(PY) 01_build_requests_dataset.py --animal $$a --round 1 || exit 1; \
	  $(PY) 01_build_requests_dataset.py --animal $$a --round 2 || exit 1; \
	done
	cd $(SB) && $(PY) 01_build_requests_dataset.py --assemble

sft: ## 02 — full SFT (450 steps; ships best-dev-probe checkpoint ~step 150)
	cd $(SB) && $(PY) 02_sft_full.py --steps 450

lora: ## 03 — LoRA rank ladder r=1,2,4,8 (450 steps each)
	cd $(SB) && for r in $(LORA_RANKS); do \
	  $(PY) 03_train_lora.py --rank $$r --steps 450 || exit 1; \
	done

prefs: ## 04 — mine DPO preference pairs (6 shards, then assemble)
	cd $(SB) && for i in 1 2 3 4 5 6; do \
	  $(PY) 04_build_preferences.py --shard $$i:6 || exit 1; \
	done
	cd $(SB) && $(PY) 04_build_preferences.py --assemble

dpo: ## 05 — DPO on the dialogue preference (ships at 25 steps, ~7s CPU)
	cd $(SB) && $(PY) 05_dpo.py --steps 25 --beta 0.2 --lr 1e-5

distill: ## 06 — knowledge distillation: KD student vs scratch control (1500 steps each)
	cd $(SB) && $(PY) 06_distill_kd.py --mode scratch --steps 1500
	cd $(SB) && $(PY) 06_distill_kd.py --mode kd --steps 1500

export: ## 07 — int8 quantize + export the web artifacts the course loads
	cd $(SB) && $(PY) 07_quantize_export.py

eval: ## 08 — run the eval suite for every model (3 gold slices + report)
	cd $(SB) && for m in $(EVAL_MODELS); do \
	  $(PY) 08_eval_suite.py --model $$m --part 1:3 || exit 1; \
	  $(PY) 08_eval_suite.py --model $$m --part 2:3 || exit 1; \
	  $(PY) 08_eval_suite.py --model $$m --part 3:3 || exit 1; \
	  $(PY) 08_eval_suite.py --model $$m --report || exit 1; \
	done

verify: ## 09 — NumPy-vs-PyTorch parity check on the shipped artifacts
	cd $(SB) && $(PY) 09_verify_web_artifacts.py

# The v1-v3 SFT task ablations and the 300-step DPO "overcooked" checkpoint are
# separate, deliberately-kept runs. Their recipes live in REPRODUCIBILITY.md;
# `make all` reproduces the shipping ladder. To regenerate the exact frozen
# numbers, keep the committed data/ and results/ and skip `make data`.

clean: ## remove derived heavy checkpoints (regenerable; kept out of git anyway)
	rm -f checkpoints/*_state.pt checkpoints/*_merged.npz checkpoints/_*_tmp.npz

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
