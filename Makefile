# finetuning-lab - reproduce every number in the "Fine-Tuning LLMs" course.
# CPU only. `make reproduce` uses the committed frozen data; `make all` also
# regenerates sampled data first. Both run the numbered pipeline in order.
# See README.md and REPRODUCIBILITY.md for the recipes and expected numbers.
#
# Running outside the lets-data-science monorepo? Point FTLAB_ARTIFACTS at the
# StoryByte base artifacts (storybyte_weights.npz / _tokenizer_hf.json /
# _config.json), e.g.:  make all FTLAB_ARTIFACTS=/path/to/build-a-tiny-llm

PY ?= python3
FTLAB_ARTIFACTS ?=
FTLAB_WEB_DIR ?=
export FTLAB_ARTIFACTS FTLAB_WEB_DIR
SB := storybyte
ANIMALS := dog cat bird bunny bear fish duck frog
LORA_RANKS := 1 2 4 8
EVAL_MODELS := base sft lora_r1 lora_r2 lora_r4 lora_r8 dpo dpo_overcooked nano_scratch nano_kd
REPRO_STAGES := env sft lora prefs dpo distill eval export verify
FRESH_STAGES := env data sft lora prefs dpo distill eval export verify

.PHONY: all reproduce fresh commands env data sft lora prefs dpo distill export eval verify clean help
.DEFAULT_GOAL := help

all: fresh ## Full pipeline including fresh sampled data; metrics may drift slightly

reproduce: ## Rebuild the exact course ladder from committed frozen data, with wall time
	@start=$$(date +%s); \
	for stage in $(REPRO_STAGES); do \
	  $(MAKE) $$stage || exit $$?; \
	done; \
	end=$$(date +%s); \
	printf 'reproduce_wall_seconds=%s\n' $$((end - start))

fresh: ## Regenerate data, then rebuild every course artifact, with wall time
	@start=$$(date +%s); \
	for stage in $(FRESH_STAGES); do \
	  $(MAKE) $$stage || exit $$?; \
	done; \
	end=$$(date +%s); \
	printf 'fresh_wall_seconds=%s\n' $$((end - start))

commands: ## Print every exact reproduce command without running Python
	@for stage in $(REPRO_STAGES); do \
	  printf '\n# make %s\n' $$stage; \
	  $(MAKE) --no-print-directory -n $$stage || exit $$?; \
	done

env: ## 00 - verify env + prove the base model port is exact
	cd $(SB) && $(PY) 00_env_check.py

data: ## 01 - synthesize the request->story dataset (2 rounds/animal, then assemble)
	cd $(SB) && for a in $(ANIMALS); do \
	  $(PY) 01_build_requests_dataset.py --animal $$a --round 1 || exit 1; \
	  $(PY) 01_build_requests_dataset.py --animal $$a --round 2 || exit 1; \
	done
	cd $(SB) && $(PY) 01_build_requests_dataset.py --assemble

sft: ## 02 - full SFT (450 steps; ships best-dev-probe checkpoint ~step 150)
	cd $(SB) && $(PY) 02_sft_full.py --steps 450 --max-seconds 3600

lora: ## 03 - LoRA rank ladder r=1,2,4,8 (450 steps each)
	cd $(SB) && for r in $(LORA_RANKS); do \
	  $(PY) 03_train_lora.py --rank $$r --steps 450 --max-seconds 3600 || exit 1; \
	done

prefs: ## 04 - mine DPO preference pairs (6 shards, then assemble)
	cd $(SB) && for i in 1 2 3 4 5 6; do \
	  $(PY) 04_build_preferences.py --shard $$i:6 || exit 1; \
	done
	cd $(SB) && $(PY) 04_build_preferences.py --assemble --parts 6

dpo: ## 05 - DPO shipping (25 steps) + measured overcooked control (300 steps)
	cd $(SB) && $(PY) 05_dpo.py --steps 25 --beta 0.2 --lr 1e-5 --max-seconds 3600 --output-name dpo
	cd $(SB) && $(PY) 05_dpo.py --steps 300 --beta 0.2 --lr 1e-5 --max-seconds 3600 --output-name dpo_overcooked

distill: ## 06 - knowledge distillation: KD student vs scratch control (1500 steps each)
	cd $(SB) && $(PY) 06_distill_kd.py --mode scratch --steps 1500 --batch 32 --lr 6e-4 --max-seconds 3600
	cd $(SB) && $(PY) 06_distill_kd.py --mode kd --steps 1500 --batch 32 --lr 6e-4 --kd-temp 2.0 --alpha 0.5 --max-seconds 3600

export: ## 07 - int8 quantize + export the web artifacts the course loads
	cd $(SB) && $(PY) 07_quantize_export.py

eval: ## 08 - run the eval suite for every model (3 gold slices + report)
	cd $(SB) && for m in $(EVAL_MODELS); do \
	  $(PY) 08_eval_suite.py --model $$m --part 1:3 || exit 1; \
	  $(PY) 08_eval_suite.py --model $$m --part 2:3 || exit 1; \
	  $(PY) 08_eval_suite.py --model $$m --part 3:3 || exit 1; \
	  $(PY) 08_eval_suite.py --model $$m --report --parts 3 || exit 1; \
	done

verify: ## 09 - NumPy-vs-PyTorch parity check on the shipped artifacts
	cd $(SB) && $(PY) 09_verify_web_artifacts.py

# The v1-v3 SFT task ablations remain separate archived runs. `make reproduce`
# rebuilds the shipping ladder and 300-step DPO control from committed frozen
# data. `make all` regenerates sampled data, so metrics may drift.

clean: ## remove derived heavy checkpoints (regenerable; kept out of git anyway)
	rm -f checkpoints/*_state.pt checkpoints/*_merged.npz checkpoints/_*_tmp.npz

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'
