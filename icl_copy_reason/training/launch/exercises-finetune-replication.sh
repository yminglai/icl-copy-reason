#!/usr/bin/env bash
# configs/exercises-finetune-replication.yaml — the exercise fine-tune replicated: block 1p7B_s1 (seed 1, new replay regions) and block S3B_s0 (SmolLM3-3B, own calibration).
# usage: launch/exercises-finetune-replication.sh <1p7B_s1|S3B_s0> <mode> [args]   modes exactly as exercises-finetune.sh (data | calibrate | calibrate-chain | pick-rate | train | final)
# Block S3B_s0 first re-tokenises the corpus (corpus-manifest with uint32) and re-packs the exercises; micro-batch 1, validation batch 4, benchmark block of the control skipped (pre-authorised cut).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
BLOCK=${1:?1p7B_s1|S3B_s0}; shift
case $BLOCK in
  1p7B_s1) export ICR_SEED=1 ICR_BLOCK=1p7B_s1 ICR_SCALE=1p7B; export BLOCK_ARGS="--block 1p7B_s1 --replay0 305152 --ctrl0 334848";;
  S3B_s0)  export ICR_SEED=0 ICR_BLOCK=S3B_s0 ICR_SCALE=S3B ICR_FAMILY_SCALE=1p7B; export BLOCK_ARGS="--block S3B_s0"; export ICR_CORPUS_DIR="$ICR_RESULTS_ROOT/exercises-finetune-replication/S3B_s0/data"
           if [ "${1:-}" = corpus ]; then $PY "$ROOT/scripts/train.py" --config exercises-finetune-replication --block S3B_s0 --mode corpus-manifest --seed 0 --set data.token_dtype=uint32 --set data.corpus=smollm17 --set data.steps=477; exit 0; fi;;
  *) echo "unknown block $BLOCK"; exit 2;;
esac
CONFIG=exercises-finetune-replication exec bash "$(dirname "${BASH_SOURCE[0]}")/exercises-finetune.sh" "$@"
