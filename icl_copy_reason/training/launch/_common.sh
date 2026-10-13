#!/usr/bin/env bash
# Shared helpers for the launchers: paths from ICR_* variables, one GPU, the standard evaluation card of a checkpoint.
# Sourced by every launch/<experiment>.sh; never run on its own.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export ICR_DATA_ROOT="${ICR_DATA_ROOT:-$ROOT/data}"
export ICR_RESULTS_ROOT="${ICR_RESULTS_ROOT:-$ROOT/runs}"
export HF_HOME="${ICR_HF_HOME:-${HF_HOME:-$HOME/.cache/huggingface}}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${ICR_PYTHON:-python}"
GPU="${ICR_GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
mkdir -p "$ICR_RESULTS_ROOT"
t0=$(date +%s)
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ) +$(( ($(date +%s) - t0) / 60 ))min] $*"; }
# standard card of one checkpoint: attention ablation, copy-probability share (word scheme + code scheme), real-language check, external panels, held-out loss
# usage: eval_checkpoint <checkpoint_dir> <tag> <panels: scan,elemental,miniscan[,arith,chainmap]> <data_dir for the loss> [--engagement heads.json]
# Every output lands in $ICR_RESULTS_ROOT/$CONFIG/morph/ (triage_/knockout_/gate_/icl_/loss files), where the report builders of the experiment read them.
eval_checkpoint() {
  local ck=$1 tag=$2 panels=$3 data=$4; shift 4
  $PY "$ROOT/scripts/score_model.py" ablation --model "$ck" --seed 2 --n 160 --sets competitor_marker,wordB,sham --tag "$tag" --experiment "$CONFIG/morph"
  $PY "$ROOT/scripts/score_model.py" copy-share --model "$ck" --seed 2 --n 160 --tasks compose_h --tag "$tag" --experiment "$CONFIG/morph"
  $PY "$ROOT/scripts/score_model.py" copy-share --model "$ck" --seed 2 --n 160 --tasks compose_h --scheme code --tag "${tag}_code" --experiment "$CONFIG/morph"
  $PY "$ROOT/scripts/score_model.py" quechua --model "$ck" --seed 2 --n 160 --tag "$tag" --experiment "$CONFIG/morph"
  $PY "$ROOT/scripts/build_external_sets.py" score-panels --model "$ck" --tasks "$panels" --tag "$tag" --out "$ICR_RESULTS_ROOT/$CONFIG/morph/icl${tag}.json"
  $PY "$ROOT/scripts/train.py" --config "$CONFIG" --mode eval-loss --model "$ck" --tag "$tag"
  if [ "${1:-}" = "--engagement" ]; then $PY "$ROOT/scripts/train.py" --config "$CONFIG" --mode engagement --model "$ck" --tag "$tag"; fi
  touch "$ck/.evaluated"
}
checkpoints_of() { ls -d "$1"/step* 2>/dev/null | sed 's/.*step//' | sort -n | sed "s#^#$1/step#"; }
