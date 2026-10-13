#!/usr/bin/env bash
# configs/heads-attention.yaml (readings side) — the attention-signature measurements on (1) the 360M base + every checkpoint of the ten synthetic-docs-360m runs,
# (2) the stage-A finals of the family survey, (3) the traced checkpoints of two families; item seeds 2 and 3, n 160, bf16, eager attention.
# usage: launch/heads-attention.sh [sets=runs,finals,traj]   Reads the registry (data/manifests/revisions/) and the run directories under $ICR_RESULTS_ROOT.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"; CONFIG=heads-attention
SETS=${1:-runs,finals,traj}
measure() { $PY "$ROOT/scripts/score_model.py" attention --model "$1" ${2:+--revision $2} --seeds "${4:-2,3}" --n 160 --bs 8 --tag "$3"; }
case ",$SETS," in *,runs,*)
  measure HuggingFaceTB/SmolLM2-360M-intermediate-checkpoints step-1600000 _attention_synthetic360m_base
  for run in A_s0 A_s1 B_s0 B_s1 B_control_s0 B_control_s1 C_s0 C_s1 C_rand_s0 C_rand_s1; do for d in $(checkpoints_of "$ICR_RESULTS_ROOT/synthetic-docs-360m/$run"); do measure "$d" "" "_attention_synthetic360m_${run}_$(basename "$d")"; done; done;; esac
case ",$SETS," in *,finals,*) $PY "$ROOT/scripts/trace_checkpoints.py" --list-finals | while read -r NAME REPO; do measure "$REPO" "" "_attention_final_$NAME" 2; done;; esac
case ",$SETS," in *,traj,*) for fam in olmo3-7b comma-7b; do $PY "$ROOT/scripts/trace_checkpoints.py" --config traj-$fam --list-revisions | while read -r REF; do measure "$($PY "$ROOT/scripts/trace_checkpoints.py" --config traj-$fam --repo)" "$REF" "_attention_traj_${fam}_$REF"; done; done;; esac
log "ATTENTION_DONE $SETS"
