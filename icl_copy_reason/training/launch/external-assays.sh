#!/usr/bin/env bash
# configs/external-assays.yaml — external in-context assays on one checkpoint. usage: launch/external-assays.sh one <tag> <model_path_or_hub> [max_batch=16] [revision]
#                                 launch/external-assays.sh screen <gpu-free>   headroom screen on the base / arm-A checkpoints (first 200 items)
#                                 launch/external-assays.sh chain <1p7B|360M>   base, A and B checkpoints of record of one scale
# (1) symbol-label / exchanged-label / natural-label in-context classification (sst2, agnews, subj; n 1000; seed 1234); (2) the BIG-bench tasks via lm_eval with the shipped task files.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"; CONFIG=external-assays
MODE=${1:?one|screen|chain}; LM_EVAL="${ICR_LM_EVAL:-lm_eval}"; TASKDIR="$ICR_DATA_ROOT/external/bigbench_tasks"
# outputs: <ICR_RESULTS_ROOT>/external-assays/full/<tag>/{sst2,agnews,subj}.jsonl + bb/ (what the report builder reads); screens under external-assays/screen_<tag>/ and external-assays/screen/<tag>/
one() { local TAG=$1 MODEL=$2 MAXBS=${3:-16} REV=${4:-}; local OUT="$ICR_RESULTS_ROOT/external-assays/full/$TAG"; mkdir -p "$OUT"; local REVARG=""; [ -n "$REV" ] && REVARG="--revision $REV"
  if [ "${SKIP_ICL:-0}" != 1 ]; then $PY "$ROOT/scripts/build_external_sets.py" score-near --config $CONFIG --model "$MODEL" $REVARG --tag "$TAG" --n 1000 --sets sst2,agnews,subj --out "$ICR_RESULTS_ROOT/external-assays/full"; fi
  local ARGS="pretrained=$MODEL,dtype=bfloat16,trust_remote_code=False"; [ -n "$REV" ] && ARGS="$ARGS,revision=$REV"
  $LM_EVAL --model hf --model_args "$ARGS" --tasks ext_modified_arithmetic,ext_operators --include_path "$TASKDIR" --batch_size auto --max_batch_size $MAXBS --seed 1234 --log_samples --output_path "$OUT/bb"
  touch "$OUT/.done"; log "EXTERNAL_ASSAYS_DONE $TAG"; }
case $MODE in
  one) one "${2:?tag}" "${3:?model}" "${4:-16}" "${5:-}";;
  screen) for m in "base_1p7B HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints step-1000000 16" "synthetic1p7b_A_s0 $ICR_RESULTS_ROOT/synthetic-docs-1p7b/A_s0/step477 - 16" \
                   "base_360M HuggingFaceTB/SmolLM2-360M-intermediate-checkpoints step-1600000 8" "synthetic360m_A_s0 $ICR_RESULTS_ROOT/synthetic-docs-360m/A_s0/step636 - 8"; do
      read -r TAG MODEL REV MAXBS <<< "$m"; REVARG=""; [ "$REV" != "-" ] && REVARG="--revision $REV"
      $PY "$ROOT/scripts/build_external_sets.py" score-near --config $CONFIG --model "$MODEL" $REVARG --tag "screen_$TAG" --n 200 --sets sst2,agnews,subj --out "$ICR_RESULTS_ROOT/external-assays"
      MA="pretrained=$MODEL,dtype=bfloat16"; [ "$REV" != "-" ] && MA="$MA,revision=$REV"
      $LM_EVAL --model hf --model_args $MA --tasks ext_modified_arithmetic,ext_operators,bigbench_symbol_interpretation_multiple_choice,ext_linguistics_puzzles,ext_conlang_translation --include_path "$TASKDIR" --limit 200 --batch_size auto --max_batch_size $MAXBS --seed 1234 --log_samples --output_path "$ICR_RESULTS_ROOT/external-assays/screen/$TAG"
    done;;
  chain) SCALE=${2:?1p7B|360M}
    if [ "$SCALE" = 1p7B ]; then one base_1p7B HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints 16 step-1000000; one synthetic1p7b_A_s0 "$ICR_RESULTS_ROOT/synthetic-docs-1p7b/A_s0/step477" 16; one synthetic1p7b_B_s0 "$ICR_RESULTS_ROOT/synthetic-docs-1p7b/B_s0/step477" 16
    else one base_360M HuggingFaceTB/SmolLM2-360M-intermediate-checkpoints 8 step-1600000; for a in A_s0 B_s0 A_s1 B_s1; do one synthetic360m_$a "$ICR_RESULTS_ROOT/synthetic-docs-360m/$a/step636" 8; done; fi;;
  *) echo "unknown mode $MODE"; exit 2;;
esac
