#!/usr/bin/env bash
# configs/exercises-variants.yaml — the eight-variant exercise loop (development history): one 120-step admixture per variant, evaluated at the end only.
# usage: launch/exercises-variants.sh pack <variant>     new exercise pack for the variant (types / density / cycling as in the config)
#        launch/exercises-variants.sh train <variant>    manifest -> 120 steps (lr 5e-4, warm-up 50, constant) -> fingerprint card, OOD + near block, unseen DEV datasets, DEV-only aggregation
#        launch/exercises-variants.sh comparator         score the A@120 comparator exactly as the loop scores a variant
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"; CONFIG=exercises-variants
MODE=${1:?mode}; V=${2:-}; RUN_ROOT="$ICR_RESULTS_ROOT/$CONFIG"; DATA="$RUN_ROOT/data"; CORPUS="$ICR_RESULTS_ROOT/synthetic-docs-1p7b/data"; PANELS=scan,elemental,miniscan,arith,chainmap; mkdir -p "$DATA"
case $MODE in
  pack) $PY "$ROOT/scripts/train.py" --config $CONFIG --mode pack --variant "$V" --seed 0 --corpus-dir "$CORPUS" --out "$DATA";;
  train)
    $PY "$ROOT/scripts/train.py" --config $CONFIG --mode row-maps --arm "$V" --seed 0 --corpus-dir "$CORPUS"
    $PY "$ROOT/scripts/train.py" --config $CONFIG --arm "$V" --seed 0
    CK="$RUN_ROOT/${V}_s0/step120"; TAG=exercises_1p7B_${V}_s0_step120
    eval_checkpoint "$CK" "_$TAG" $PANELS "$CORPUS"
    $PY "$ROOT/scripts/build_external_sets.py" assays --config exercises-finetune --scale 1p7B --model "$CK" --tag "$TAG" --near 1 --max-batch 16
    $PY "$ROOT/scripts/build_external_sets.py" score-labels --config $CONFIG --model "$CK" --tag "$TAG" --n 1000 --sets dbpedia7,yahoo5 --out "$ICR_RESULTS_ROOT/exercises-finetune/icl"
    log "VARIANT_DONE $V";;
  comparator) CK="$ICR_RESULTS_ROOT/synthetic-docs-1p7b/A_s0/step120"; TAG=synthetic1p7b_A_s0_step120
    $PY "$ROOT/scripts/build_external_sets.py" assays --config exercises-finetune --scale 1p7B --model "$CK" --tag "$TAG" --near 1 --max-batch 16
    bash "$(dirname "${BASH_SOURCE[0]}")/external-assays.sh" one "$TAG" "$CK" 16
    $PY "$ROOT/scripts/build_external_sets.py" score-labels --config exercises-finetune --model "$CK" --tag "$TAG" --n 1000 --sets trec6,dbpedia7,yahoo5 --out "$ICR_RESULTS_ROOT/exercises-finetune/icl";;
  *) echo "unknown mode $MODE"; exit 2;;
esac
