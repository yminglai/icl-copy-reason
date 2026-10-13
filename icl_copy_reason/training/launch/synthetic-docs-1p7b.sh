#!/usr/bin/env bash
# configs/synthetic-docs-1p7b.yaml — continued pretraining of SmolLM2-1.7B for 1B tokens; arms A B D; seed 0.
# usage: launch/synthetic-docs-1p7b.sh screen | validate | manifest | calibrate <lr> | pick-rate | train <arm>
# Recipe of record: 477 steps x 1,024 rows, lr 5e-4 (from 1.5e-4 / 5e-4 / 1.5e-3), save every 24, micro-batch 4, no probe graph; every checkpoint evaluated.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"; CONFIG=synthetic-docs-1p7b
MODE=${1:?mode}; RUN_ROOT="$ICR_RESULTS_ROOT/$CONFIG"; DATA="$RUN_ROOT/data"; mkdir -p "$DATA"
BASE=HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints; REV=step-1000000; RELEASED=HuggingFaceTB/SmolLM2-1.7B; PANELS=scan,elemental,miniscan,arith,chainmap
case $MODE in
  screen) for s in 4 5; do
      $PY "$ROOT/scripts/score_model.py" ablation --model $BASE --revision $REV --seed $s --n 160 --sets competitor_marker,wordB,sham --tag _screen --experiment $CONFIG/morph
      $PY "$ROOT/scripts/score_model.py" copy-share --model $BASE --revision $REV --seed $s --n 160 --tasks compose_h --tag _screen --experiment $CONFIG/morph; done;;
  validate) $PY "$ROOT/scripts/build_external_sets.py" score-panels --model $RELEASED --tasks $PANELS --tag _valid_$(basename $RELEASED) --out "$RUN_ROOT/morph/icl_valid_$(basename $RELEASED).json"
            $PY "$ROOT/scripts/build_external_sets.py" score-panels --model $BASE --revision $REV --tasks $PANELS --tag _valid_$(basename $BASE)_$REV --out "$RUN_ROOT/morph/icl_valid_$(basename $BASE)_$REV.json";;
  manifest) $PY "$ROOT/scripts/train.py" --config $CONFIG --mode corpus-manifest --seed 0;;
  calibrate) $PY "$ROOT/scripts/train.py" --config $CONFIG --mode calibrate --lr "${2:?lr}" --seed 0;;
  pick-rate) $PY "$ROOT/scripts/train.py" --config $CONFIG --mode pick-rate;;
  train)
    ARM=${2:?arm}; RUN="$RUN_ROOT/${ARM}_s0"
    $PY "$ROOT/scripts/train.py" --config $CONFIG --arm "$ARM" --seed 0
    for d in $(checkpoints_of "$RUN"); do [ -f "$d/.done" ] && [ ! -f "$d/.evaluated" ] && eval_checkpoint "$d" "_synthetic1p7b_${ARM}_s0_$(basename "$d")" $PANELS "$DATA"; done
    log "TRAIN_DONE $ARM";;
  *) echo "unknown mode $MODE"; exit 2;;
esac
