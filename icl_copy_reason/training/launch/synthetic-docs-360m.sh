#!/usr/bin/env bash
# configs/synthetic-docs-360m.yaml — continued pretraining of SmolLM2-360M with 3 % synthetic definition-use documents; arms A B B_control C C_rand; seeds 0 1.
# usage: launch/synthetic-docs-360m.sh prepare            heads + eligibility screen + corpus manifests (seeds 0,1) + rate calibration + lambda calibration
#        launch/synthetic-docs-360m.sh train <arm> <seed>  636 steps, checkpoint every 32, the evaluation card on every checkpoint
# Same hyper-parameters as the runs of record (lr 3e-3 selected from 3e-4 / 1e-3 / 3e-3; wd .01; warm-up 50; micro-batch 8; probe graph on).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"; CONFIG=synthetic-docs-360m
MODE=${1:?prepare|train}; RUN_ROOT="$ICR_RESULTS_ROOT/$CONFIG"; DATA="$RUN_ROOT/data"; mkdir -p "$DATA"
BASE=HuggingFaceTB/SmolLM2-360M-intermediate-checkpoints; REV=step-1600000
case $MODE in
  prepare)
    $PY "$ROOT/scripts/score_model.py" attribution --model $BASE --revision $REV --seed 3 --n 160 --tag _prep --experiment $CONFIG/morph
    $PY "$ROOT/scripts/train.py" --config $CONFIG --mode select-heads --dla "$ICR_RESULTS_ROOT/$CONFIG/morph/dla_$(basename $BASE)_${REV}_s3_prep.json"
    for s in 4 5; do   # eligibility screen of the base on two fresh item seeds
      $PY "$ROOT/scripts/score_model.py" ablation --model $BASE --revision $REV --seed $s --n 160 --sets competitor_marker,wordB,sham --tag _screen --experiment $CONFIG/morph
      $PY "$ROOT/scripts/score_model.py" copy-share --model $BASE --revision $REV --seed $s --n 160 --tasks compose_h --tag _screen --experiment $CONFIG/morph
    done
    for s in 0 1; do $PY "$ROOT/scripts/train.py" --config $CONFIG --mode corpus-manifest --seed $s; done
    for lr in 3e-4 1e-3 3e-3; do $PY "$ROOT/scripts/train.py" --config $CONFIG --mode calibrate --lr $lr --seed 0; done
    $PY "$ROOT/scripts/train.py" --config $CONFIG --mode pick-rate
    for w in sel rand; do $PY "$ROOT/scripts/train.py" --config $CONFIG --mode calibrate-lambda --which $w --seed 0; done
    log "PREPARE_DONE";;
  train)
    ARM=${2:?arm}; SEED=${3:?seed}; RUN="$RUN_ROOT/${ARM}_s$SEED"
    $PY "$ROOT/scripts/train.py" --config $CONFIG --arm "$ARM" --seed "$SEED"
    for d in $(checkpoints_of "$RUN"); do [ -f "$d/.done" ] && [ ! -f "$d/.evaluated" ] && eval_checkpoint "$d" "_synthetic360m_${ARM}_s${SEED}_$(basename "$d")" scan,elemental,miniscan "$DATA" --engagement; done
    log "TRAIN_DONE $ARM s$SEED ($(ls "$RUN"/step*/.evaluated | wc -l) checkpoints evaluated)";;
  *) echo "unknown mode $MODE"; exit 2;;
esac
