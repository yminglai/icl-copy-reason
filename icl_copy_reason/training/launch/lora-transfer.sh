#!/usr/bin/env bash
# configs/lora-transfer.yaml — LoRA recipe transfer on released 7-8B models. usage: launch/lora-transfer.sh data | run <base-key> [arms=Z,T,L] [seeds="0"|"0 1"]
# base-key = qwen | olmo2 | llama31 | comma | redpajama. Per base: calibration pack + 4-rate calibration once; then per (seed, arm): pack, train (r 16, alpha 32,
# micro-batch 4, global batch 32), merge, evaluation card with the scan / elemental / miniscan panels and the held-out text loss. Z = the base, evaluated once.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"; CONFIG=lora-transfer
MODE=${1:?data|run}; RUN_ROOT="$ICR_RESULTS_ROOT/$CONFIG"; DATA="$RUN_ROOT/data"
case $MODE in
  data) $PY "$ROOT/scripts/train.py" --config $CONFIG --mode build-data;;
  run)
    BKEY=${2:?base key}; ARMS=${3:-Z,T,L}; SEEDS=${4:-0}
    BASE=$($PY -c "import yaml;print(yaml.safe_load(open('$ROOT/configs/$CONFIG.yaml'))['bases']['$BKEY']['repo'])")
    [ -f "$RUN_ROOT/$BKEY/lr_sel.json" ] || $PY "$ROOT/scripts/train.py" --config $CONFIG --base $BKEY --mode calibrate
    for SEED in $SEEDS; do for ARM in ${ARMS//,/ }; do
      [ "$ARM" = Z ] && [ "$SEED" != 0 ] && continue
      TAG=_lora_${BKEY}_${ARM}$([ "$SEED" != 0 ] && echo "_s$SEED")
      if [ "$ARM" = Z ]; then M=$BASE; else
        $PY "$ROOT/scripts/train.py" --config $CONFIG --base $BKEY --arm $ARM --seed $SEED || { log "TRAIN FAILED $BKEY $ARM s$SEED"; continue; }
        M="$RUN_ROOT/$BKEY/${ARM}_s$SEED/merged"; fi
      $PY "$ROOT/scripts/score_model.py" ablation --model "$M" --seed 2 --n 160 --sets competitor_marker,wordB,sham --tag "$TAG" --experiment $CONFIG/morph
      $PY "$ROOT/scripts/score_model.py" copy-share --model "$M" --seed 2 --n 160 --tasks compose_h --tag "$TAG" --experiment $CONFIG/morph
      $PY "$ROOT/scripts/score_model.py" copy-share --model "$M" --seed 2 --n 160 --tasks compose_h --scheme code --tag "${TAG}_code" --experiment $CONFIG/morph
      $PY "$ROOT/scripts/score_model.py" quechua --model "$M" --seed 2 --n 160 --tag "$TAG" --experiment $CONFIG/morph
      $PY "$ROOT/scripts/build_external_sets.py" score-panels --model "$M" --tasks scan,elemental,miniscan --tag "$TAG" --out "$RUN_ROOT/morph/icl${TAG}_$([ "$ARM" = Z ] && echo base || echo merged).json"
      $PY "$ROOT/scripts/train.py" --config $CONFIG --base $BKEY --mode eval-loss --model "$M" --tag "$TAG"
    done; done; log "RUN_DONE $BKEY";;
  *) echo "unknown mode $MODE"; exit 2;;
esac
