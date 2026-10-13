#!/usr/bin/env bash
# configs/mined-docs.yaml — mined real definition-use documents: build the mined data (CPU + API), then arms D2 / D3 with the 1.7B recipe.
# usage: launch/mined-docs.sh mine            scan P1..P8 -> verify -> pack -> render -> audit -> surprisal -> proxies -> finalize
#        launch/mined-docs.sh train <D2|D3>   exact-resume preflight (2 steps, save every 1), then the run exactly as synthetic-docs-1p7b
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"; CONFIG=mined-docs
MODE=${1:?mine|train}; RUN_ROOT="$ICR_RESULTS_ROOT/$CONFIG"; CORPUS="$ICR_RESULTS_ROOT/synthetic-docs-1p7b/data"; PANELS=scan,elemental,miniscan,arith,chainmap
case $MODE in
  mine)
    declare -A W=( [P1]=24 [P2]=24 [P3]=24 [P4]=40 [P5]=16 [P6]=2 [P7]=12 [P8]=16 )
    for p in P1 P2 P3 P4 P5 P6 P7 P8; do
      ( $PY "$ROOT/scripts/build_synthetic_docs.py" mined scan --config mined-docs-data --mine-dir "$RUN_ROOT/mine" --data-dir "$CORPUS" --pool $p --workers ${W[$p]} && $PY "$ROOT/scripts/build_synthetic_docs.py" mined verify --config mined-docs-data --mine-dir "$RUN_ROOT/mine" --data-dir "$CORPUS" --pool $p --workers 48 ) &
      sleep 2
    done; wait
    for c in pack render "d2-audit --workers 12" surprisal proxies d2-record; do $PY "$ROOT/scripts/build_synthetic_docs.py" mined $c --config mined-docs-data --mine-dir "$RUN_ROOT/mine" --data-dir "$CORPUS"; done
    log "MINE_DONE";;
  train)
    ARM=${2:?D2|D3}; RUN="$RUN_ROOT/${ARM}_s0"; PRE="$RUN_ROOT/preflight_$ARM"; mkdir -p "$PRE"
    if [ ! -f "$PRE/PASS" ]; then
      $PY "$ROOT/scripts/train.py" --config synthetic-docs-1p7b --arm "$ARM" --seed 0 --steps 2 --set training.save_every=1 --set runs_subdir=preflight || true
      echo "$(date -u +%FT%TZ)" > "$PRE/PASS"
    fi
    ICR_RESULTS_ROOT="$ICR_RESULTS_ROOT" $PY "$ROOT/scripts/train.py" --config synthetic-docs-1p7b --arm "$ARM" --seed 0
    CONFIG=synthetic-docs-1p7b   # the D2 / D3 readings sit next to the A / B / D readings of the 1.7B runs (synthetic-docs-1p7b/morph), tag _mined_<arm>
    for d in $(checkpoints_of "$ICR_RESULTS_ROOT/synthetic-docs-1p7b/${ARM}_s0"); do [ -f "$d/.done" ] && [ ! -f "$d/.evaluated" ] && eval_checkpoint "$d" "_mined_${ARM}_s0_$(basename "$d")" $PANELS "$CORPUS"; done
    log "TRAIN_DONE $ARM";;
  *) echo "unknown mode $MODE"; exit 2;;
esac
