#!/usr/bin/env bash
# configs/heads-induce-train.yaml — direct fine-tuning of Pythia-1.4B on alternate-template prompts.
# usage: launch/heads-induce-train.sh plain       compose s0, compose s1, control s0 (2,000 steps, save every 250); every saved step scored on item seeds 2 and 3
#        launch/heads-induce-train.sh fine        compose s0 / s1 with the fine save grid 0,10,...,200
#        launch/heads-induce-train.sh guided      arms A B C B_rand C_rand x train seeds 0 / 1 (select seeds 3 / 2, eval seeds 2 / 3); per step copy-share + attribution; at 200 ablation + real-language check
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"; CONFIG=heads-induce-train
MODE=${1:?plain|fine|guided}; RUN_ROOT="$ICR_RESULTS_ROOT/heads-induce/induce"   # configs/heads-induce-train.yaml: results_dir heads-induce, runs_subdir induce
score_plain() { local d=$1 tag=$2; for s in 2 3; do
  $PY "$ROOT/scripts/score_model.py" ablation --model "$d" --seed $s --n 160 --tag "$tag" --experiment heads-induce
  $PY "$ROOT/scripts/score_model.py" copy-share --model "$d" --seed $s --n 160 --tasks compose_h --tag "$tag" --experiment heads-induce; done; }
case $MODE in
  plain) for run in compose:0 compose:1 control:0; do arm=${run%%:*}; seed=${run##*:}
      $PY "$ROOT/scripts/train.py" --config $CONFIG --arm $arm --seed $seed
      for k in 0 250 500 750 1000 1250 1500 1750 2000; do score_plain "$RUN_ROOT/${arm}_s${seed}/step$k" "_ind_${arm}_s${seed}_step$k"; done; done;;
  fine) for seed in 0 1; do
      $PY "$ROOT/scripts/train.py" --config $CONFIG --arm compose_fine --seed $seed --out-suffix _fine
      for k in 0 10 20 30 40 50 75 100 150 200; do score_plain "$RUN_ROOT/compose_s${seed}_fine/step$k" "_indfine_compose_s${seed}_step$k"; done; done;;
  guided)
    ACCUM=1; $PY "$ROOT/scripts/train.py" --config $CONFIG --arm B --seed 0 --select-seed 3 --steps 1 --dla "$ICR_RESULTS_ROOT/heads-steer/dla_pythia-1.4b_step143000_s3.json" || ACCUM=4   # memory smoke decides the accumulation fallback
    for ts in 0 1; do sel=$(( ts == 0 ? 3 : 2 )); ev=$(( ts == 0 ? 2 : 3 )); DLA="$ICR_RESULTS_ROOT/heads-steer/dla_pythia-1.4b_step143000_s$sel.json"
      for arm in A B C B_rand C_rand; do
        $PY "$ROOT/scripts/train.py" --config $CONFIG --arm $arm --seed $ts --select-seed $sel --accum $ACCUM --dla "$DLA"
        R="$RUN_ROOT/guided_${arm}_s${ts}_f${sel}"
        for k in 0 10 20 30 40 50 75 100 150 200; do
          $PY "$ROOT/scripts/score_model.py" copy-share --model "$R/step$k" --seed $ev --n 160 --tasks compose_h --tag "_guided_${arm}_s${ts}_step$k" --experiment heads-induce
          $PY "$ROOT/scripts/score_model.py" attribution --model "$R/step$k" --seed $ev --n 160 --tag "_guided_${arm}_s${ts}_step$k"; done
        $PY "$ROOT/scripts/score_model.py" ablation --model "$R/step200" --seed $ev --n 160 --sets competitor_marker,wordB --tag "_guided_${arm}_s${ts}_step200" --experiment heads-induce
        $PY "$ROOT/scripts/score_model.py" quechua --model "$R/step200" --seed $ev --n 160 --tag "_guided_${arm}_s${ts}" --experiment heads-induce
      done; done;;
  *) echo "unknown mode $MODE"; exit 2;;
esac
log "DONE $MODE"
