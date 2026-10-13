#!/usr/bin/env bash
# configs/exercises-finetune.yaml — exercises as an admixture (F1 / F1_ctrl) and as a fine-tune with replay (F2 / F2_ctrl) at 1.7B, seed 0.
# usage: launch/exercises-finetune.sh data                      pack the exercise files (F1, F2) + write the per-row manifests + overlap check
#        launch/exercises-finetune.sh calibrate <calib|calib_ctrl> <lr>   64-step rate calibration run
#        launch/exercises-finetune.sh calibrate-chain <calib|calib_ctrl>  all three rates in order
#        launch/exercises-finetune.sh pick-rate                 the calibration rule on both sweeps
#        launch/exercises-finetune.sh train <F1|F1_ctrl|F2|F2_ctrl>   to the fixed end; evaluation card at the configured steps; OOD at the boundaries; end blocks
#        launch/exercises-finetune.sh final <arm>               re-run the end blocks on the end checkpoint
#        launch/exercises-finetune.sh comparators               the unseen-symbol screens and the comparator-of-record scoring (base, A@477, A@240, A@120)
# Recipe of record: F1 240 steps x 1,024 rows lr 5e-4 warm-up 50 constant save 24; F2 232 x 256 lr 3e-4 (calibrated) warm-up 20 cosine to zero save 58; micro-batch 4.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"; CONFIG=${CONFIG:-exercises-finetune}; BLOCK_ARGS=${BLOCK_ARGS:-}
MODE=${1:?mode}; SEED=${ICR_SEED:-0}; SCALE=${ICR_SCALE:-1p7B}; RUN_ROOT="$ICR_RESULTS_ROOT/$CONFIG${ICR_BLOCK:+/$ICR_BLOCK}"; DATA="$RUN_ROOT/data"; CORPUS="${ICR_CORPUS_DIR:-$ICR_RESULTS_ROOT/synthetic-docs-1p7b/data}"; PANELS=scan,elemental,miniscan,arith,chainmap
mkdir -p "$DATA"
arm_plan() { $PY "$ROOT/scripts/train.py" --config $CONFIG --arm "$1" --seed $SEED $BLOCK_ARGS --dry-run; }
# tags follow the released scheme exercises_<scale>_<arm>_s<seed>_step<k>; readings -> $CONFIG/morph, unseen-symbol sets -> $CONFIG/icl/<tag>/, far assays -> rewritten-docs-admixture/far/<tag>/
unseen_sets() { $PY "$ROOT/scripts/build_external_sets.py" score-labels --config $CONFIG --model "$1" --tag "$2" --sets trec6,dbpedia7,yahoo5 --n "${3:-1000}" --out "$ICR_RESULTS_ROOT/$CONFIG/icl"; }
bbh_pair() { "${ICR_LM_EVAL:-lm_eval}" --model hf --model_args "pretrained=$1,dtype=bfloat16,trust_remote_code=False" --tasks bbh_fewshot_dyck_languages,bbh_fewshot_word_sorting --batch_size auto --max_batch_size 16 --seed 1234 --log_samples --output_path "$ICR_RESULTS_ROOT/rewritten-docs-admixture/ood/$2/bbh"; }
end_blocks() { local ck=$1 tag=$2
  bash "$(dirname "${BASH_SOURCE[0]}")/benchmarks.sh" one "$tag" "$ck"
  unseen_sets "$ck" "$tag" 1000; }
case $MODE in
  data) for arm in F1 F2; do $PY "$ROOT/scripts/train.py" --config $CONFIG --mode pack --arm $arm --seed $SEED --corpus-dir "$CORPUS" --out "$DATA" $BLOCK_ARGS; done
        $PY "$ROOT/scripts/train.py" --config $CONFIG --mode row-maps --seed $SEED --corpus-dir "$CORPUS" $BLOCK_ARGS
        $PY "$ROOT/scripts/train.py" --config $CONFIG --mode overlap-check --corpus-dir "$CORPUS" --manifests "$DATA"/manifest_calib_s$SEED.json,"$DATA"/manifest_calibctrl_s$SEED.json $BLOCK_ARGS;;
  calibrate) $PY "$ROOT/scripts/train.py" --config $CONFIG --mode calibrate --arm "${2:?calib|calib_ctrl}" --lr "${3:?lr}" --seed $SEED $BLOCK_ARGS;;
  calibrate-chain) for lr in 3e-5 1e-4 3e-4; do $PY "$ROOT/scripts/train.py" --config $CONFIG --mode calibrate --arm "${2:?calib|calib_ctrl}" --lr $lr --seed $SEED $BLOCK_ARGS; done;;
  pick-rate) $PY "$ROOT/scripts/train.py" --config $CONFIG --mode pick-rate $BLOCK_ARGS;;
  train|final)
    ARM=${2:?arm}; RUN="$RUN_ROOT/${ARM}_s$SEED"; PLAN=$(arm_plan "$ARM")
    STEPS=$(echo "$PLAN" | $PY -c "import sys,json;print(json.load(sys.stdin)['steps'])"); EVAL_AT=$($PY -c "import yaml;a=yaml.safe_load(open('$ROOT/configs/$CONFIG.yaml'))['arms']['$ARM'];print(' '.join(map(str,a.get('eval_at','all'))) if a.get('eval_at')!='all' else 'all')"); OOD_AT=$($PY -c "import yaml;a=yaml.safe_load(open('$ROOT/configs/$CONFIG.yaml'))['arms']['$ARM'];print(' '.join(map(str,a.get('ood_at',[]))))")
    [ "$MODE" = train ] && $PY "$ROOT/scripts/train.py" --config $CONFIG --arm "$ARM" --seed $SEED $BLOCK_ARGS
    for d in $(checkpoints_of "$RUN"); do k=$(basename "$d" | sed s/step//); if [ "$EVAL_AT" = all ] || [[ " $EVAL_AT " == *" $k "* ]]; then [ -f "$d/.done" ] && [ ! -f "$d/.evaluated" ] && eval_checkpoint "$d" "_exercises_${SCALE}_${ARM}_s${SEED}_step$k" $PANELS "$CORPUS"; fi; done
    for k in $OOD_AT; do NEAR=0; [ "$k" = "$STEPS" ] && NEAR=1; TAG="exercises_${SCALE}_${ARM}_s${SEED}_step$k"
      $PY "$ROOT/scripts/build_external_sets.py" assays --config $CONFIG --scale ${ICR_FAMILY_SCALE:-1p7B} --model "$RUN/step$k" --tag "$TAG" --near $NEAR --max-batch 16
      bbh_pair "$RUN/step$k" "$TAG"; done
    end_blocks "$RUN/step$STEPS" "exercises_${SCALE}_${ARM}_s${SEED}_step$STEPS"; log "${MODE^^}_DONE $ARM s$SEED end=$STEPS";;
  comparators)   # the unseen-symbol screens (first 200 items) and the comparators of record: base, A@477, A@240, A@120 of the 1.7B synthetic-document run
    A="$ICR_RESULTS_ROOT/synthetic-docs-1p7b/A_s0"
    unseen_sets "$A/step0" "screen_base_1p7B" 200; unseen_sets "$A/step477" "screen_synthetic1p7b_A_s0_step477" 200
    for ck in step0 step477 step240 step120; do TAG=synthetic1p7b_A_s0_$ck; [ "$ck" = step0 ] && TAG=base_1p7B
      $PY "$ROOT/scripts/build_external_sets.py" assays --config $CONFIG --scale 1p7B --model "$A/$ck" --tag "$TAG" --near 1 --max-batch 16
      bbh_pair "$A/$ck" "$TAG"; unseen_sets "$A/$ck" "$TAG" 1000; done;;
  *) echo "unknown mode $MODE"; exit 2;;
esac
