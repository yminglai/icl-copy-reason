#!/usr/bin/env bash
# configs/rewritten-docs-admixture.yaml — rewritten real documents as an admixture, staged training with the continuation rule.
# usage: launch/rewritten-docs-admixture.sh build                                   sample -> generate -> verify -> audit -> pack (both scales)
#        launch/rewritten-docs-admixture.sh train <360M|1p7B> <E_M1|E_M2|E_src> <seed>   stage by stage; the OOD block at every boundary; near block at the end
#        launch/rewritten-docs-admixture.sh final <360M|1p7B> <arm> <seed>              external assays + far assays (+ benchmarks at 1.7B) on the last checkpoint
#        launch/rewritten-docs-admixture.sh comparators <360M|1p7B>                     far assays + bbh pair on the base / A / B checkpoints of record
# Recipe of record per scale: 360M lr 3e-3, 636 x 768 rows, save 32, micro-batch 8, stages 320 / 636; 1.7B lr 5e-4, 477 x 1,024, save 24, micro-batch 4, stages 120 / 240 / 360 / 477.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"; CONFIG=rewritten-docs-admixture
MODE=${1:?mode}; SCALE=${2:-1p7B}; ARM=${3:-E_M1}; SEED=${4:-0}
case $SCALE in 360M) CORPUS_EXP=synthetic-docs-360m; LR=3e-3; STEPS=636; SAVE=32; MBS=8; STAGES="320 636"; MAXBS=8; PANELS=scan,elemental,miniscan;;
               1p7B) CORPUS_EXP=synthetic-docs-1p7b; LR=5e-4; STEPS=477; SAVE=24; MBS=4; STAGES="120 240 360 477"; MAXBS=16; PANELS=scan,elemental,miniscan,arith,chainmap;; *) echo "scale 360M|1p7B"; exit 2;; esac
RUN="$ICR_RESULTS_ROOT/$CONFIG/runs/$SCALE/${ARM}_s$SEED"; CORPUS="$ICR_RESULTS_ROOT/$CORPUS_EXP/data"   # run directories: <experiment>/runs/<scale>/<arm>_s<seed> (what the report builder reads)
ood_block() { # OOD suite at a boundary (far assays + composition panels + the bbh pair); near=1 adds the symbol / exchanged-label block
  local tag=$1 ck=$2 near=$3
  $PY "$ROOT/scripts/build_external_sets.py" assays --config $CONFIG --scale $SCALE --model "$ck" --tag "$tag" --near "$near" --max-batch $MAXBS
  "${ICR_LM_EVAL:-lm_eval}" --model hf --model_args "pretrained=$ck,dtype=bfloat16,trust_remote_code=False" --tasks bbh_fewshot_dyck_languages,bbh_fewshot_word_sorting --batch_size auto --max_batch_size $MAXBS --seed 1234 --log_samples --output_path "$ICR_RESULTS_ROOT/$CONFIG/ood/$tag/bbh"
}
case $MODE in
  build) $PY "$ROOT/scripts/build_exercises.py" sample --config rewritten-docs-data --target 60000 --workers 32
         $PY "$ROOT/scripts/build_exercises.py" generate --config rewritten-docs-data --workers 96
         $PY "$ROOT/scripts/build_exercises.py" verify --config rewritten-docs-data --workers 128
         $PY "$ROOT/scripts/build_exercises.py" audit --config rewritten-docs-data --n 300 --workers 16
         for M in M1 M2; do for S in 0 1; do $PY "$ROOT/scripts/train.py" --config $CONFIG --mode pack --mixture $M --seed $S --corpus-dir "$ICR_RESULTS_ROOT/synthetic-docs-360m/data" --set model.base=HuggingFaceTB/SmolLM2-360M-intermediate-checkpoints --set model.revision=step-1600000; done
           $PY "$ROOT/scripts/train.py" --config $CONFIG --mode pack --mixture $M --seed 0 --corpus-dir "$ICR_RESULTS_ROOT/synthetic-docs-1p7b/data" --set model.base=HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints --set model.revision=step-1000000; done;;
  train)
    PREV=0
    for S in $STAGES; do
      $PY "$ROOT/scripts/train.py" --config $CONFIG --arm "$ARM" --seed "$SEED" --steps "$S" --set model.base="$($PY -c "import yaml;print(yaml.safe_load(open('$ROOT/configs/$CONFIG.yaml'))['scales']['$SCALE']['base'])")" \
          --set model.revision="$($PY -c "import yaml;print(yaml.safe_load(open('$ROOT/configs/$CONFIG.yaml'))['scales']['$SCALE']['revision'])")" --set training.lr_selected=$LR --set training.save_every=$SAVE --set training.micro_bs=$MBS --set data.rows_per_step=$([ $SCALE = 360M ] && echo 768 || echo 1024) --set runs_subdir=runs/$SCALE
      for d in $(checkpoints_of "$RUN"); do [ -f "$d/.done" ] && [ ! -f "$d/.evaluated" ] && eval_checkpoint "$d" "_rewritten_${SCALE}_${ARM}_s${SEED}_$(basename "$d")" $PANELS "$CORPUS"; done
      NEAR=0; [ "$S" = "$STEPS" ] && NEAR=1; TAG=rewritten_${SCALE}_${ARM}_s${SEED}_step$S
      ood_block "$TAG" "$RUN/step$S" $NEAR; touch "$RUN/step$S/.ood_done"
      # stage boundary: apply the continuation rule of the config to the OOD block just written; write "STOPPED_AT $S" to "$RUN/stopped.txt" and break to stop
      PREV=$S
    done; log "TRAIN_DONE $SCALE $ARM s$SEED last boundary $PREV";;
  final) CK="$RUN/step$STEPS"; TAG=rewritten_${SCALE}_${ARM}_s${SEED}_step$STEPS
         bash "$(dirname "${BASH_SOURCE[0]}")/external-assays.sh" one "$TAG" "$CK" $MAXBS
         $PY "$ROOT/scripts/build_external_sets.py" score-far --config $CONFIG --model "$CK" --tag "$TAG" --n 1000 --out "$ICR_RESULTS_ROOT/$CONFIG/far"
         [ "$SCALE" = 1p7B ] && bash "$(dirname "${BASH_SOURCE[0]}")/benchmarks.sh" one "$TAG" "$CK"; log "FINAL_DONE $TAG";;
  comparators)   # far assays + bbh pair on the base / A / B checkpoints of record of the scale
    if [ "$SCALE" = 1p7B ]; then LIST="base_1p7B:$ICR_RESULTS_ROOT/synthetic-docs-1p7b/A_s0/step0 synthetic1p7b_A_s0:$ICR_RESULTS_ROOT/synthetic-docs-1p7b/A_s0/step477 synthetic1p7b_B_s0:$ICR_RESULTS_ROOT/synthetic-docs-1p7b/B_s0/step477"
    else LIST="base_360M:$ICR_RESULTS_ROOT/synthetic-docs-360m/A_s0/step0 synthetic360m_A_s0:$ICR_RESULTS_ROOT/synthetic-docs-360m/A_s0/step636 synthetic360m_B_s0:$ICR_RESULTS_ROOT/synthetic-docs-360m/B_s0/step636"; fi
    for item in $LIST; do ood_block "${item%%:*}" "${item#*:}" 0; done;;
  *) echo "unknown mode $MODE"; exit 2;;
esac
