#!/usr/bin/env bash
# configs/benchmarks.yaml — standard benchmarks with lm-evaluation-harness 0.4.13 on one checkpoint. usage: launch/benchmarks.sh one <tag> <model_path_or_hub> [revision]
# Three invocations with the fixed few-shot counts: mmlu 5-shot; gsm8k 8-shot; the zero-shot / built-in set. Per-item logs via --log_samples. Outputs under $ICR_RESULTS_ROOT/benchmarks/<tag>/.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"; CONFIG=benchmarks
MODE=${1:?one}; TAG=${2:?tag}; MODEL=${3:?model}; REV=${4:-}; MAXBS=${MAXBS:-32}
OUT="$ICR_RESULTS_ROOT/benchmarks/$TAG"; mkdir -p "$OUT"
ARGS="pretrained=$MODEL,dtype=bfloat16,trust_remote_code=False"; [ -n "$REV" ] && ARGS="$ARGS,revision=$REV"
LM_EVAL="${ICR_LM_EVAL:-lm_eval}"
$LM_EVAL --model hf --model_args "$ARGS" --tasks mmlu --num_fewshot 5 --batch_size auto --max_batch_size $MAXBS --seed 1234 --log_samples --output_path "$OUT/mmlu"
$LM_EVAL --model hf --model_args "$ARGS" --tasks gsm8k --num_fewshot 8 --batch_size auto --max_batch_size $MAXBS --seed 1234 --log_samples --output_path "$OUT/gsm8k"
$LM_EVAL --model hf --model_args "$ARGS" --tasks arc_challenge,hellaswag,winogrande,piqa,lambada_openai,bbh_fewshot_dyck_languages,bbh_fewshot_word_sorting,bbh_fewshot_object_counting --batch_size auto --max_batch_size $MAXBS --seed 1234 --log_samples --output_path "$OUT/zero"
touch "$OUT/.done"; log "BENCHMARKS_DONE $TAG"
