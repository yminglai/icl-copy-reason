# From Copying to Composition — code and data

Code and data for the paper *From Copying to Composition: An Empirical Study on How Training Data Shapes In-Context Learning*
(anonymous, under review). A model shown a few made-up word forms and asked for a form it has never seen can either copy the
form it was shown or compose the correct new form from the rule. Two readings tell which one a model does: the
copy-probability share over four candidate forms, and an attention ablation of the demonstration lines that shows what the
answer depends on. The data-construction and training code builds the interventions of the paper: synthetic definition-use
documents, definition-use exercises rewritten from real documents, mined documents, and their admixture, fine-tune and
LoRA runs.

## Layout

```
icl_copy_reason/
  tasks/              diagnostic prompt generator (made-up endings, one combination withheld), item sets of record
  readings/           copy-probability share, attention ablation, deletion, activation patching, head analyses
  data_construction/  definition-use exercises, synthetic definition-use documents, mined documents, external label sets
  training/           continued pretraining with an admixture, fine-tuning, LoRA transfer, manifests, packing; launch/*.sh
  trajectories/       checkpoint registry and tracing of the two readings over a pretraining run
configs/              one YAML per experiment (model, data manifest, hyper-parameters, seeds)
scripts/              the seven command-line entry points
data/                 released data (see data/README.md)
tests/                one CPU smoke-test module
```

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .                  # torch, transformers, numpy, pandas, matplotlib, tqdm, pyyaml, scipy
pip install -e ".[train,api]"     # training extras and the OpenAI-compatible client
pip install -e ".[dev]" && pytest tests -q
```

Environment: `ICR_API_BASE` / `ICR_API_KEY` (OpenAI-compatible endpoint for the exercise generator and verifier; model names
are defaults in the configs), `ICR_DATA_ROOT` (default `data/`), `ICR_RESULTS_ROOT` (default `runs/`), `ICR_HF_HOME`
(weights cache), `ICR_GPU` (the GPU a launcher uses). Every CLI takes `--config <name>` and `--set key=value`.

## Running

```bash
# diagnostic prompts and the two readings of any Hugging Face causal LM
python scripts/make_prompts.py --show
python scripts/score_model.py copy-share --model Qwen/Qwen3-1.7B-Base --seed 0
python scripts/score_model.py ablation   --model Qwen/Qwen3-1.7B-Base --seed 0
python scripts/score_model.py patching   --model EleutherAI/pythia-1.4b --revision 143000 --seed 0 --position word --layers 4

# tracing a pretraining run (one config per family: python scripts/trace_checkpoints.py --list)
python scripts/trace_checkpoints.py --config traj-olmo2-7b

# building the training data
python scripts/build_synthetic_docs.py generate --mode compose --seed 0 --n 3 --version 3
python scripts/build_exercises.py generate --config exercises-data        # needs ICR_API_BASE / ICR_API_KEY
python scripts/build_external_sets.py label-maps

# training (dry run first; then the launcher of the experiment)
python scripts/train.py --config synthetic-docs-360m --arm B --seed 0 --dry-run
ICR_GPU=0 bash icl_copy_reason/training/launch/synthetic-docs-360m.sh train B 0
bash icl_copy_reason/training/launch/synthetic-docs-1p7b.sh train B
bash icl_copy_reason/training/launch/mined-docs.sh mine
bash icl_copy_reason/training/launch/rewritten-docs-admixture.sh train 1p7B E_M1 0
bash icl_copy_reason/training/launch/exercises-finetune.sh train F2
bash icl_copy_reason/training/launch/exercises-finetune-replication.sh S3B_s0 train F2
bash icl_copy_reason/training/launch/exercises-variants.sh pack T1_dose30
bash icl_copy_reason/training/launch/lora-transfer.sh run qwen Z,T,L "0"
bash icl_copy_reason/training/launch/heads-induce-train.sh plain
bash icl_copy_reason/training/launch/benchmarks.sh one my_tag runs/exercises-finetune/F2_s0/step232
bash icl_copy_reason/training/launch/external-assays.sh one my_tag runs/exercises-finetune/F2_s0/step232 16
```

Each launcher's header comment names its config and lists its sub-commands with their arguments.
Every job ran on a single 80 GB GPU; a run takes from under an hour to a few hours.

## Data

`data/` holds the diagnostic item sets, the 23,080 verified definition-use exercises (with the hash and locator of each
source document), samples of the synthetic documents, the label maps of the external evaluation sets, the training manifests
and the checkpoint registries. Source documents, token packs, checkpoints and per-item outputs are not included; the code
rebuilds them from the shipped records and the public sources. See `data/README.md`.

## Licence

Code: Apache License 2.0 (`LICENSE`). Data we created: CC BY 4.0; sources and their terms: `DATA_LICENSE.md`.

## Citation

```bibtex
@unpublished{anonymous2026copying,
  title  = {From Copying to Composition: An Empirical Study on How Training Data Shapes In-Context Learning},
  author = {Anonymous},
  note   = {Under review},
  year   = {2026}
}
```
