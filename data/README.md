# Data

Data we created is CC BY 4.0; third-party sources and their terms are listed in `../DATA_LICENSE.md`.

## diagnostic/
The item sets of record of the diagnostic: 160 items per task and item seed, the panels used by the ablation, deletion and
patching analyses, the alternate presentations used as controls, plus `templates.json` and `pools.json`. Regenerate with
`python scripts/make_prompts.py`.

## exercises/
`exercises.jsonl.gz`: the verified definition-use exercises written by a language model from real documents (one record per
exercise: the exercise, its type, the generator and verifier verdicts, and the source document's SHA-256 with a pool / file /
row locator). `prompts.json`, `rewrite_prompts.json`, `variant_prompts.json`: the generator, verifier and auditor prompts.
`banned_labels.json`, `source_pools.json`, `examples.json`: the label words excluded from exercises, the document pools with
pinned revisions, and one example per type. Source texts are not included: fetch them from the pool at the pinned revision and
check the hash.

## synthetic/
Generator configuration and 1,000-document samples of the synthetic definition-use documents (compose mode and the
every-combination-shown control, two generator versions), plus the small records of the mined-document arms. The generator is
deterministic: `python scripts/build_synthetic_docs.py generate --mode compose --seed 0 --n 5 --version 3`.

## external/
`label_maps.json`: source dataset, kept classes, label names, symbols and exchange map of every symbol-label / exchanged-label
evaluation set; the sets are built on demand from the Hub (`python scripts/build_external_sets.py build-labels`). `scan/` and
`elemental/` ship with their licences; BIG-bench tasks are fetched on demand.

## manifests/
What every training run read: corpus manifests (base model, corpus mixture, rows, hashes of the token files), row-map manifests
of the exercise fine-tunes, pack and data records, the configuration of every run, the rate calibrations, and `revisions/`, the
registry of released checkpoints per traced family. Token packs, checkpoints and per-item outputs are rebuilt from these records
with `scripts/train.py` and the launchers.
