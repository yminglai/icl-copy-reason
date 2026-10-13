# Data licence and attribution

## Data created for this paper — CC BY 4.0

The following files were produced by us and are released under the Creative Commons Attribution 4.0 International
licence (https://creativecommons.org/licenses/by/4.0/): the diagnostic item sets, templates and word pools
(`data/diagnostic/`), the definition-use exercises and their verifier verdicts (`data/exercises/`), the synthetic
definition-use documents and their generator configuration (`data/synthetic/`), the label maps and builder inputs
for the external evaluation sets (`data/external/`), the training manifests (`data/manifests/`), and every released
result table (`data/results/`). Attribution: "Anonymous. From Copying to Composition: An Empirical Study on How
Training Data Shapes In-Context Learning. Under review." (to be replaced by the published citation).

The exercises are new text written by a language model from a source document; each record carries the SHA-256 of
its source document and a locator (pool, file, row) but not the source text itself. Source texts are NOT
redistributed here; `data/README.md` says how to fetch them from the public pools below and verify the hashes.

## Sources used by the data-construction code

| source | used for | licence / terms (as stated by the source) |
|---|---|---|
| HuggingFaceTB/smollm-corpus, `fineweb-edu-dedup` | pretraining mixture for the admixture runs | ODC-By 1.0 |
| HuggingFaceTB/smollm-corpus, `cosmopedia-v2` | pretraining mixture for the admixture runs | Apache-2.0 |
| HuggingFaceFW/fineweb-edu (CC-MAIN-2025-26) | document pool P4; held-out validation text | ODC-By 1.0 (plus CommonCrawl terms) |
| HuggingFaceTB/finemath (`finemath-4plus`) | document pool P3 | ODC-By 1.0 |
| EleutherAI/proof-pile-2 (`arxiv`, `open-web-math`) | document pools P1, P2 | per-component: arXiv articles under their own arXiv licences; OpenWebMath ODC-By 1.0 |
| marin-community/ar5iv-no-problem-markdown | document pool P5 | arXiv articles under their own licences (ar5iv conversion) |
| pile-of-law/pile-of-law (`eurlex`) | document pool P6 | CC BY 4.0 (EUR-Lex reuse policy) |
| marin-community/stackexchange-markdown | document pool P7 | CC BY-SA 4.0 (Stack Exchange) |
| marin-community/wikipedia-markdown | document pool P8 | CC BY-SA 4.0 (Wikipedia) |
| monology/pile-uncopyrighted (`val.jsonl.zst`) | held-out validation loss | MIT (dataset card); component sources' own terms |
| CogComp/trec (TREC-6) | external evaluation set builder | free for research use (TREC question classification data) |
| fancyzhx/dbpedia_14 (DBpedia-14, 7 classes used) | external evaluation set builder | CC BY-SA 3.0 (DBpedia) |
| community-datasets/yahoo_answers_topics (5 classes used) | external evaluation set builder | Yahoo! Webscope terms: non-commercial research use, no redistribution — we ship label maps and the builder only |
| SetFit/20_newsgroups | banned-label bookkeeping / near assay | free for research use |
| SetFit/sst2 | near in-context assay | free for research use (Stanford Sentiment Treebank) |
| facebook bAbI (tasks 1, 20) | near / far in-context assays | BSD licence (Facebook AI Research) |
| hails/bigbench (`list_functions`, `modified_arithmetic`, `operators`, `linguistics_puzzles`, `conlang_translation`) | far in-context assays | Apache-2.0 |
| SCAN (`tasks_test_addprim_jump`) | far in-context assay (shipped under `data/external/scan/` with its licence) | BSD licence (Lake & Baroni) |
| ELEMENTAL (compositional / simple) | far in-context assay (shipped under `data/external/elemental/` with its licence) | MIT (Kaiser Sun, 2026) |
| lm-evaluation-harness tasks (standard benchmarks) | benchmark evaluation after training | each benchmark's own licence (not redistributed) |

Where a source's terms restrict redistribution (Yahoo! Answers, arXiv full texts, Stack Exchange and Wikipedia
derivatives beyond short quotation), this repository ships only identifiers, hashes, label maps and the builder
code, and the user obtains the texts from the source under its terms.

## Model weights

No model weights or checkpoints are redistributed. The code loads public checkpoints from the Hugging Face Hub
under each model's own licence (Pythia, OLMo, Qwen, Gemma, SmolLM, Llama and the other families named in the configs).
