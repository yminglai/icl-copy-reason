"""Training-data interventions: how the documents that move a model from copying toward composing were built.

Modules
  documents       the eight public document pools, eligibility and the deterministic content-blind sampler
  verification    answer normalisation, the evaluation-window store, the mechanical checks (1-6) and the pass/retry rule
  exercises       real documents -> definition-use exercises: generate / retry / verify / audit (one API client)
  augment_real    the first exercise set (types a-g, hash-assigned types) built with the same machinery and its own prompt table
  variants        the two extra exercise shapes tried in the exercise-variant loop (type k, type h2) and their checks
  synthetic_docs  synthetic definition-use documents (compose) and the every-combination-shown control; preparation checks
  mining          mined real definition-use documents: classifier-selected (arm D) and verifier-selected windows (D2) with the
                  dictionary-format rendering (D3), pool surveys, audits
  external_sets   symbol-label / exchanged-label evaluation sets, SCAN / ELEMENTAL / far assays and their scoring
"""
