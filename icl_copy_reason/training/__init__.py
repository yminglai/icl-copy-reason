"""Training-data interventions: continued pretraining with an admixture, exercise fine-tuning, LoRA transfer, direct fine-tuning.

Modules:
  pretrain        one continued-pretraining trainer (replacement arms and row-map arms, calibration mode, exact resume)
  lora            LoRA recipe transfer on released 7-8B models (packing, rate calibration, training, merge check)
  direct_finetune direct fine-tuning on diagnostic prompts, plain or with an attention / attribution regulariser
  manifests       corpus manifests (token packs + validation files) and per-row source maps for the exercise arms
  pack            token packing: corpus rows, synthetic-document rows, exercise rows (exposure-gated), LoRA alternating rows
  freeze          the records fixed before each run (hashes, rules, families) as one module
  launch/*.sh     one launcher per experiment (same hyper-parameters as the runs of record; paths from ICR_* variables)
"""
