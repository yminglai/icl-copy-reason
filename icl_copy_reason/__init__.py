"""icl_copy_reason: diagnostic prompts, two readings of in-context behaviour, training-data interventions.

Sub-packages:
  tasks              diagnostic prompt generator (made-up endings, one combination withheld), item sets of record
  readings           copy-probability share, attention ablation, activation patching, deletion, scoring utilities
  data_construction  definition-use exercises from real documents, synthetic definition-use documents, external sets
  training           continued pretraining with an admixture, fine-tuning, LoRA transfer, manifests and packing
  trajectories       checkpoint tracing over pretraining, sign-change interval, fitted accuracy onset
  analysis           tables of record from released per-run outputs, FACTS check
Configuration: icl_copy_reason.config (YAML files under configs/, environment variables ICR_*).
"""

__version__ = "0.1.0"
