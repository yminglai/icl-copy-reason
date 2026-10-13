"""`python -m icl_copy_reason` — lists the command-line entry points and their configs."""
from __future__ import annotations

import argparse
import sys

COMMANDS = [
    ("scripts/make_prompts.py", "build the diagnostic prompts (made-up endings, one combination withheld)"),
    ("scripts/score_model.py", "score a Hugging Face causal LM: copy-probability share, attention ablation, patching"),
    ("scripts/build_exercises.py", "real documents -> definition-use exercises (generate / verify / audit / pack)"),
    ("scripts/build_synthetic_docs.py", "synthetic definition-use documents and the every-combination-shown control"),
    ("scripts/build_external_sets.py", "symbol-label and exchanged-label versions of TREC-6 / DBpedia-7 / Yahoo-5"),
    ("scripts/train.py", "continued pretraining with an admixture, fine-tuning, LoRA transfer (configs/*.yaml)"),
    ("scripts/trace_checkpoints.py", "trace the two readings over released pretraining checkpoints"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m icl_copy_reason", description=__doc__)
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args(argv)
    if args.version:
        from . import __version__

        print(__version__)
        return 0
    print("icl_copy_reason command-line entry points (run each with --help):\n")
    for cmd, what in COMMANDS:
        print(f"  {cmd:<34} {what}")
    print("\nConfiguration: configs/<experiment>.yaml; environment ICR_API_BASE, ICR_API_KEY, ICR_DATA_ROOT, ICR_RESULTS_ROOT, ICR_HF_HOME.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
