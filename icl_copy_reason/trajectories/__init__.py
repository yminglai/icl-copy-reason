"""Checkpoint tracing over pretraining: registry of released checkpoints, tracing of the two readings, the sign-change
interval of the two input dependences, and the fitted accuracy onset.

Modules: registry (families, revisions, positions, Hub survey), trace (readings over revisions, merging, summaries),
crossings (per-checkpoint statistics, D series, sign-change interval, descriptors, first crossings), onsets (logistic fit),
cohort (final-checkpoint screen of a family set and the trajectory dispositions), span_audit (tokenizer span alignment).
"""
