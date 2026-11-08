"""Reviewer-response (Phase R2) infrastructure for the NDSS FedTrace line.

Everything here is additive: the frozen NDSS modules (FedTrace v3 detector,
attacks, federated stack, runner script) are imported and used unchanged,
never edited. Raw outputs go to the git-ignored ``results_rev/``,
``checkpoints_rev/`` and ``logs_rev/`` trees.

Modules:
    hashing     -- canonical tensor/JSON hashing and atomic file writes.
    provenance  -- git/environment/config/partition/initialization provenance.
    update_log  -- optional, read-only fp32 submitted-update logger.
    replay      -- offline verification and replay of an update log.
    hooks       -- runner subclass and NDSS-script hook installation.
    online      -- causal (online) wrapper around frozen FedTrace v3.
    baselines   -- client-wise z-score + persistence, CUSUM, Page-Hinkley.
    attacks     -- backdoor, targeted label flip, constrained-backdoor prototype.
    metrics     -- run-level outcomes and seed-level statistics.
    config      -- reviewer-response run configuration.
"""
