# 13 — Benign Client-Specific Drift Control: Preregistration (development only)

Status: **FROZEN 2026-10-25, before any drift run** (none has been executed). Changes only by dated amendment (§8).
Seeds 123 and 2026 are DEVELOPMENT seeds; sealed confirmation seeds are not used. This is not independent validation.

## 1. Question

Does standard client-own norm evidence (the best descriptive Phase S baselines) falsely accuse an honest client under
legitimate, correctly labelled non-IID drift? (H-BEN: it does not.)

## 2. Intervention (implemented in `src/rev/attacks.py::BenignClassDriftDataset`; tested)

From round 20, honest `client_002` (fixed rule: the same client identity used by every intervention arm, chosen before
any drift outcome, so drift, BadNets and relabel-only are compared on the same client and partition) trains on a
class-composition shift of its own data: its classes are taken in a private random order (generator keyed on seed,
client, "benign_class_drift") until they cover ≥ 50 % of its samples; each sample of a non-retained class is replaced
by a fixed, uniformly drawn (with replacement) sample of a retained class. Real partitions: seed 123 — 25 of 53
classes retained, 44.1 % of samples replaced, 4,027 unique samples after onset (multiplicity ≤ 7); seed 2026 — 30 of 57
classes, 47.2 %, 4,842 unique (≤ 8).

Why legitimate drift, not an attack: every label is the true label of the image returned; no trigger; no other
client's data; no objective chosen by an adversary; it models a client whose data collection narrows to part of its
own categories (e.g. seasonal or site-specific acquisition).

Confounders: sample count, batches, local optimiser steps, epochs, optimiser and schedule — **controlled** (dataset
length unchanged). DataLoader RNG use — controlled (length unchanged; each access reads one base sample). Unique-sample
count (≈ 0.56 / 0.53 of the original, with duplicates) and class imbalance — **intended part of the drift, not
controlled**. Global-model trajectory after round 20 — not controllable (it depends on the drifted updates).

## 3. Design

| Item | Value |
|---|---|
| Runs | `X4_drift_d123`, `X4_drift_d2026` (`configs/rev/X4_drift_d{123,2026}.rev.yaml`) |
| Matched controls | existing `X1_clean_d{seed}` (same base config; since their commits the training path changed only by opt-in wrappers that are absent in the clean arm); the matched-prefix check (rounds < 20 bit-identical, honest round-20 submissions identical) is **required**; failure ⇒ comparison invalid |
| Onset | round 20 (= T_ref; onset-30 not included) |
| Outputs | trace, 301-record sealed update log, checkpoints; pair analysis with `--benign` (no malicious client); `src.rev.benign_drift` report |

## 4. Endpoints (Phase S definitions unchanged)

1. Honest false alarms under frozen v3 (online wrapper; τ = 3, P = 3).
2. Honest false alarms under the two-sided norm z-score with persistence, h = **1.75** (Phase S L-56; category 1:
   retrospective calibration on the two clean development runs).
3. Honest false alarms under two-sided norm CUSUM, h = **11.75** (Phase S L-56; category 1).
4. Update-norm ratio drift / clean, rounds 25–35, per client; drifted client classified DECREASE (≤ 0.90; poisoning
   gave 0.79–0.87), INCREASE (≥ 1.10), otherwise NO MATERIAL CHANGE.
5. Raw self-cosine change of the drifted client, rounds 21–29 (poisoning gave +0.075 … +0.120).
6. Shared-reference movement caused by the drifted client (inclusive median; diagnosis only).
7. Final clean top-1 (drift vs clean).
8. Detection decisions (any confirmed client = honest false accusation).

Information categories: features and per-client standardisation are causal (category 2); h = 1.75 and 11.75 are
retrospective clean-development calibrations (category 1) and are **not** recalibrated on drift outcomes; τ, P, k
are fixed (category 3). Online decisions = confirmations at the round they occur.

## 5. Predictions and verdict rules (FIXED)

- **H-BEN, per detector (endpoints 1–3):** FALSIFIED if any client is confirmed in either drift run; otherwise
  "not falsified on two development runs" — never a false-positive guarantee.
- **Specificity of the poisoning signatures (endpoints 4–5):** if the drifted client's norm class is DECREASE, or its
  self-cosine change is ≥ +0.075, the corresponding signature is recorded as **not specific to poisoning**.
- **Reference contamination by benign drift (endpoint 6):** recorded if any other client's mean drift-induced
  movement is ≥ +1 z.

## 6. Known risks stated before execution

The z-score-on-norm threshold (1.75) exceeds the clean statistic of `client_002` at seed 2026 by only 0.03
(calibration sensitivity, L-61); clean client-rounds are strongly dependent (lag-1 autocorrelation 0.82–0.91,
cross-client correlation ≈ 0.95), so the calibration rests on very few effective observations.

## 7. Commands (not executed; require PI approval)

```
python -m src.rev.preflight --runs X4_drift_d123 X4_drift_d2026
python -m src.rev.run_rev_experiment --config configs/rev/X4_drift_d123.rev.yaml
python -m src.rev.run_rev_experiment --config configs/rev/X4_drift_d2026.rev.yaml
python -m src.rev.pilot --clean X1_clean_d123 --attacked X4_drift_d123 --malicious client_002 --onset 20 --target-class 0 --trigger-size 3 --seed 123 --benign
python -m src.rev.pilot --clean X1_clean_d2026 --attacked X4_drift_d2026 --malicious client_002 --onset 20 --target-class 0 --trigger-size 3 --seed 2026 --benign
python -m src.rev.benign_drift --seeds 123 2026
```
(`--malicious` names the intervened client for the prefix and diagnostics; `--benign` scores the run with no
malicious client.)

## 8. Amendments

### Amendment 1 — 2026-10-25 (Prompt 27; written before any drift run). Interpretation only: the intervention, the selected client, thresholds, predictions and evaluation rules above are unchanged.

- This is a **deliberately selected stress test**, not a representative estimate of benign-drift false-positive rates.
- The drifted client, client_002, was previously the poisoning client in the BadNets and relabel-only arms.
- Its seed-2026 clean norm statistic (1.719) was already known to lie close to the z-score threshold (1.75); this prior
  knowledge may influence how the result is interpreted, and an alarm on this client at seed 2026 would be partly
  anticipated rather than surprising.
- The intervention preserves dataset length and local optimiser steps but changes unique-example frequency (repeated
  samples) and class composition; an observed effect cannot be attributed to one of these factors alone.
- A single threshold crossing is not a confirmed alarm: only a confirmation under the frozen persistence rule
  (three consecutive rounds for v3 and the z-score; the CUSUM statistic exceeding h) counts as an accusation.
- The effective-sample-size estimate (L-61) is descriptive, not a formal guarantee.
