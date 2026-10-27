# 29 — Confirmation Results (protocol 26; seeds W1, W2) and Development-vs-Confirmation Comparison

Prepared 2026-10-27 (Prompt 34). Labels: **CONFIRMATION** (W1 = 1178015705, W2 = 1463065807; preregistered,
frozen methods and thresholds; CLAUDE-REPORTED), **DEVELOPMENT ONLY** (123, 2026). Seed-level results only; nothing is
pooled; no rate is estimated. Interpretation rules: protocol 26 Amendment 1 §D (frozen before execution).

## 1. Execution and integrity

| Item | Result |
|---|---|
| Runs | 6 / 6 completed, exit 0, preregistered order; 2026-10-26 20:46:41Z → 2026-10-27 04:21:29Z (7.58 h wall; per run 70.6–82.4 min) |
| Provenance | all six `execution_clean = True` at commit `4720670`; no tracked modification during execution |
| Update logs | six sealed logs (≈ 13 GiB each); manifests recorded in `rev_status.json` |
| Matched prefixes | **valid** for all four pairs (21 globals, 100 submissions, 120 rows bit-identical; first difference round 20) |
| GPU / disk | peak GPU memory 1,619 MiB of 8,151 MiB (455 one-minute samples); storage ≈ 81 GiB (6 × 13 GiB logs + 6 × 515 MiB checkpoints + 6 × 1.1 MiB results); free disk 414 GB after |
| Abort criteria | none triggered |
| Analyses | pair analyses, contamination (`contamination_confirmation.json`), evidence matrix (`evidence_matrix_confirmation/`), unified table (`unified_outcomes_confirmation.json`, doc 30) — all CPU (CUDA hidden), exit 0 |

## 2. Development vs confirmation, per seed (never pooled)

| Quantity | 123 (dev) | 2026 (dev) | **W1** | **W2** |
|---|---|---|---|---|
| Clean final top-1 | 56.88 % | 57.94 % | 58.04 % | 60.22 % |
| BadNets ASR (criterion ≥ 50 %) | 43.5 % — partially effective | 64.7 % — effective | **63.3 % — effective** | **44.2 % — partially effective** |
| Δ top-1 BadNets / drift (pp) | −1.94 / −1.75 | −0.94 / −1.79 | −0.96 / −2.29 | −0.86 / −1.47 |
| client_002 self-consistency rank before onset (r15–19; 1 = lowest of 5) | 4 | **3 (median client)** | 4 | 4–5 |
| Upward crossing of the inclusive median after onset | no (round 20 only, downward) | **yes** (+0.08 for ≥ 9 rounds) | no (round 20 only, downward −0.006) | no (round 20 only, downward −0.040) |
| Attacker peak z_m (r20–49) vs τ = 3 | 3.26 | 3.57 | 3.04 (2 rounds ≥ 3) | 2.94 (none) |
| **v3 on BadNets** | correct (alarm 30, onset error 8, delay 10) | **wrong**: honest client_001, client_004 at r23, attacker r24 | **missed** (no alarm) | **missed** (no alarm) |
| v3 on benign drift | none | **accuses client_001, client_004 (r23; reference-driven)** | accuses client_002 (drifted, honest) r43 — **norm-driven** | none |
| v3 on clean | none | none | none | none |
| ZP-norm (h 1.75) BadNets | correct, error 1, delay 3 | correct, 0 / 2 | correct, 1 / 3; **also honest client_004 r40** | correct, 1 / 3 |
| CUSUM-norm (h 11.75) BadNets | correct, 0 / 6 | correct, 0 / 5 | correct, 0 / 6; **also honest client_004 r39** | correct, 0 / 6 |
| ZP / CUSUM on drift | client_002 r23 / r28 | client_001 r49 / client_002 r37 | client_002 r22 + client_004 r43 / client_002 r26 + client_004 r39 | client_002 r42 / r36 |
| ZP / CUSUM on clean | none / none | none / none | **client_004 r43 / r41** | none / none |
| v3 historical-reference on clean | 3 clients | 2 clients | 4 clients | 2 clients |
| Reference-driven honest accusations (protocol 25 H-C3 class) | 0 | 6 (3 arms × 2 clients) | **0** | **0** |

Relabel-only was not part of protocol 26 (development seeds only).

## 3. Does the seed-2026 failure reproduce? (rule D5)

**No — on neither W1 nor W2.** In neither BadNets run did v3 confirm an honest client before (or instead of) the
attacker, and the only drift accusation (W1, client_002 at r43) is norm-driven, not reference-driven. Stated
plainly: the reference-contamination false attribution observed at seed 2026 **did not recur** on the two
preregistered confirmation seeds.

## 4. Mechanism results, four categories kept separate (Task 6)

| Category | W1 | W2 |
|---|---|---|
| **A. Exact algebraic identity** (Lemma A, clip form vs median) | max error 0.0 | max error 0.0 |
| **B. Retrospective matched-clean explanation** | intervened client's displacement −0.006 at round 20, **0 in rounds 21–29** (it stayed above the band in the attacked and the counterfactual run) → no contamination to explain | −0.040 at round 20, 0 afterwards → same |
| **C. Deployable proxy** | spurious positive displacement 0.001–0.011 after onset where B is 0 (MAE vs B 0.58–0.65 z on the W1 matched runs); clean-run flag rate 0.09 (development 0.00) | spurious 0.005–0.014 (MAE 2.5–2.7 z); clean-run flag rate 0.20 |
| **D. Actual detector decisions** | no temporal-residual-necessary honest alarm; one norm-driven drift alarm (client_002, r43) | no honest alarm under v3 |
| Frozen verdicts | H-C2 PASS on 12 sign-test client-rounds (round 20 only; uninformative); H-C3 NOT EVALUABLE (no reference-driven alarm); H-C4 PASS (0.09 / 0.20 vs B0 1.00 / 1.00); coded overall string "MECHANISM CLAIM NOT SUPPORTED" — under rule D1 this means **mechanism not exercised**, not falsified | |

Reading (rules D1–D3): the algebraic identity holds (A); **the false-attribution mechanism was not exercised on
the new seeds** because its precondition — an upward crossing of the band by the intervened client — did not occur:
in three of the four seeds (123, W1, W2) client_002 started **above** the median (rank 4) and its rise to rank 5 could
not move the inclusive median; only at seed 2026 was it the median client (rank 3). This is consistent with
Lemma A but is **not** a confirmation of the mechanism on new seeds. The deployable proxy (C) performed worse on the
confirmation seeds (MAE 0.6–2.7 z vs 0.07 z in development) because its assumption that the intervened client keeps
its historical offset failed: its counterfactual self-cosine also moved. No honest accusation on the new seeds is
left unexplained (none is reference-driven).

## 5. What the confirmation runs establish (CONFIRMATION, two seeds, no rates)

1. **Benign drift produces honest false accusations again:** norm z-score and CUSUM accuse the drifted (honest) client
   on both W seeds (and an additional honest client on W1); v3 accuses the drifted client on W1 (norm channel).
   H-BEN is falsified for the norm detectors on all four seeds.
2. **v3 does not reliably detect BadNets data poisoning:** missed on W1 (effective attack, 63 % ASR) and W2;
   marginal on seed 123 (alarm 30); the attacker's norm z peaks at 2.9–3.6 against τ = 3, consistent with the
   calibration-sd inflation (L-31).
3. **Plain norm detectors attribute BadNets correctly on all four seeds** (onset error 0–1), but their
   development-calibrated thresholds raise honest alarms on W1 (clean run and BadNets run, client_004) — threshold
   transfer across seeds fails.
4. **The client-historical reference variant accuses honest clients in every clean run** (all four seeds).
5. **Reference-driven false attribution is seed-dependent:** it occurred only when the intervened client was the
   pre-onset median client (1 of 4 seeds); it did not occur on either confirmation seed.

## 6. Remaining gaps (no experiment authorised)

The mechanism's precondition (intervened client at the median rank before onset) has been observed once; a direct
test would require a preregistered design that fixes the intervened client by its pre-onset rank — a new experiment
needing PI approval. Relabel-only has no confirmation runs. FLDetector deferred (P-58). K = 5, CIFAR-100,
ResNet-18 only; one drift type; onset = T_ref in all confirmation runs.
