# EXP_026 — Attacker-Identity Generalization: Directional Attack, Seed 123, `client_004`

**Status:** Analysis-only document. `src/analysis/fedtrace_detector.py` (frozen FedTrace v3), `src/attacks/*` (attack implementation), every existing config, test, README, and Research Database file were read but **not modified**. No new experiment was run. FedTrace v3's formula, threshold (`>3`), persistence (`P=3`), and calibration rules were applied exactly as already frozen — nothing was tuned after seeing `EXP_026`'s result. This document was produced by directly reading `EXP_026`'s own result artifacts (`summary.json`, `client_round_trace.csv`) and by applying the unmodified `src/analysis/fedtrace_detector.py` module (imported read-only; no repository file was written to by this process) to that trace — the same procedure `EXP_016`–`EXP_019`'s own analyses used.

---

## 1. Pairing Verification

| Field | `EXP_026` | `EXP_018` | Match |
|---|---|---|---|
| `seed` | 123 | 123 | ✅ |
| `partition` / `dirichlet_alpha` | dirichlet / 0.1 | dirichlet / 0.1 | ✅ |
| `num_clients` / `clients_per_round` | 5 / 5 | 5 / 5 | ✅ |
| `communication_rounds_completed` | 50/50 | 50/50 | ✅ |
| `local_epochs` | 1 | 1 | ✅ |
| `model` / `total_parameters` | resnet18_cifar / 11,220,132 | same | ✅ |
| `optimizer` / `learning_rate` / `batch_size` | sgd / 0.01 / 32 | same | ✅ |
| `aggregation_strategy` | fedavg | fedavg | ✅ |
| `attack_type` | norm_preserving_directional | norm_preserving_directional | ✅ |
| `attack_start_round` | 20 | 20 | ✅ |
| `directional_target_cosine` | 0.0 | 0.0 | ✅ |
| `malicious_client_id` | **`client_004`** | `client_002` | **the one scientific change** |
| `client_sample_counts` | `{000:6024, 001:9628, 002:7202, 003:10361, 004:11785}` | identical, value-for-value | ✅ (same partition realization, seed 123) |

**`malicious_client_id` is confirmed to be the only scientific difference between `EXP_026` and `EXP_018`**, both directly from `summary.json` and by direct comparison of `configs/cifar100_fedavg_dirichlet_a01_direction_attack_client004_round20_seed123.yaml` against `configs/cifar100_fedavg_dirichlet_a01_direction_attack_client002_round20_seed123.yaml`.

**Additional cross-check, honest pre-attack fidelity:** `client_000`'s `update_l2_norm` at rounds 0–4 was compared directly between `EXP_018`'s and `EXP_026`'s trace files and found bit-identical (`5.8213851589389884, 5.151707704570757, 4.9021672767585756, 4.763422015790946, 4.796160350600854`) — confirming that changing which client is nominally malicious (a fact that only becomes operative at round 20) does not perturb any client's honest pre-attack training trajectory, exactly as expected from the single-variable-change design.

**Verified largest-partition client at seed 123: `client_004`, 11,785 samples** — read directly from `summary.json`'s `client_sample_counts` field (`EXP_008`'s and `EXP_026`'s own copies agree): `client_000`=6024, `client_001`=9628, `client_002`=7202, `client_003`=10361, **`client_004`=11785**. `client_004` is unambiguously the largest partition, and — as already established in `analysis/MULTIKRUM_DIRECTIONAL_CROSS_SEED_SYNTHESIS.md` Part 10 — is also the client Multi-Krum's clean-control rejection proxy persistently, falsely excludes at this same seed, making it a doubly-motivated (largest-partition *and* already-independently-flagged-as-structurally-distinct) alternative identity, not an arbitrary pick.

## 2. Attack Correctness

From `EXP_026`'s `client_round_trace.csv` (250 rows): `client_004`'s `attack_active` is `True` for every one of its 30 rows with `round_id >= 20`, and `False` for all 20 of its rows with `round_id < 20`. **No other client's `attack_active` is ever `True`, at any round** (confirmed by inspecting all 200 non-`client_004` rows).

**Norm preservation** (`delta_attack_norm_ratio`, 30 active rows): min `0.999999999819`, max `1.000000000125`, mean `1.000000000003` — **maximum absolute error from `1.0`: `1.81 × 10⁻¹⁰`**.

**Directional targeting** (`realized_cosine_to_honest`, 30 active rows): min `-0.000000000274`, max `0.000000000680`, mean `0.000000000112` — **maximum absolute error from `0.0`: `6.80 × 10⁻¹⁰`**.

**The attack is confirmed genuinely norm-preserving and correctly targeted, to the same floating-point precision already established for every prior directional-attack run in this project (`EXP_015`, `EXP_018`, `EXP_019`, `EXP_024`, `EXP_025`).** Changing the malicious client's identity did not alter the attack transform's own fidelity.

## 3. FedTrace v3 Applied Exactly

The frozen v3 formula (`research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md` §16, restated unaltered in `analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md` §3) was applied via `src/analysis/fedtrace_detector.py::evaluate_all_detectors_v3`/`evaluate_detector_v3` with no modification of any kind.

| Client | First raw anomaly | Confirmation round | Onset estimate | Max `z_m` | Max `z_t_res` | Max combined | Persistent detection? |
|---|---|---|---|---|---|---|---|
| `client_000` | 0 | — | — | 3.8943 (round 0) | 2.5303 (round 1) | 3.8943 (round 0) | **No** |
| `client_001` | 0 | — | — | 3.2558 (round 0) | 0.3234 (round 1) | 3.2558 (round 0) | **No** |
| `client_002` | 0 | — | — | 3.9303 (round 0) | 3.2888 (round 1) | 3.9303 (round 0) | **No** |
| `client_003` | 0 | — | — | 3.6073 (round 0) | 1.9904 (round 21) | 3.6073 (round 0) | **No** |
| **`client_004`** | 0 | **22** | **20** | 1.9097 (round 48) | 8.2836 (round 20) | 8.2836 (round 20) | **Yes** |

**Critical questions, answered directly from the above:**

1. **Is the predicted malicious client `client_004`?** **Yes.**
2. **Is the onset estimate exactly round 20?** **Yes** (onset error `0`).
3. **Is the confirmation round 22?** **Yes.**
4. **Is the detection delay 2?** **Yes** — the theoretical minimum under `P=3`, identical to all four Stage B runs.
5. **How many honest persistent false positives?** **0.**
6. **Does any honest client confirm before `client_004`?** **No — no honest client is ever confirmed at all** (`confirmation_round = None` for `client_000`/`client_001`/`client_002`/`client_003`).

**Note on the "First raw anomaly = round 0" column, reported per this task's explicit instruction not to hide raw threshold crossings — see §5 for the full explanation.** This is a real, but isolated and non-persistent, single-round crossing present for *every* client (including `client_004` itself, while still honest) and does not affect any of the six answers above; it is fully distinguished from the pre-registered, attack-relevant, persistent-run onset (round 20, §5).

## 4. Malicious-Client (`client_004`) Signal Analysis, Rounds 19–25

| Round | `z_m` | `z_t_res` | Combined score | Dominant |
|---|---|---|---|---|
| 19 (pre-attack) | 0.8327 | 0.7692 | 0.8327 | `m` |
| 20 (onset) | 0.9560 | **8.2836** | 8.2836 | `t_res` |
| 21 | 0.3819 | 7.9513 | 7.9513 | `t_res` |
| 22 (confirm) | 0.1149 | 7.3869 | 7.3869 | `t_res` |
| 23 | 0.1070 | 7.1073 | 7.1073 | `t_res` |
| 24 | 0.1051 | 6.8411 | 6.8411 | `t_res` |
| 25 | 0.0800 | 6.6737 | 6.6737 | `t_res` |

**`z_t_res` drives detection, decisively and immediately**, jumping from `0.77` (round 19, pre-attack, unremarkable) to `8.28` at round 20 itself — the same immediate, one-round-onset separation already established for `client_002` at seed 123 (`EXP_018`: `9.19` at round 20). Across the full 30-round post-attack window, `client_004`'s `z_t_res` ranges `4.44`–`8.28`, never dropping below threshold even once.

**`z_m` never becomes informative for this run at all.** Unlike `EXP_018` (where `client_002`'s `z_m` eventually crept above threshold late, rounds 47–49, driven by accumulated honest-counterfactual global-model drift), `client_004`'s post-attack `z_m` peaks at only `1.91` (round 48) and **never once exceeds `3` anywhere in the 50-round window** — confirmed directly: `B_magnitude_only` (magnitude-only detector) confirms **no client at all** in `EXP_026` (`confirmation_round = None` for every one of the five clients). This is a genuine, quantitative, seed/identity-dependent difference from `EXP_018`, not a discrepancy to be smoothed over: for this particular attacker identity, magnitude-only detection would have failed completely, not merely late.

**Dominance count:** `z_t_res > z_m` in **30/30 (100%)** of `client_004`'s post-attack anomalous rounds — identical in kind to `EXP_018`'s own 30/30 (100%) `z_t_res` dominance for `client_002`.

**Does changing attacker identity preserve the same temporal-residual mechanism? Yes — within this one controlled comparison.** The directional attack's defining property (norm preservation, §2) is identity-independent by construction, and the mechanism `z_t_res` exploits — a collapse in `self_prev_round_cosine` when a client's direction is replaced by an independently-drawn rotation — depends on the attack transform, not on which client executes it. `client_004`'s result reproduces `client_002`'s mechanism exactly: immediate, decisive, exclusively `z_t_res`-driven detection, with `z_m` at most marginally, non-decisively informative. If anything, `client_004`'s case is a *cleaner* demonstration of the mechanism than `client_002`'s, since `z_m` never becomes informative at all here, removing any ambiguity about which signal is doing the work.

## 5. Honest-Client Safety — Exhaustive Check, Including Isolated Raw Anomalies

All 5 clients' `z_m` and `z_t_res` were computed and inspected at every round with a defined signal — the complete 250-row trace, not a sample. Per this task's explicit instruction, isolated raw threshold crossings are reported in full, not omitted.

**Raw anomaly counts (score `> 3`, any single round, not necessarily persistent):**

| Client | Raw anomalous rounds | Persistent (`P=3`) anomalous runs |
|---|---|---|
| `client_000` | 1 — round 0 (`z_m = 3.8943`) | **0** |
| `client_001` | 1 — round 0 (`z_m = 3.2558`) | **0** |
| `client_002` | 2 — round 0 (`z_m = 3.9303`), round 1 (`z_t_res = 3.2888`) | **0** |
| `client_003` | 1 — round 0 (`z_m = 3.6073`) | **0** |

**Maximum honest `z_m` observed anywhere: `3.9303` (`client_002`, round 0). Maximum honest `z_t_res` observed anywhere: `3.2888` (`client_002`, round 1). Maximum honest combined score observed anywhere: `3.9303` (`client_002`, round 0).**

**No honest client produces even a single raw anomaly anywhere in rounds 2–49** (i.e., the entire post-attack window, and all of the pre-attack window except round 0/1) — every honest raw crossing is confined to the first two rounds of the run. None of the four isolated crossings above chains into a second consecutive anomalous round for the same client, so **zero honest clients are ever confirmed persistent, pre-attack or post-attack.**

### Incidental finding: a round-0 structural magnitude artifact, not specific to `EXP_026`

Every one of the four raw `z_m` crossings above occurs at **round 0**, for every client, honest or not (`client_004`'s own pre-attack round 0 `z_m = 3.1892` shows the identical pattern). Direct inspection of the underlying calibration explains this exactly, not merely observes it: `compute_longitudinal_calibration`'s per-client `z_m` reference window is `round_id < T_ref = 20` — a **self-inclusive** 20-observation sample that includes round 0 itself. For every one of the 5 clients at this seed/partition, **round 0 is the maximum `update_l2_norm` value within its own 20-round calibration window** (e.g. `client_000`: round-0 value `5.8214`, window mean `4.8952`, window stdev `0.2378` — a genuine, largest-in-sample outlier, not a computational error). This is consistent with round 0 being the very first local-training step from the shared random initialization — plausibly, and here confirmedly, a structurally larger update than subsequent, better-converged rounds for every client. A self-inclusive z-score for the sample maximum in an `n=20` window is bounded by `(n-1)/√n ≈ 4.25`; the observed values (`3.19`–`3.93`) sit within that bound, not anomalously so.

**This does not change any PASS/attribution/onset conclusion in this document** — each occurrence is a single, isolated, never-repeated round, and the persistence rule (`P=3` consecutive rounds) is specifically the mechanism designed to prevent an isolated fluctuation like this from ever being mistaken for a detection (§11 of `analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md` makes exactly this point in the abstract; this is a concrete instance of it).

**It is, however, worth reporting transparently as a cross-check finding beyond `EXP_026`'s own scope:** re-running the identical, unmodified `evaluate_all_detectors_v3`/`compute_all_scores_v3` code directly against `EXP_018`'s own `client_round_trace.csv` (performed here purely as a diagnostic comparison, not as a re-analysis or modification of `EXP_018`) reproduces the same round-0 pattern for all five of *its* clients (`z_m` at round 0: `client_000` `3.8943`, `client_001` `3.2558`, `client_002` `3.9303`, `client_003` `3.6073`, `client_004` `3.1892` — identical values to `EXP_026`'s, as expected from §1's bit-identical-pre-attack-data confirmation). `analysis/EXP_018_STAGE_B_DIRECTIONAL_SEED123_VALIDATION.md` §10 states "No honest client's `z_m` or `z_t_res` ever exceeds `3` at any round, pre- or post-attack" — under this direct re-verification, that specific sentence does not hold as literally stated (four of its five clients do cross `3`, at round 0 specifically). **This does not change `EXP_018`'s own PASS verdict or its "0 honest persistent false positives" finding, which remain correct** — the discrepancy is confined to that one descriptive sentence about isolated raw crossings, not to any pre-registered criterion, all of which were evaluated correctly against persistence, not against isolated raw crossings. Per this task's explicit restriction, `EXP_018`'s document is **not** modified here; this finding is reported for transparency and is flagged as a candidate for a future, separately-authorized documentation correction. Whether the same round-0 pattern is present in `EXP_016`, `EXP_017`, and `EXP_019` was **not** checked in this task and is not claimed either way.

## 6. `EXP_026` vs. `EXP_018` — Attacker-Identity Generalization

| | `EXP_018` | `EXP_026` |
|---|---|---|
| Malicious client | `client_002` | `client_004` |
| Partition sample count | 7,202 | 11,785 (largest) |
| FedTrace v3 predicted client | `client_002` | `client_004` |
| First raw anomaly (any client) | round 0 (structural artifact, §5) | round 0 (structural artifact, §5) |
| First anomaly of the confirmed run | 20 | 20 |
| Confirmation round | 22 | 22 |
| Onset estimate | 20 | 20 |
| Onset error | 0 | 0 |
| Detection delay | 2 | 2 |
| Honest persistent FP | 0 | 0 |
| Dominant signal | `z_t_res`, 30/30 (100%) | `z_t_res`, 30/30 (100%) |
| Did magnitude-only ever confirm? | Yes — very late (round 49) | **No — never, within 50 rounds** |
| Final test Top-1 | 53.10% | 50.23% |
| Final test Top-5 | 81.40% | 80.49% |

**Within this one controlled comparison — same seed, same partition, same attack family, same onset, same severity, only the malicious identity changed — frozen FedTrace v3's behavior is invariant on every pre-registered criterion:** identical correct attribution, identical zero onset error, identical minimum-possible delay, identical zero honest false positives, and the identical dominant mechanism (`z_t_res`). The one substantive difference (`client_004`'s `z_m` never becoming informative, versus `client_002`'s eventually, weakly doing so) makes `EXP_026`'s result a *more*, not less, decisive confirmation that the temporal-residual channel — not magnitude — is what generalizes across attacker identity for this attack family.

**Cautious framing, as instructed:** this is evidence of generalization to **one** new, deliberately-chosen (largest-partition) alternative identity. It is not proof of invariance to an arbitrary client, still less to an adversarially-chosen "hardest" identity (e.g. a client picked specifically to resemble the honest reference distribution as closely as possible) — no such search was performed, and this project's own established discipline (`analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md` §17) already prohibits overstating a small evidence sample as a general guarantee. One new identity narrows, but does not close, the identity axis of the generalization question (§8).

## 7. Performance Interpretation

| Comparison | Top-1 Δ | Top-5 Δ | Test loss Δ | Best validation Top-1 Δ |
|---|---|---|---|---|
| `EXP_026` vs. `EXP_008` (attacked-`client_004` vs. clean) | **−6.60 pp** | **−7.30 pp** | +0.5295 | **−7.46 pp** |
| `EXP_026` vs. `EXP_018` (`client_004` vs. `client_002`, both attacked) | **−2.87 pp** | **−0.91 pp** | +0.1367 | −4.30 pp |
| (reference) `EXP_018` vs. `EXP_008` | −3.73 pp | −6.39 pp | +0.3927 | −3.16 pp |

**The directional attack executed by `client_004` causes materially larger global-model degradation than the identical attack executed by `client_002`** (−6.60pp vs. −3.73pp Top-1, relative to the same clean `EXP_008` baseline) — consistent with, though not proof of, the intuitive mechanism that a norm-preserving directional perturbation contributed by the *largest* partition's update carries more aggregate weight under FedAvg's sample-size-weighted averaging than the same perturbation from a smaller partition (`client_002`, 7,202 samples, is the second-smallest of the five).

**This performance difference is reported as a separate, distinct outcome from attribution, exactly as instructed — it is not used as, and does not function as, a detection or attribution signal anywhere in §3–§6.** FedTrace v3 correctly attributed and localized `client_004` (§3) using only its own trace-level `update_l2_norm`/`self_prev_round_cosine` signals, computed identically regardless of the eventual global-model impact; the global accuracy numbers above characterize the attack's *damage*, not FedTrace's ability to *detect* it. Model utility and forensic attribution remain two separate outcomes in this analysis, as they have throughout every prior document in this series.

## 8. Evidence-Gap Verdict

`analysis/ICML_ICLR_EVIDENCE_GAP_AUDIT.md` Part 6 identified exactly one essential remaining gap: "malicious-client identity / onset generalization," concentrated in the fact that every attack-bearing run through `EXP_025` used the identical attacker (`client_002`) at the identical onset (round 20).

**A. Attacker-identity generalization: narrowed, with one new data point supporting generalization.** `EXP_026` demonstrates that frozen FedTrace v3, applied with no modification, correctly attributes and exactly localizes a *different*, deliberately-chosen (largest-partition) malicious client under the same attack family, seed, and onset. This directly, if narrowly, closes part of the audit's identified gap — the single most concentrated confound (a fixed attacker identity across every prior run) is no longer universal.

**B. Onset-round generalization: remains completely untested.** `EXP_026`'s `attack_start_round` is `20`, identical to every prior attack run in this project (`EXP_014`–`EXP_019`, `EXP_022`–`EXP_025`). **No run in this project has ever used an onset round other than 20.** This axis of the audit's identified gap is untouched by `EXP_026` and remains fully open.

**Stated explicitly, per this task's instruction: `EXP_026` changes identity only. The attacker-identity gap is narrowed, not eliminated (one new identity, not an exhaustive or randomized sample of identities). The onset-round gap remains entirely open and is not addressed by this experiment.**

## 9. Next-Gate Decision

**A. Attacker-identity generalization PASSES (within the scope of this one controlled comparison); one onset-generalization run is justified as the final essential experiment identified by the ICML/ICLR audit.**

Not B: the result is not ambiguous — every pre-registered criterion (attribution, onset error, delay, honest FP) matched `EXP_018` exactly, and the one difference observed (magnitude-only's complete failure to ever confirm) strengthens rather than weakens the case that the temporal-residual mechanism is what generalizes.

Not C: no scientific issue was found in frozen FedTrace v3's behavior. The one incidental finding (§5's round-0 structural artifact) is a pre-existing, fully explained, non-persistence-breaking property of the already-frozen magnitude calibration, present identically in `EXP_018`'s own trace under direct re-verification — it is a documentation-completeness observation about a prior analysis document's specific sentence, not a defect in the frozen detector's decisions, which remain correct on every pre-registered criterion in both runs.

**Recommended next experiment (not created in this task):** exactly one onset-generalization run, changing **only** `attack_start_round` relative to `EXP_018` (not relative to `EXP_026`, to keep the two generalization axes independently, not jointly, varied — avoiding a compounded-identity-and-onset confound). Concretely: `malicious_client_id: client_002` (held at `EXP_018`'s value), `attack_type: norm_preserving_directional`, `directional_target_cosine: 0.0`, `seed: 123` — all unchanged — with `attack_start_round` changed from `20` to a different round (e.g., **round 30**, chosen to be substantially distinct from round 20 while still leaving a comparable post-onset observation window — 20 rounds — to the 30-round window `EXP_018`'s own onset left; the exact round value is a recommendation only and is not binding). This keeps `EXP_018` as the single shared anchor for two independent, one-factor-away comparisons (`EXP_026`: identity changed; the recommended next run: onset changed), rather than stacking both changes into one run, which would confound the two axes the ICML/ICLR audit asked to be tested separately.

---

## Final Report

1. **Files created:** `analysis/EXP_026_ATTACKER_IDENTITY_GENERALIZATION_ANALYSIS.md`, `analysis/FEDTRACE_GENERALIZATION_GATE_STATUS.md`.
2. **Pairing verdict:** confirmed — `EXP_026` differs from `EXP_018` scientifically only in `malicious_client_id` (`client_002` → `client_004`); every other scientific field (dataset, model, partition, alpha, seed, client count, rounds, local epochs, optimizer, LR, batch size, aggregation, attack family, onset, target cosine) is identical, verified field-by-field (§1). `client_004` is confirmed the largest-partition client at seed 123 (11,785 samples).
3. **Attack correctness:** confirmed — `client_004`'s `attack_active` is `True` for exactly its 30 post-round-20 rows and `False` elsewhere; no other client is ever active; `delta_attack_norm_ratio` = `1.0` to within `1.81×10⁻¹⁰`; `realized_cosine_to_honest` = `0.0` to within `6.80×10⁻¹⁰` (§2).
4. **FedTrace predicted client:** `client_004` — correct (§3).
5. **First anomaly / confirmation / onset:** first anomaly of the confirmed run = round 20; confirmation = round 22; onset estimate = 20 (§3, §4).
6. **Onset error:** **0** (§3).
7. **Detection delay:** **2** — the theoretical minimum under `P=3` (§3).
8. **Honest persistent false positives:** **0** (§3, §5).
9. **Dominant signal:** `z_t_res`, 30/30 (100%) of `client_004`'s post-attack anomalous rounds; `z_m` never became informative at all in this run (never exceeded 3, anywhere) (§4).
10. **Exact `client_004` `z_m`/`z_t_res` values around onset (rounds 19–25):** round 19: `z_m=0.83, z_t_res=0.77`; round 20: `z_m=0.96, z_t_res=8.28`; round 21: `0.38, 7.95`; round 22: `0.11, 7.39`; round 23: `0.11, 7.11`; round 24: `0.11, 6.84`; round 25: `0.08, 6.67` (§4).
11. **`EXP_026` vs. `EXP_018` attribution comparison:** identical outcome on every pre-registered criterion (correct attribution, onset error 0, delay 2, honest FP 0, dominant signal `z_t_res` 100%); the one difference (magnitude-only never confirmed in `EXP_026`, vs. eventually/late in `EXP_018`) makes `EXP_026` a cleaner, not weaker, demonstration of the same mechanism (§6).
12. **`EXP_026` vs. `EXP_008` performance difference:** Top-1 −6.60pp, Top-5 −7.30pp, test loss +0.5295, best-validation Top-1 −7.46pp — larger degradation than `EXP_018`'s (−3.73pp Top-1), consistent with the largest-partition client's greater aggregate weight under FedAvg; reported as a separate outcome from attribution, not interpreted as a detection metric (§7).
13. **Attacker-identity generalization verdict:** narrowed/supported by one new, deliberately-chosen (largest-partition) identity — every pre-registered criterion held; not proof of arbitrary-identity invariance (§6, §8).
14. **Whether onset-generalization remains open:** **yes, entirely** — `EXP_026`'s onset is unchanged at round 20, identical to every prior attack run in this project; no onset value other than 20 has ever been tested (§8).
15. **Evidence-gap status:** the ICML/ICLR audit's single essential gap (attacker-identity/onset) is narrowed on the identity axis, untouched on the onset axis (§8).
16. **Next-gate decision:** **A** — attacker-identity generalization passes within this one controlled comparison; one onset-generalization run is the recommended final essential experiment (§9).
17. **Exact next experiment recommended:** one run, anchored against `EXP_018` (not `EXP_026`), changing only `attack_start_round` (suggested: round 30), holding `malicious_client_id=client_002`, attack family, `directional_target_cosine=0.0`, and seed 123 fixed — **not created in this task** (§9).
18. **Confirmation FedTrace v3 unchanged:** confirmed — `src/analysis/fedtrace_detector.py` was read and imported for computation, never edited; every formula, threshold, and calibration rule used is exactly as already frozen.
19. **Confirmation attack code unchanged:** confirmed — no file under `src/attacks/` was read or touched in this task.
20. **Confirmation no config/test/result/database file changed:** confirmed — no existing config, test, README, result artifact, or Research Database file was created, edited, or deleted; only the two new analysis documents listed in item 1 were created.
21. **Confirmation no experiment was run:** confirmed — no training/federated-runner script was invoked; only `EXP_026`'s already-existing result artifacts were read.
22. **Confirmation whether commands were executed:** **yes, disclosed explicitly** — this task used `Read`, `Glob`, `Write`, and `Bash` tool calls. The `Bash` calls ran a read-only Python analysis script (written to the scratchpad directory, not the repository) that imports the unmodified `src/analysis/fedtrace_detector.py` module and computes scores from already-collected `client_round_trace.csv`/`summary.json` files — the same class of computation `EXP_016`–`EXP_019`'s own analyses required, producing no write to any repository file. No training, no experiment runner, and no modification command of any kind was executed.
