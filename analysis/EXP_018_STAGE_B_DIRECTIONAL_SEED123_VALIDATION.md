# EXP_018 — Stage B Independent Validation: Directional Attack, Seed 123

**Status:** Independent validation analysis. `EXP_018` was **not** used to design, diagnose, or tune FedTrace v3. FedTrace v3 was applied here **exactly as frozen** (`research_plans/FEDTRACE_STAGE_B_INDEPENDENT_VALIDATION_PLAN.md` §1) — nothing was tuned after inspecting the trace, regardless of outcome. This is the first of the two directional-family Stage B runs.

---

## 1. Pairing Verification

| Field | `EXP_018` | `EXP_008` | Match |
|---|---|---|---|
| `seed` | 123 | 123 | ✅ |
| `partition` | dirichlet | dirichlet | ✅ |
| `dirichlet_alpha` (config) | 0.1 | 0.1 | ✅ |
| `client_sample_counts` | `{000:6024, 001:9628, 002:7202, 003:10361, 004:11785}` | identical, value-for-value | ✅ (same partition realization) |
| `model` / `total_parameters` | resnet18_cifar / 11,220,132 | same | ✅ |
| `optimizer` / `learning_rate` | sgd / 0.01 | same | ✅ |
| `local_epochs` | 1 | 1 | ✅ |
| `communication_rounds_completed` | 50/50 | 50/50 | ✅ |
| `aggregation_strategy` | fedavg | fedavg | ✅ |
| `batch_size` | 32 | 32 | ✅ |

**The frozen directional attack is confirmed to be the only scientific difference.** As an additional cross-check beyond `summary.json`: `EXP_018`'s pre-attack (`round_id < 20`) `update_l2_norm` and `peer_mean_cosine_similarity` values were compared directly against `EXP_016`'s (the other seed-123 attacked run already validated) and found bit-identical row for row — e.g. `client_002` round 0 `update_l2_norm = 6.743401186732322` in both — confirming the honest pre-attack trajectory at seed 123 is reproduced exactly regardless of which attack family is subsequently activated, exactly as expected from the single-variable-change design.

## 2. Attack Correctness

`summary.json`: `attack_type=norm_preserving_directional`, `malicious_client_id=client_002`, `attack_start_round=20`, `directional_target_cosine=0.0`.

Full `client_round_trace.csv` (250 rows): `client_002`'s `attack_active` is `False` for `round_id < 20` and `True` for `round_id >= 20`. **No other client's `attack_active` is ever `True`, at any round** (confirmed by inspecting all 200 non-`client_002` rows).

**On `attack_strength_rho = 0.0` in `summary.json`:** this is confirmed to be the inert, backward-compatible metadata field, not an indication that no attack occurred. `attack_strength_rho` is documented (`configs/cifar100_fedavg_dirichlet_a01_direction_attack_client002_round20_seed123.yaml`'s own header comment, `src/attacks/traceability.py::AttackConfig`) as a Gaussian-attack-specific parameter that is simply absent from this config and falls back to its default `0.0` — it plays no role in the `norm_preserving_directional` attack, whose operative severity parameter is `directional_target_cosine` (`= 0.0` here, confirmed both in `summary.json` and in every active row's `directional_target_cosine` column of the trace). The trace's own verification fields settle this decisively (§3): `delta_attack_norm_ratio` and `realized_cosine_to_honest` are populated only on active attack rows and take values consistent with a genuinely active, correctly-parameterized directional attack, not a no-op.

## 3. Norm-Preservation Verification

Not assumed from the attack's name — verified directly from `update_l2_norm` and the trace's own dedicated verification field.

**`delta_attack_norm_ratio`** (the ratio of the submitted attacked delta's norm to the honest delta's norm it replaced) is present on every active `client_002` row and is `1.0` to within floating-point precision at every single one of the 30 post-attack rounds — e.g. `0.9999999999900032` (round 20), `1.0000000001075486` (round 43), `0.9999999998205212` (round 49). This is the trace's own direct, per-round confirmation that the attack is genuinely norm-preserving, not an assumption from its name.

**`update_l2_norm(client_002)` itself**, independently, tells the same story with a nuance: pre-attack (rounds 0–19) mean `5.5097`, sample stdev `0.3139` (20 observations, identical to `EXP_016`'s calibration, §1). Post-attack, the value stays close to this baseline at first (`5.3945` at round 20, essentially indistinguishable from honest variation) but **drifts gradually upward over the remaining 29 rounds**, reaching `6.5639` by round 49 — a real, monotonic-ish climb, not noise. This is not a contradiction of norm preservation: `delta_attack_norm_ratio` confirms the *submitted delta* is always rescaled to match that round's *own honest counterfactual delta's* norm exactly; what drifts is the *honest counterfactual's own norm itself*, because the global model `client_002`'s local training starts from has been diverging from what it would have been absent the attack, ever since round 20. This is the same mechanism already documented for the directional attack at seed 42 (`analysis/EXP_015_Directional_Attack_Traceability_Analysis.md`), now reproduced at an independent seed.

## 4. FedTrace v3 Predicted Malicious Client

`client_002`.

## 5–9. First Anomaly / Confirmation / Onset / Onset Error / Detection Delay

| | Value |
|---|---|
| First anomalous round | 20 |
| Confirmation round (`P=3`) | 22 |
| Estimated onset | 20 |
| True attack onset | 20 |
| Onset localization error | **0** |
| Detection delay | **2** |

(Driven entirely by `z_t_res` — see §15.)

## 10. Exhaustive Honest-Client False-Positive Analysis

All 5 clients' `z_m` and `z_t_res` were computed and inspected at every round with a defined signal — every client-round in the complete 250-row trace, not a sample.

**`z_t_res`:** reusing `EXP_016`'s already-verified per-client historical calibration (`D_k` depends only on rounds 1–19, bit-identical at seed 123 regardless of attack family, §1) — `client_000` and `client_001`'s residuals are more negative than their own historical center at every one of the 30 post-attack rounds (never below `0` after clipping); `client_003` is the literal round median (`d=0`) at **every single one of the 30 post-attack rounds** (a stronger, more consistent pattern than in either Gaussian-family run); `client_004`'s residual never exceeds its own historical mean. **Maximum honest `z_t_res` observed anywhere post-attack: `0.0`** (all four honest clients). Pre-attack, the same calibration-window values already exhaustively checked in `analysis/EXP_016_STAGE_B_GAUSSIAN_SEED123_VALIDATION.md` §5 apply unchanged (maximum `2.53`, `client_000`, round 1) — never re-derived here, reused because the underlying rounds are identical.

**`z_m`:** a data-entry/calibration cross-check caught and corrected during this analysis is worth reporting explicitly for transparency: `client_001`'s pre-attack calibration mean was initially miscopied from a *different* seed's figure, which would have (incorrectly) shown `client_001` crossing `z_m > 3` at round 20 itself. Recomputing directly from `EXP_018`'s own (seed-123, bit-identical to `EXP_016`'s) pre-attack `update_l2_norm` values gives the correct calibration: mean `6.0674`, stdev `0.4222` — under which `client_001`'s post-attack `z_m` never exceeds `1.04` (round 26). With every client's calibration independently re-verified by direct summation from this run's own trace data: **maximum honest `z_m` observed anywhere: `1.84`** (`client_000`, round 47). No honest client's `z_m` or `z_t_res` ever exceeds `3` at any round, pre- or post-attack.

**No honest client is confirmed before `client_002` — no honest client is ever confirmed at all.**

- Honest persistent false-positive count: **0**
- Honest persistent false-positive rate: **0/4 (0%)**

## 11. Peer Baseline

`naive_flag` is `False` for all 250 rows of the trace. **The peer-cosine baseline detects no one** — the same failure mode already documented for this baseline against both prior attack families (`EXP_014`, `EXP_015`, `EXP_016`, `EXP_017`).

## 12. Magnitude-Only (`z_m`)

| | Value |
|---|---|
| Predicted client | `client_002` |
| First anomalous round | **47** |
| Confirmation round | **49** |
| Estimated onset | **47** |
| Honest persistent FP | 0 |

`z_m(client_002)` stays below `3` for the first 27 post-attack rounds (peaking gradually from `0.37` at round 20 to `2.89` at round 46 as the honest-counterfactual drift described in §3 accumulates), crosses `3` for the first time at round 47 (`z=3.06`), and stays above `3` for the remaining three rounds of the run (`48`: `3.26`; `49`: `3.36`) — exactly `P=3` consecutive rounds, confirming at the very last round of the observed 50-round window. **Attribution is still correct** (no other client ever approaches threshold), but detection is extremely late: onset error `27`, detection delay `29`. This is a materially *better* outcome for magnitude-only than at seed 42 (`analysis/EXP_015_Directional_Attack_Traceability_Analysis.md`: the norm signal there only touched threshold once, at round 49, and never sustained 3 consecutive rounds — i.e. it would **not** have confirmed at all under this same `P=3` rule) — a genuine, seed-dependent difference in how quickly the honest-counterfactual drift accumulates, reported exactly as found, not smoothed toward either extreme.

## 13. Temporal-Residual-Only (`z_t_res`)

| | Value |
|---|---|
| Predicted client | `client_002` |
| First anomalous round | **20** |
| Confirmation round | **22** |
| Estimated onset | **20** |
| Honest persistent FP | 0 |

`z_t_res(client_002)` is `9.19` at round 20 itself and remains far above `3` for the entire post-attack window (minimum `5.62`, round 49) — the same immediate, decisive separation already established for the directional attack family at seed 42 (`EXP_015`), now confirmed at an independent seed.

## 14. Full FedTrace v3 (`max(z_m, z_t_res)`)

| | Value |
|---|---|
| Predicted malicious client | `client_002` |
| First anomalous round | 20 |
| Confirmation round | 22 |
| Onset estimate | 20 |
| Onset error | **0** |
| Detection delay | **2** |
| Honest persistent FP | 0 |

Identical to temporal-residual-only in every respect — `z_t_res` dominates the combined score at every single anomalous round (§15), so combining with `z_m` neither helps nor hurts the primary outcome here.

## 15. Signal Dominance

For every one of `client_002`'s 30 post-attack anomalous rounds, `z_m` and `z_t_res` were compared directly, not assumed:

| Comparison | Count | Percentage |
|---|---|---|
| `z_t_res > z_m` | **30** | **100%** |
| `z_m > z_t_res` | 0 | 0% |
| Tied | 0 | 0% |

`z_t_res` ranges `5.62`–`9.19` across the post-attack window; `z_m` ranges `0.25`–`3.36` over the same window and only exceeds `3` at all in the final three rounds (§12) — even there, `z_m`'s peak (`3.36`, round 49) stays well below `z_t_res`'s trough (`5.62`, round 49) at the same round. **`z_t_res` dominates every single anomalous round; `z_m` never wins, not even at the rounds where it is itself anomalous.** This was determined from the actual per-round data, not assumed from the attack's design intent.

## 16. Precision / Recall / F1

Same convention as `EXP_016`/`EXP_017` (malicious client-round event framing; TP = malicious client correctly among confirmed detections, FP = honest clients confirmed, FN = malicious client never confirmed):

| Detector | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| A. Peer-cosine baseline | 0 | 0 | 1 | — | 0.0 | — |
| B. Magnitude-only | 1 | 0 | 0 | 1.0 | 1.0 | 1.0 |
| C. Temporal-residual-only | 1 | 0 | 0 | 1.0 | 1.0 | 1.0 |
| D. FedTrace v3 | 1 | 0 | 0 | 1.0 | 1.0 | 1.0 |

Precision/recall/F1 do not penalize lateness (that is what onset error and detection delay are for, §12) — B, C, and D all attribute correctly and are therefore scored identically on this metric despite the large gap in detection speed.

## 17. EXP_018 vs. EXP_008 Performance

| Metric | `EXP_018` (attacked) | `EXP_008` (honest) | Δ (018 − 008) |
|---|---|---|---|
| Test loss | 1.858175 | 1.465440 | **+0.392735** |
| Test Top-1 | 0.5310 | 0.5683 | **−3.73 pp** |
| Test Top-5 | 0.8140 | 0.8779 | **−6.39 pp** |
| Macro precision | 0.588985 | 0.633761 | **−4.48 pp** |
| Macro recall | 0.5311 | 0.5683 | **−3.72 pp** |
| Macro F1 | 0.506105 | 0.548573 | **−4.25 pp** |
| Best validation Top-1 | 0.5510 (round 48) | 0.5826 (round 47) | **−3.16 pp** |

**Unlike the Gaussian family (`EXP_016`/`EXP_017`, essentially neutral), this is a real, substantial, consistently unfavorable degradation across every metric** — no metric improves under the attack, so there is no ambiguity here requiring the "do not interpret an increase as a benefit" caution. Validation Top-1 trajectories for rounds 14–26 (inspected directly from both runs' `round_history.csv`) confirm this is not merely a final-round artifact: both runs track closely through round 19 (`EXP_018`: `0.4372`; `EXP_008`: `0.4482`), diverge modestly right at round 20 (`0.4496` vs. `0.4608`), and the gap widens steadily thereafter (round 24: `0.462` vs. `0.4928`, already a ~3pp gap). This matches, and independently reproduces at a new seed, the "real, sustained, boundary-localized global-model degradation" already documented for this attack family at seed 42 (`analysis/EXP_015_Directional_Attack_Traceability_Analysis.md`).

## 18. Pre-Registered Per-Run Criteria

| # | Criterion | Result |
|---|---|---|
| 1 | `predicted_client == client_002` | **PASS** |
| 2 | No honest client confirmed before `client_002` | **PASS** |
| 3 | Onset error `<= 2` rounds | **PASS** (0) |
| 4 | Detection delay `<= 4` rounds | **PASS** (2) |
| 5 | Zero honest persistent false positives | **PASS** (0) |
| 6 | No FedTrace parameter changed | **PASS** |

## 19. Overall EXP_018 Verdict

**PASS.** All six pre-registered criteria are satisfied using FedTrace v3's full combined score (§14), with no reinterpretation after seeing the result and no detector modification of any kind. Criteria 3 and 4 are evaluated against the full `FedTraceV3Score`, not against magnitude-only alone (§12's much slower, marginal magnitude-only result is reported in full for scientific completeness, §20 below, but is not itself the basis for the PASS/FAIL determination).

## 20. Scientific Interpretation: Does the Combination Help, and Does the Design Intent Hold?

Determined from the actual trace, not assumed in advance:

- **Does `z_m` react?** Yes, but weakly and very late — it crosses threshold only in the final 3 of 30 post-attack rounds (§12), a real but marginal signal driven by the honest-counterfactual drift described in §3, not by the attack directly manipulating magnitude (which it does not — `delta_attack_norm_ratio ≈ 1.0` throughout, §3).
- **Does `z_t_res` react?** Yes, immediately and decisively — anomalous from round 20 itself, confirmed by round 22 (§13).
- **Which one provides detection?** `z_t_res`, entirely — it dominates 100% of anomalous rounds (§15), and the combined detector's onset/delay (§14) are identical to `z_t_res` alone.
- **Does the combination improve or merely preserve detection?** Neither improves nor degrades it here: `FedTraceV3Score` produces the exact same onset/confirmation as `z_t_res` alone, because `z_m` never exceeds `z_t_res` at any round where either is anomalous. This is consistent with the "coverage, not fusion" framing already established for FedTrace v3 across every prior Stage A/B analysis in this series (`analysis/FEDTRACE_V3_RETROSPECTIVE_ANALYSIS.md` §10 and elsewhere) — the combiner did not need to trade off between two competing strong signals here; one component was simply overwhelmingly more informative than the other for this attack family.
- **Does the result support FedTrace v3's intended complementary design?** Yes, in the specific sense the design was built for: a magnitude-preserving attack that would defeat (or, as seen here, only very marginally and belatedly inform) a magnitude-only detector is caught immediately by the temporal-residual component instead, without requiring any advance knowledge of which attack family would occur or any per-attack retuning of the frozen detector. This is exactly the cross-attack coverage property `research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md` was designed to provide, now independently confirmed (not merely redesigned around) on an unseen seed.

## 21. Independent Validation Discipline

**`EXP_018` was not used to design FedTrace v3.** Every formula and threshold applied in §4–§16 was frozen before this run's trace was read. **No result in this document modified, or would have modified, the frozen detector** — including the magnitude-only calibration cross-check in §10, which was corrected against this run's own data and did not alter any production code, config, or frozen formula (`src/analysis/fedtrace_detector.py` was not touched; the correction was a manual-arithmetic check performed for this document only).

## 22. Next Step

`EXP_018` is a technically valid run: attack activation, deactivation, trace schema, norm-preservation verification, and honest-phase fidelity (§1) all check out with no unexplained anomaly. No implementation failure occurred.

Per the frozen Stage B protocol, with `EXP_018` PASSing (§19), **the Stage B protocol permits proceeding to `EXP_019`, the second and final directional-family validation run** (seed 2026, paired with `EXP_009`), to complete the directional attack family's cross-seed matrix — mirroring the Gaussian family's already-completed `EXP_016`/`EXP_017` pair. A directional-family cross-seed verdict (`DIRECTIONAL FAMILY VALIDATED` or `NOT VALIDATED`, per `research_plans/FEDTRACE_STAGE_B_INDEPENDENT_VALIDATION_PLAN.md` §11) cannot yet be reached — it requires both seeds, and `EXP_019` has not been run. **`EXP_019`'s config is not created by this document**, per this task's explicit restriction.
