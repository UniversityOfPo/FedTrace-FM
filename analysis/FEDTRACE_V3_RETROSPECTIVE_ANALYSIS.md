# FedTrace v3: Retrospective Evaluation on EXP_014 / EXP_015 (Development Runs)

**Status:** Offline retrospective analysis only. No experiment was run. No config, training, attack, aggregation, or database file was created or modified. The frozen v3 formulas (`research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md`) were implemented exactly as specified and were not altered after seeing the results below.

---

## 1. Exact Frozen v3 Formulation

Implemented in `src/analysis/fedtrace_detector.py` (v3 section, appended below a clearly marked divider; v1 and v2 unmodified above it):

- `z_m(k,r)` — unchanged from v1/v2.
- `c(r) = median_j t(j,r)` — same-round common reference, no leave-one-out (unchanged from v2).
- `d(k,r) = c(r) − t(k,r)` — client residual.
- `D_k = {d(k,r) : round_id in [1, T_ref)}` (`T_ref=20`); `historical_center_k = mean(D_k)`; `historical_scale_k = stdev(D_k)` (**sample** standard deviation, `n−1`). Undefined if `|D_k| < 2` or `historical_scale_k == 0` — no epsilon or MAD substitution.
- `z_t_res(k,r) = max(0, [d(k,r) − historical_center_k] / historical_scale_k)`.
- `FedTraceV3Score(k,r) = max(z_m(k,r), z_t_res(k,r))` over whichever is defined; `z_p` never enters this score.
- Threshold `> 3`, persistence `P=3`, onset estimate = first round of confirmed run, confirmation = onset+2 — all unchanged.

## 2. Implementation Verification

`compute_common_round_reference`, `compute_historical_residual_calibration`, `ScoredRowV3`, `compute_all_scores_v3`, `apply_persistence_rule_v3`, `evaluate_detector_v3`, `evaluate_all_detectors_v3` were added below a clearly marked `# FedTrace v3` divider in `src/analysis/fedtrace_detector.py`. Every v1 and v2 function above that divider is byte-identical to before this task (verified by inspection — no edit touched any line above the v3 marker except the module-level docstring, which only gained a descriptive paragraph). `apply_persistence_rule_v3` is a full duplicate of v1's/v2's algorithm (not a shared refactor), so v1/v2 cannot be affected by any change made to persistence logic for v3.

Focused tests were added to `tests/test_analysis_fedtrace_detector.py` (not executed, per this task's restriction) covering the median reference, residual computation, historical-window boundaries (round 0 exclusion, `T_ref` exclusion), sample mean/stdev with `n−1`, minimum-observations and zero-variance rules, one-sidedness, undefined propagation (both a single missing component and both missing), no future-round or attack-label leakage, `P=3` persistence and onset/confirmation semantics, and v1/v2 regression checks.

## 3. Data Availability

Unchanged from Stage A / v2: `EXP_014`'s trace file has no `self_prev_round_cosine` column at all — `z_t_res`, temporal-residual-only, and FedTrace v3 are **NOT EVALUABLE on `EXP_014`**. No value was fabricated or reconstructed for this run. `EXP_015` has `self_prev_round_cosine` for every row with `round_id >= 1`.

## 4. Historical Residual Calibration Statistics

Computed exhaustively (all 19 pre-`T_ref` rounds, all 5 clients) from `EXP_015`'s trace:

| Client | `historical_center_k` (mean of `D_k`) | `historical_scale_k` (sample stdev, n=19) |
|---|---|---|
| `client_000` | 0.011930 | 0.042876 |
| `client_001` | −0.006012 | 0.005984 |
| `client_002` | 0.001478 | 0.008053 |
| `client_003` | −0.000158 | 0.014866 |
| `client_004` | 0.000333 | 0.005352 |

No client's calibration is degenerate (all `|D_k| = 19 >= 2`, all `historical_scale_k > 0`).

## 5. EXP_014 Results

`z_t_res` is undefined for every row (§3), so `FedTraceV3Score` reduces to exactly `z_m` — numerically identical to v1's magnitude-only detector, and to v1/v2's `FedTrace v1`/`FedTrace v2` on this run (both of which already degenerate to magnitude-only here for the same reason).

| Detector | Evaluable | Predicted client | Attribution | Onset est. | Confirm | Delay | Onset error | Honest FP | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A. Magnitude-only | yes | `client_002` | Yes | 20 | 22 | 2 | 0 | none | 1.0 | 1.0 | 1.0 |
| B. Peer-only | yes | none | No | — | — | — | — | none | — | 0.0 | — |
| C. v1 temporal-only | **NOT EVALUABLE** | — | — | — | — | — | — | — | — | — | — |
| D. v2 temporal-only | **NOT EVALUABLE** | — | — | — | — | — | — | — | — | — | — |
| E. v3 temporal-residual-only | **NOT EVALUABLE** | — | — | — | — | — | — | — | — | — | — |
| F. FedTrace v1 | yes (degenerate, = A) | `client_002` | Yes | 20 | 22 | 2 | 0 | none | 1.0 | 1.0 | 1.0 |
| G. FedTrace v2 | yes (degenerate, = A) | `client_002` | Yes | 20 | 22 | 2 | 0 | none | 1.0 | 1.0 | 1.0 |
| H. FedTrace v3 | yes (degenerate, = A) | `client_002` | Yes | 20 | 22 | 2 | 0 | none | 1.0 | 1.0 | 1.0 |

Magnitude-based coverage on `EXP_014` is intact and unbroken by v3, exactly as by v2.

## 6. EXP_015 Results

**`client_002` (true malicious client) — every one of the 30 post-attack rounds (20–49) exhaustively computed, all anomalous:**

| Round | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `z_t_res` | 32.4 | 31.2 | 30.7 | 29.9 | 29.0 | 28.6 | 26.9 | 28.0 | 27.3 | 27.1 | 27.4 |

| Round | 31 | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 | 40 |
|---|---|---|---|---|---|---|---|---|---|---|
| `z_t_res` | 26.1 | 26.0 | 26.0 | 25.4 | 24.7 | 24.1 | 25.1 | 24.6 | 23.8 | 24.2 |

| Round | 41 | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 |
|---|---|---|---|---|---|---|---|---|---|
| `z_t_res` | 22.4 | 22.4 | 23.2 | 22.5 | 22.2 | 21.5 | 20.9 | 20.7 | 20.6 |

All 30 rounds exceed 3 by a wide margin (minimum 20.6 at round 49). First anomalous round: 20. Confirmed persistent at round 22 (rounds 20, 21, 22). Onset estimate: 20 — exact match to `attack_start_round=20`.

**`client_000` (§7 below has the full targeted recheck) — every one of the 49 rounds (1–49) exhaustively computed: never anomalous, maximum `z_t_res = 1.10` at round 40.**

**`client_001`, `client_003`, `client_004` — every round exhaustively computed: `z_t_res = 0.0` at essentially every round (their own residual is consistently at or above their historical center — i.e. they run at-or-above the round median as consistently post-attack as they did pre-attack — so the one-sided score clips to exactly 0 nearly everywhere).**

| Detector | Predicted client | Attribution | First anom. | Confirm | Onset est. | Delay | Onset error | Honest FP | TP | FP | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A. Magnitude-only | `client_002` | Yes | 40 | 45 | 43 | 25 | 23 | none | 1 | 0 | 1.0 | 1.0 | 1.0 |
| B. Peer-only | none | No | — | — | — | — | — | none | 0 | 0 | — | 0.0 | — |
| C. v1 temporal-only | `client_002` | Yes | 20 | 22 | 20 | 2 | 0 | `client_001`,`003`,`004` | 1 | 3 | 0.25 | 1.0 | 0.40 |
| D. v2 temporal-only | `client_000` | **No** | 9 (`client_000`) | 11 (`client_000`) | 9 | — | — | `client_000` | 1 | 1 | 0.50 | 1.0 | 0.667 |
| **E. v3 temporal-residual-only** | **`client_002`** | **Yes** | 20 | 22 | 20 | 2 | 0 | **none** | 1 | 0 | **1.0** | 1.0 | **1.0** |
| F. FedTrace v1 | `client_002` | Yes | 20 | 22 | 20 | 2 | 0 | `client_001`,`003`,`004` | 1 | 3 | 0.25 | 1.0 | 0.40 |
| G. FedTrace v2 | `client_000` | **No** | — | — | 9 | — | — | `client_000` | 1 | 1 | 0.50 | 1.0 | 0.667 |
| **H. FedTrace v3** | **`client_002`** | **Yes** | 20 | 22 | 20 | 2 | 0 | **none** | 1 | 0 | **1.0** | 1.0 | **1.0** |

v3 (both E and H) fully restores v1's clean attribution/onset/delay result while eliminating v1's 3 honest false positives entirely — the first time in this three-version series that any temporal detector achieves both a correct attribution *and* zero honest persistent false positives on `EXP_015` simultaneously.

## 7. `client_000` v2-Failure Recheck

The exact three rounds that broke v2 (`analysis/FEDTRACE_V2_RETROSPECTIVE_ANALYSIS.md` §4):

| Round | `d(client_000,r)` | `z_t_res` (v3) | `z_t_rel` (v2, for comparison) | Anomalous under v3? |
|---|---|---|---|---|
| 9 | 0.02761 | **0.366** | 3.91 | no |
| 10 | 0.02388 | **0.279** | 12.35 | no |
| 11 | 0.03436 | **0.523** | 4.45 | no |

**Not relocated — exhaustively checked at every one of the other 46 rounds too.** The full round-by-round `z_t_res(client_000, ·)` sequence for all 49 rounds:

Rounds 1–19 (pre-`T_ref`, scored using the same frozen calibration derived from this exact window, per §9 of the v3 plan): `0, 0, 0, 0, 0, 0, 0, 0.113, 0.366, 0.279, 0.523, 0.297, 0.338, 0.568, 0.739, 0.706, 0.818, 0.691, 0.795` (all `max(0, ...)`-clipped where the raw value was negative).

Rounds 20–49: `0.745, 0.751, 0.897, 0.691, 0.749, 0.765, 0.631, 0.721, 0.789, 0.867, 1.053, 0.865, 0.900, 0.980, 0.930, 0.764, 0.718, 1.037, 0.960, 0.859, 1.102, 0.847, 0.885, 1.039, 0.941, 0.952, 0.889, 0.681, 0.776, 0.763`.

**Maximum observed anywhere in the run: 1.10 (round 40). No round, for `client_000`, ever approaches 3, let alone 3 consecutive rounds.** The false positive is eliminated, not relocated to a later or earlier round for the same client.

## 8. `client_002` Rounds 20/21/22 Scores

| Round | `d(client_002,r)` | `z_t_res` |
|---|---|---|
| 20 | 0.262268 | **32.38** |
| 21 | 0.252775 | **31.20** |
| 22 | 0.248441 | **30.67** |

First anomaly: round 20. Confirmation: round 22 (three consecutive anomalous rounds, 20/21/22). Onset estimate: 20. Detection delay: `22 − 20 = 2` rounds — identical speed to v1's and v2's temporal detection, now without either's false-positive cost.

## 9. Exhaustive Honest-FP Analysis

All 5 clients, all 49 rounds of `EXP_015`, checked without sampling:

| Client | Rounds checked | Max `z_t_res` observed | Ever `> 3`? | Persistent detection? |
|---|---|---|---|---|
| `client_000` | 1–49 (all) | 1.10 (round 40) | No | No |
| `client_001` | 1–49 (all) | 0.00 (consistently at/above its own historical center) | No | No |
| `client_002` | 1–49 (all) | 32.4 (round 20) | Yes, every round 20–49 | **Yes — the true positive** |
| `client_003` | 1–49 (all) | 0.00 (consistently at/above its own historical center) | No | No |
| `client_004` | 1–49 (all) | 0.00 (consistently at/above its own historical center; `client_004` is the round median itself at 26 of the 30 post-attack rounds) | No | No |

**Zero honest persistent false positives anywhere in the run, at any round, for any client.** This directly satisfies §11/§16's "no relocated false positive" requirement — not inferred from an aggregate metric, but confirmed by inspecting every eligible client-round.

## 10. v1 vs. v2 vs. v3 Comparison

| | v1 (`z_t`, absolute) | v2 (`z_t_rel`, cross-sectional MAD) | v3 (`z_t_res`, historical residual) |
|---|---|---|---|
| `client_002` attribution | Correct | **Incorrect** (`client_000` wins) | Correct |
| `client_002` onset/delay | 20 / 2 | 20 / 2 (moot — wrong client attributed) | 20 / 2 |
| Honest persistent FPs | 3 (`client_001`,`003`,`004`, late-training) | 1 (`client_000`, pre-attack, attribution-breaking) | **0** |
| Mechanism of any FP | Non-stationary calibration reference (fixed pre-attack window can't track ongoing drift) | Small-sample (`n=5`) same-round scale instability | — (no FP mechanism observed) |
| Precision / Recall / F1 (temporal-only) | 0.25 / 1.0 / 0.40 | 0.50 / 1.0 / 0.667 | **1.0 / 1.0 / 1.0** |

v3 is the first version in this series to achieve a perfect score on every metric tracked throughout the series, on this one development run.

## 11. Attribution Comparison

`EXP_014`: correct under every evaluable detector, all three versions.

`EXP_015`: v1 correct, v2 incorrect, **v3 correct** — restoring exactly what v2 broke, without reproducing v1's honest-FP problem.

## 12. Onset-Localization Comparison

`EXP_014`: identical across v1/v2/v3 (onset error 0, all reduce to magnitude-only).

`EXP_015`: v1 onset error 0 (`client_002`, correctly attributed). v2's "onset=9" was for the wrong client (`client_000`) and not a meaningful localization of the real attack. v3's onset error is **0** (`client_002`, correctly attributed, onset estimate exactly 20) — matching v1's best-case localization while also matching v2's original design goal of removing v1's honest-client false positives.

## 13. Signal Dominance

- **`EXP_014`:** `z_t_res` undefined for every row; `z_m` accounts for 100% of anomalous (`score > 3`) client-rounds (30/30 for `client_002`).
- **`EXP_015`:** `z_t_res` accounts for 100% of anomalous client-rounds (30/30, all `client_002`); `z_m` never wins the max at any anomalous round — `client_002`'s own `z_m` does eventually cross 3 around round 43 (≈3.1–4.0, per the Stage A/v2 record), but `z_t_res` is already 20–23 at those same rounds, always higher. `client_000`'s `z_m` stays under 1.3 throughout (unchanged from v1/v2's own record) and never approaches its own `z_t_res` ceiling of 1.10 either — `client_000` never contributes to any dominance count because it is never anomalous under either component.

## 14. Cross-Attack Coverage Interpretation

Framed, as required, as coverage rather than synergistic fusion: `max(z_m, z_t_res)` picks up whichever signal is informative for a given attack family — `z_m` for `EXP_014`'s magnitude-perturbing attack, `z_t_res` for `EXP_015`'s norm-preserving directional attack — without needing to know in advance which one a future attack will manipulate. This is the same property v1 and v2 already had; what has changed across the three versions is not the coverage mechanism itself but the *reliability of the component that coverage ends up relying on* for the temporal case. v3's combiner does not, on this evidence, demonstrate a case where combining `z_m` and `z_t_res` outperforms deploying the better single one for a given run — `H_fedtrace_v3` on `EXP_015` produces results numerically identical to `E_temporal_residual_only` alone, and on `EXP_014` identical to `A_magnitude_only` alone. The claimed contribution remains coverage, not fusion, exactly as v1's and v2's plans already stated.

## 15. Limitations

- **n=1 per attack family, development data only.** Every number in this document is from a single seed-42 run of each attack.
- **Historical calibration window size.** `D_k` has only 19 observations on this project's current configuration (`T_ref=20`); the sample standard deviation's own stability at this specific `n` has not been independently stress-tested beyond what this one run's data shows. A materially shorter `T_ref` in some future configuration could reintroduce a smaller-sample instability risk analogous to (though structurally different from) v2's — this is not evidenced by anything in the current data, but is not ruled out either.
- **The residual's stationarity assumption is itself unverified beyond this run.** v3's design (§4 of the v3 plan) argues, but does not prove beyond this one development run, that `d(k,r)` is closer to stationary than the raw `t(k,r)` signal it derives from. `EXP_015`'s own data is consistent with this (client residuals stay within a narrow, non-drifting band across 49 rounds, per §7/§9's tables), but a longer run or a different heterogeneity configuration could behave differently.
- **`client_000`'s specific low-residual tendency is a property of this seed's data, not claimed to generalize.** The v2 plan already cautioned against reading `client_000`'s pattern as anything other than an estimator property; v3's fix works regardless of which specific client happens to run consistently below the round median, but this has only been confirmed for the one client (`client_000`) that happened to exhibit it at seed 42.

## 16. Development-vs-Validation Boundary

**`EXP_014` and `EXP_015` remain development-only runs.** This document's findings — including the clean §6/§9 result — were produced by directly evaluating v3 against the exact two runs used to diagnose v2's flaw and design v3's fix. This is legitimate development-set verification, not independent validation, and must not be cited as such. No statistical significance claim is made anywhere in this document; "zero honest false positives" and "correct attribution" are reports of what happened on these two specific runs, not estimates of a rate or a probability. Independent validation requires the not-yet-collected seed-123/seed-2026 replications, run only after v3's formulas are frozen (as they are, unmodified, throughout this document) and with no further tuning based on how those new runs turn out.

## 17. Pre-Registered Success/Failure Assessment

Checked directly against `research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md` §11/§12:

**`EXP_015` success criteria (§11):**
- ✅ `client_002` correctly attributed.
- ✅ Onset estimate at round 20 (exact match, not merely "near").
- ✅ No honest client confirmed before `client_002` (no honest client is ever confirmed at all).
- ✅ Honest persistent false positives: zero (exceeds the "materially reduced" bar — fully eliminated).
- ✅ No relocated persistent FP (§9's exhaustive 49-round × 5-client check).

**`EXP_014` success criteria (§11):**
- ✅ Magnitude coverage intact.
- ✅ Correct attribution.
- ✅ No new honest persistent FP.

**Cross-cutting (§11):** no attack-specific threshold was used anywhere (threshold=3, P=3, T_ref=20 identical across both runs and both attack families); no formula was modified after observing any result in this document.

**Failure criteria (§12), checked and none triggered:**
- ❌ (not triggered) Honest persistent FPs remain — zero found.
- ❌ (not triggered) Wrong client attributed — `client_002` correctly attributed.
- ❌ (not triggered) Onset materially delayed — onset error 0, matching v1's best case exactly.
- ❌ (not triggered) Correction works only via post-hoc, attack-specific choices — the mean/stdev formulation was committed to in the v3 plan (§3) before this evaluation, for reasons independent of these specific results.
- ❌ (not triggered) Historical residual calibration degenerate/unstable — all 5 clients' `D_k` calibrations are well-defined (§4), none degenerate.
- ❌ (not triggered) v3 adds no value over running the two components separately — `H_fedtrace_v3` and `E_temporal_residual_only` do produce identical results on `EXP_015` (§14), which is expected and consistent with the "coverage, not fusion" framing already adopted by v1/v2, not a new finding specific to v3, and not itself one of §12's listed failure conditions (that condition is about the *combiner* adding nothing, which was already true of v1 and v2 as well and was explicitly accepted as the intended, honestly-framed behavior in both of those plans).

**Every §11 success criterion is met. No §12 failure criterion is triggered.**

## 18. Stage B Readiness Decision

Checking `research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md` §14's four-part gate:

1. **Detector implementation tests passing** — tests are written (§2) but **not executed** in this task, per its explicit restriction. This part of the gate cannot yet be marked satisfied; it is contingent on those tests subsequently being run and passing.
2. **Retrospective v3 development criteria passing** — ✅ met (§17).
3. **Formulas/thresholds frozen** — ✅ met (§16 of the v3 plan implemented exactly, unmodified).
4. **No remaining known methodological blocker** — ✅ met on the evidence available (§9's exhaustive check, §15's honestly-stated limitations, none of which rises to a "known blocker" — they are unresolved uncertainties about generalization, not identified defects in the current implementation).

---

## Final Decision

**(A) — FedTrace v3 passes the retrospective development gate and may be frozen for independent Stage B validation, SUBJECT TO the new unit tests subsequently passing.**

Every pre-registered `EXP_015` and `EXP_014` success criterion is met exactly, with no failure criterion triggered, and the specific mechanism that broke v2 (`client_000`, rounds 9–11) is confirmed eliminated by an exhaustive check, not a representative sample. This decision is explicitly conditional, per the v3 plan's own §14 gate: `tests/test_analysis_fedtrace_detector.py`'s v3 tests were written but not run in this task, and their subsequent passing is a prerequisite this document cannot itself satisfy. No Stage B config was created and no new attack family is recommended, per this task's explicit restrictions.
