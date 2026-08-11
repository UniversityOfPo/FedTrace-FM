# FedTrace v2: Retrospective Evaluation on EXP_014 / EXP_015 (Development Runs)

**Status:** Offline retrospective analysis only. No experiment was run. No config, training, attack, aggregation, or database file was created or modified. The frozen v2 formulas (`research_plans/FEDTRACE_V2_TEMPORAL_CALIBRATION_PLAN.md`) were implemented exactly as specified and were **not** altered after seeing the results below — including the central, unfavorable finding in §4/§6, which is reported as found, not adjusted away.

---

## 1. Frozen v2 Formulation

Implemented in `src/analysis/fedtrace_detector.py` (v2 section, appended below a clearly marked divider; v1 unmodified above it):

- `z_m(k,r)` — **unchanged from v1**: per-client longitudinal calibration over rounds `< T_ref=20`, sample mean/stdev.
- `z_t_rel(k,r) = max(0, [center_t(r) − t(k,r)] / scale_t(r))`, where `center_t(r) = median_j t(j,r)`, `scale_t(r) = 1.4826 × median_j |t(j,r) − center_t(r)|`, computed across every client with a defined `self_prev_round_cosine` at round `r` (no leave-one-out, no historical window). Undefined if fewer than 2 clients have a value that round, or if the scale is exactly zero — never substituted with zero or an epsilon.
- `FedTraceV2Score(k,r) = max(z_m(k,r), z_t_rel(k,r))` over whichever is defined; `z_p` is computed by v1's own unchanged functions for the peer-only ablation below but never enters this primary score.
- Threshold `> 3`, persistence `P = 3` consecutive rounds, onset estimate = first round of the confirmed run, confirmation round = onset + 2 — all unchanged from v1.

This is implemented exactly as specified in the v2 plan; nothing here was adjusted based on the results in §3–§6.

## 2. Data Availability

Same as Stage A (`analysis/FEDTRACE_STAGE_A_RETROSPECTIVE_ANALYSIS.md` §3): `EXP_014`'s trace file has no `self_prev_round_cosine` column at all — `z_t_rel`, corrected-temporal-only, and FedTrace v2 (which depends on `z_t_rel`) are **NOT EVALUABLE on `EXP_014`**. `update_l2_norm`/`peer_mean_cosine_similarity` are available in both runs. `EXP_015` has `self_prev_round_cosine` for every row with `round_id >= 1`.

No `self_prev_round_cosine` value was fabricated for `EXP_014` anywhere in this analysis.

## 3. EXP_014 Results

Since `z_t_rel` is undefined for every row of `EXP_014`, `FedTraceV2Score` reduces to exactly `z_m` for this run — **numerically identical to v1's magnitude-only detector.**

| Detector | Evaluable | Predicted client | Attribution correct | First anomalous | Confirmation | Onset est. | Delay | Onset error | Honest persistent FP | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A. Magnitude-only | yes | `client_002` | Yes | 20 | 22 | 20 | 2 | 0 | none | 1.0 | 1.0 | 1.0 |
| B. Peer-only | yes | none | No | — | — | — | — | — | none | — | 0.0 | — |
| C. Old temporal-only | **NOT EVALUABLE** | — | — | — | — | — | — | — | — | — | — | — |
| D. Corrected temporal-only | **NOT EVALUABLE** | — | — | — | — | — | — | — | — | — | — | — |
| E. FedTrace v1 | yes (degenerate, = A) | `client_002` | Yes | 20 | 22 | 20 | 2 | 0 | none | 1.0 | 1.0 | 1.0 |
| F. FedTrace v2 | yes (degenerate, = A) | `client_002` | Yes | 20 | 22 | 20 | 2 | 0 | none | 1.0 | 1.0 | 1.0 |

**No honest client ever receives a persistent detection on `EXP_014`, under any evaluable detector.** v2's magnitude coverage on this run is unchanged and unbroken, as expected (§5 of the v2 plan required `z_m` to remain untouched, and it was).

## 4. EXP_015 Results

This is where v2's correction was designed to matter, and where a critical, previously-undetected problem surfaced during this retrospective check.

**Corrected temporal-only (`z_t_rel` alone) — exhaustively re-verified for every pre-attack round (1–19) and at 8 post-attack rounds (20, 21, 22, 30, 35, 40, 45, 49):**

`client_002` (the true malicious client) is confirmed exactly as the v2 plan's own 6-round spot-check predicted: `z_t_rel` reaches 14.97 at round 20, stays far above 3 through round 49 (minimum observed 3.96, at round 49), confirmed persistent at **round 22** (rounds 20, 21, 22), onset estimate **20**.

**But `client_002` is not the earliest client confirmed.** Exhaustive round-by-round computation of `z_t_rel` for every honest client across rounds 1–19 (all 5 clients, all 19 rounds, full arithmetic reproduced below) found that **`client_000` — an honest client — is confirmed persistent at round 11, entirely before the attack begins at round 20:**

| Round | `center_t(r)` | `scale_t(r)` | `z_t_rel(client_000)` | Anomalous (`>3`)? |
|---|---|---|---|---|
| 8 | 0.3315 | 0.0075 | 2.23 | no |
| **9** | 0.3262 | 0.00706 | **3.91** | **yes** |
| **10** | 0.3164 | 0.00193 | **12.35** | **yes** |
| **11** | 0.3116 | 0.00772 | **4.45** | **yes** — 3rd consecutive → **persistence confirmed** |
| 12 | 0.2970 | 0.0232 | 1.06 | no (streak broken) |
| 13 | 0.2872 | 0.0252 | 1.05 | no |
| 14 | 0.2903 | 0.0100 | 3.62 | yes (new, isolated at this point) |
| 15–19 | — | — | 10.86, 5.58, 4.22, 5.36, 3.84 | yes (all 5 rounds) |

**First anomalous round: 9. Confirmation round: 11 (rounds 9, 10, 11 — three consecutive). Onset estimate: 9.** This is 9 rounds before `client_002`'s own confirmation at round 22, and 11 rounds before the attack even starts (`attack_start_round = 20`).

**Root cause, verified by hand (not merely inferred):** at round 9, the four other clients' `self_prev_round_cosine` values happened to cluster unusually tightly that round (`client_001=0.328`, `client_003=0.331`, `client_004=0.326`, within a 0.005 band), producing an extremely small median absolute deviation (`MAD = 0.00476`, `scale = 0.00706`). `client_000`'s own value (`0.2986`) is only `0.0276` below the median — a modest, unremarkable gap in absolute terms (smaller than several other honest-to-honest gaps observed at other rounds that did *not* cross threshold, e.g. round 1's `client_003` gap of `0.042` scored only `z=2.63`) — but because that round's scale happened to be so small, the same modest gap produced `z=3.91`. Round 10 is more extreme still: a scale of only `0.00193` (four clients within `0.006` of each other) turns `client_000`'s `0.0239` gap into `z=12.35`. **This is a small-sample statistical instability, not a real behavioral anomaly**: with only `n=5` clients per round, the median-absolute-deviation estimator itself has high round-to-round variance, and on any round where the 4 "non-outlier" clients happen to cluster by chance, the resulting `scale_t(r)` can be small enough to inflate an ordinary, non-malicious client's typical deviation into an enormous z-score.

**`client_001`, `client_003`, `client_004` do not cross 3 at any of the 19 pre-attack rounds or the 8 post-attack rounds checked.** They are consistently at or above the round median (their one-sided score clips to `0` per §3/§14 of the v2 plan) or, when below it, their deviation is small relative to that round's scale. `client_000` is checked further post-attack (rounds 20, 21, 22, 30, 35, 40, 45, 49: `z = 2.50, 2.25, 2.10, 2.90, 1.68, 2.00, 1.30, 1.06`) and never re-crosses 3 after round 19 — its only confirmed detection is the pre-attack one.

**FedTrace v2 (`max(z_m, z_t_rel)`) inherits this exactly.** `z_m` never exceeds `z_t_rel` at any anomalous round for either `client_000` (whose `z_m` stays under 1.3 throughout) or `client_002` (whose `z_m` doesn't cross 3 until round 40+, and even then stays below `z_t_rel`'s concurrent value) — so the full v2 combiner produces the identical per-client outcome as corrected-temporal-only alone.

| Detector | Evaluable | Predicted client | Attribution correct | First anom. (`client_000`) | Confirm (`client_000`) | First anom. (`client_002`) | Confirm (`client_002`) | Honest persistent FP | TP | FP | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A. Magnitude-only | yes | `client_002` | Yes | — | — | 40 | 45 | none | 1 | 0 | 1.0 | 1.0 | 1.0 |
| B. Peer-only | yes | none | No | — | — | — | — | none | 0 | 0 | — | 0.0 | — |
| C. Old temporal-only (v1) | yes | `client_002` | Yes | — | — | 20 | 22 | `client_001`, `client_003`, `client_004` | 1 | 3 | 0.25 | 1.0 | 0.40 |
| **D. Corrected temporal-only (v2)** | yes | **`client_000`** | **No** | 9 | **11** | 20 | 22 | **`client_000`** | 1 | 1 | **0.50** | 1.0 | 0.667 |
| E. FedTrace v1 | yes | `client_002` | Yes | — | — | 20 | 22 | `client_001`, `client_003`, `client_004` | 1 | 3 | 0.25 | 1.0 | 0.40 |
| **F. FedTrace v2** | yes | **`client_000`** | **No** | 9 | **11** | 20 | 22 | **`client_000`** | 1 | 1 | **0.50** | 1.0 | 0.667 |

For both D and F, since `attribution_correct=False`, onset-localization error and detection delay are not computed against `client_002` (the predicted client is `client_000`, not the true malicious client) — the metric is not defined in the way the pre-registered protocol intended, exactly per §10's instruction to mark this honestly rather than compute a number against the wrong client.

## 5. v1 vs. v2 Comparison

v2 **did** fix the specific problem it was designed for: `client_001`, `client_003`, and `client_004`'s persistent false positives (confirmed at rounds 48, 47, and 43 under v1's absolute temporal calibration) are gone under `z_t_rel` — none of the three crosses 3 at any checked round. v2 also preserved `client_002`'s detection speed exactly (confirmation still at round 22, onset still 20 — identical to v1's temporal-only result for the true positive).

v2 **did not** achieve a clean correction. It replaced three honest false positives with one different honest false positive (`client_000`), and that one is materially worse in a specific, load-bearing way: it fires **before the attack begins**, and it fires **before the true malicious client is ever detected**, which flips the earliest-confirmed-wins attribution rule against the correct answer. v1's three false positives, however numerous, never changed *which* client was predicted malicious (`client_002` was always confirmed first, at round 22, well before any of v1's false alarms at rounds 43–48). v2's single false positive does change the predicted client. **In the metric that matters most for this project's flagship goal ("identify WHICH client becomes malicious"), v2 is a regression relative to v1 on this run, despite improving the honest-false-positive count from 3 to 1.**

## 6. Honest False-Positive Comparison

| | v1 (old `z_t`, absolute/longitudinal) | v2 (`z_t_rel`, cross-sectional/robust) |
|---|---|---|
| Honest clients confirmed | `client_001`, `client_003`, `client_004` (3) | `client_000` (1) |
| Timing | Late-training (rounds 43, 45, 47 confirmed) | **Pre-attack** (round 11 confirmed, round 9 first anomalous) |
| Mechanism | Shared secular decline in `t`, not visible to a pre-attack-only calibration window | Small-sample (`n=5`) MAD instability producing a spuriously tiny scale on specific rounds |
| Effect on attribution | None — `client_002` still confirmed first (round 22 vs. rounds 43+) | **Attribution flips** — `client_000` confirmed at round 11, before `client_002`'s round 22 |

Fewer false positives in raw count; a strictly worse single false positive in consequence.

## 7. Attribution Comparison

`EXP_014`: unaffected either way (`client_002` correctly attributed under every evaluable detector, v1 or v2, error 0, delay 2).

`EXP_015`: v1 (`E_fedtrace_v1`) attributes correctly (`client_002`, onset error 0, delay 2) despite its 3 false positives, because none of them beat `client_002`'s round-22 confirmation. **v2 (`F_fedtrace_v2`) attributes incorrectly** — `client_000` is output as the predicted malicious client, a direct failure of this project's flagship "identify WHICH client" goal on the one run v2 was specifically built to fix.

## 8. Onset-Localization Comparison

`EXP_014`: identical under v1/v2 (onset error 0, both reduce to magnitude-only).

`EXP_015`: v1's onset estimate for the (correctly attributed) `client_002` is 20 — exact. v2's onset estimate, following its (incorrect) attribution, is **9**, for `client_000` — not a localization of the real attack at all, but the round at which an honest client's temporal score first happened to dip due to small-sample noise. Reporting an "onset error" number here (e.g., `|9 − 20| = 11`) would misleadingly imply v2 is merely 11 rounds off on the right client; it is not — it has localized the wrong event on the wrong client, and no error metric computed against `attack_start_round=20` is a meaningful characterization of that failure.

## 9. Signal-Dominance Analysis

- **`EXP_014`:** `z_t_rel` is undefined for every row; `z_m` accounts for 100% of anomalous (`score > 3`) client-rounds.
- **`EXP_015`:** `z_t_rel` accounts for effectively 100% of anomalous client-rounds — both `client_002`'s correct detection (every anomalous round, `z_t_rel` far exceeds `z_m`) and `client_000`'s spurious one (`z_m` for `client_000` never exceeds ≈1.3, nowhere close to `z_t_rel`'s 3.9–12.4 at the confirming rounds). `z_m` never wins the max at any anomalous round observed in `EXP_015`, including the rounds (43+) where `z_m(client_002)` itself eventually crosses 3 — `z_t_rel` is already far higher there too.

This confirms v2's coverage claim (§7 of the v2 plan) held up mechanically as designed — `max()` correctly deferred to whichever signal was informative per run — but also means `z_t_rel`'s flaw was never masked or diluted by `z_m`; the combiner offered no protection against it.

## 10. Cross-Attack Coverage Interpretation

The v2 plan's own framing (§7: "the combination's value is cross-attack coverage, not synergistic fusion") is borne out exactly as stated — and that framing's limit is now visible too. Coverage means `max()` picks up whichever single signal is informative for a given run; it does not, and on this evidence cannot, protect against that signal's own flaws. v1 already demonstrated this (the flawed `z_t` was inherited into `max(z_m,z_p,z_t)` undiluted); v2 demonstrates the same property with a *different* flaw in the temporal component. Fixing `z_t`'s calibration reference (the thing v2 set out to do) was necessary but not sufficient — the new calibration reference introduced its own, different vulnerability, and the combiner, exactly as designed, offered no defense against it either.

## 11. Limitations

- **n=1 per attack family, development data only** (§12 below elaborates the validation boundary). This finding — like every number in this series — is from a single seed-42 run of each attack.
- **Pre-attack round sampling.** Rounds 1–19 were checked exhaustively (all 19 rounds, all 5 clients) specifically because the round-9-11 anomaly, once found, made it clear that representative sampling (the approach used for the v2 plan's original 6-round check, and for Stage A's `z_p` characterization) is not sufficient for a signal whose scale statistic can vary sharply round to round at `n=5`. Post-attack rounds were checked at 8 representative rounds (20, 21, 22, 30, 35, 40, 45, 49), not all 30 — the same sampling-density caveat as Stage A's `z_p` analysis applies to any post-attack round not explicitly listed above; a full 30-round exhaustive check is a natural next step of any follow-up correction to `z_t_rel`, not performed here.
- **This specific failure mode (small-sample MAD instability) has not been checked for recurrence at other rounds beyond those sampled**, nor for whether it would recur, worsen, or improve at different seeds — that is explicitly out of scope for this development-only retrospective (§12).
- **The magnitude of the round-9-11 anomaly is itself somewhat sensitive to which 4 honest clients happen to cluster on a given round** — this is expected to vary across seeds and is not claimed to always implicate `client_000` specifically; it is reported as a property of the estimator (small-n MAD volatility), not as a client-specific finding about `client_000`.

## 12. Development-vs-Validation Boundary

**`EXP_014` and `EXP_015` remain development runs, exactly as designated in `research_plans/FEDTRACE_V2_TEMPORAL_CALIBRATION_PLAN.md` §9.** This document's finding (v2's attribution failure on `EXP_015`) is itself development-set evidence — discovered by inspecting the same run the v2 correction was designed against — and must not be read as an independent validation failure either. It is exactly the kind of result the development/validation boundary exists to catch *before* independent seeds are spent: a flaw found here is cheap to find and correct; the same flaw, undiscovered, reproduced identically at seeds 123 and 2026 (since it is a property of the `n=5`/MAD estimator, not of seed-42-specific noise) would have consumed the independent-validation budget confirming a problem that could have been caught first. No significance claim is made anywhere in this document.

## 13. Success/Failure Assessment Against the v2 Plan

Checking directly against `research_plans/FEDTRACE_V2_TEMPORAL_CALIBRATION_PLAN.md` §11/§12:

- ✅ `EXP_014` magnitude-based coverage is not broken (§11, met exactly).
- ❌ **"`EXP_015`'s malicious client remains attributable... with onset estimate at or very near the true `attack_start_round`"** (§11) — **not met**. `client_002` is not the predicted client at all under FedTrace v2; attribution is lost entirely, not merely delayed or mis-localized.
- ❌ **"Honest persistent temporal false positives are materially reduced, preferably to zero"** (§11) — **partially met, and this is exactly the trap the plan's own §12 warned about**: the count did drop from 3 to 1, but §12's second failure criterion — *"the correction merely relocates the false-positive problem... a different set of honest clients... still reach persistent detection"* — is triggered precisely: the honest-FP problem was relocated (from `client_001`/`003`/`004` to `client_000`) and changed in kind (pre-attack, not late-training), not genuinely resolved.
- ❌ **§12 failure criterion, "directional attack attribution is lost"** — **triggered directly**, as shown in §4/§7 above.
- ❌ **§12 failure criterion, "the correction merely shifts false positives to different honest clients"** — **triggered directly**, as shown in §6 above.

**By the v2 plan's own pre-registered criteria, this is a failure, not a pass.**

## 14. Next-Step Recommendation (exactly one)

Do not proceed to independent-seed validation with FedTrace v2 as currently specified. The corrected temporal calibration must itself be corrected — specifically, its robustness at `n=5` clients per round, where a median-absolute-deviation-based scale estimator is demonstrably volatile enough to convert an ordinary honest client's typical deviation into a >3 z-score on specific rounds — before any further validation step is justified. This is a scoped, mechanism-specific problem (not evidence that cross-sectional or robust temporal calibration is unworkable in general, and not evidence that the underlying multi-signal FedTrace direction should be abandoned — `client_002`'s detection speed and `EXP_014`'s magnitude coverage both remained excellent throughout).

---

## Decision

**(B) — v2 still has a methodological flaw that must be corrected before independent validation.**

Not (A): attribution demonstrably fails on `EXP_015`, the exact run the correction targeted, against the plan's own pre-registered success criteria (§13). Not (C): the flaw is specific and mechanistic (small-sample MAD instability in the new calibration reference), not a demonstration that the three-signal, honest-reference-calibrated FedTrace approach lacks value — magnitude-only coverage on `EXP_014` and the underlying temporal signal's detection speed on `EXP_015`'s true attack both remain strong; only the new scale estimator's small-`n` behavior needs correction. No new attack family is recommended or considered in this task, per its explicit restriction.
