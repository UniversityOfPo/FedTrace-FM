# EXP_017 — Stage B Independent Validation: Gaussian Attack, Seed 2026

**Status:** Independent validation analysis. `EXP_017` was **not** used to design, diagnose, or tune FedTrace v3. FedTrace v3 was applied here **exactly as frozen** — nothing was tuned after inspecting the trace, regardless of outcome. This is the second and final Gaussian-family Stage B run; combined with `EXP_016` (seed 123), it completes the cross-seed matrix for this attack family.

---

## 1. Pairing Verification

| Field | `EXP_017` | `EXP_009` | Match |
|---|---|---|---|
| `seed` | 2026 | 2026 | ✅ |
| `partition` | dirichlet | dirichlet | ✅ |
| `dirichlet_alpha` (config) | 0.1 | 0.1 | ✅ |
| `client_sample_counts` | `{000:5233, 001:12318, 002:9171, 003:6556, 004:11722}` | identical, value-for-value | ✅ (same partition realization) |
| `model` / `total_parameters` | resnet18_cifar / 11,220,132 | same | ✅ |
| `optimizer` / `learning_rate` | sgd / 0.01 | same | ✅ |
| `local_epochs` | 1 | 1 | ✅ |
| `communication_rounds_completed` | 50/50 | 50/50 | ✅ |
| `aggregation_strategy` | fedavg | fedavg | ✅ |
| `batch_size` | 32 | 32 | ✅ |

**The frozen Gaussian attack is confirmed to be the only scientific difference** — `EXP_017`'s `summary.json` has an `"attack"` block absent from `EXP_009`'s; every other tracked setting matches exactly.

## 2. Attack Correctness

`summary.json`: `attack_type=gaussian_update_noise`, `malicious_client_id=client_002`, `attack_start_round=20`, `attack_strength_rho=1.0`.

Full `client_round_trace.csv` (250 rows): `client_002`'s `attack_active` is `False` for `round_id < 20` and `True` for `round_id >= 20`. **No other client's `attack_active` is ever `True`.**

`update_l2_norm` (`client_002`): pre-attack (rounds 0–19) mean `6.0220`, sample stdev `0.3510` (20 observations). Post-attack: jumps to `8.4121` at round 20 and climbs to `9.4666` by round 49 — every post-attack value exceeds the pre-attack mean by more than `6.5` standard deviations. Same immediate, sustained signature as `EXP_014` and `EXP_016`.

## 3. Apply Frozen FedTrace v3

Applied exactly as frozen, per §1 of `research_plans/FEDTRACE_STAGE_B_INDEPENDENT_VALIDATION_PLAN.md`, with no parameter changes and no post-hoc tuning at any point.

## 4. Primary Result

| | Value |
|---|---|
| Predicted malicious client | `client_002` |
| Attribution correct | **Yes** |
| First anomalous round | 20 |
| Confirmation round | 22 |
| Estimated onset | 20 |
| True attack onset | 20 |
| Onset localization error | **0** |
| Detection delay | **2** |

## 5. Exhaustive False-Positive Analysis

All 5 clients' `z_m` and `z_t_res` were computed and inspected at every round with a defined signal (`z_m`: rounds 0–49; `z_t_res`: rounds 1–49) — every client-round, not a sample.

**Raw anomalies (any single round `> 3`):**
- `z_m`: only `client_002`, every round 20–49 (30/30). No honest client's `z_m` ever exceeds `1.68` (maximum: `client_001`, round 49).
- `z_t_res`: only `client_002`, every round 20–49 (30/30, ranging `3.90`–`16.40`). No honest client's `z_t_res` ever exceeds `0.038` post-attack, or `~0` at any pre-attack round inside its own calibration window (`client_000`'s and `client_003`'s residuals are always more negative than their own historical center — i.e. consistently *above* the round median, never below it — clipping their one-sided score to exactly `0` at every single round checked; `client_004` is the literal round median, `d=0`, at all 30 post-attack rounds).

**Persistent (`P=3`) confirmed detections:** only `client_002` (rounds 20/21/22, confirmed round 22). **No honest client reaches even a single raw anomaly, let alone a 3-round persistent run** — there is no isolated-calibration-window anomaly to distinguish from a confirmed detection here, because none occurred at all.

**No honest client is confirmed before `client_002` — no honest client is ever confirmed at all.**

- Honest persistent false-positive count: **0**
- Honest persistent false-positive rate: **0/4 (0%)**

## 6. Pre-Registered Run Criteria

| # | Criterion | Result |
|---|---|---|
| 1 | `predicted_client == client_002` | **PASS** |
| 2 | No honest client confirmed before `client_002` | **PASS** |
| 3 | Onset error `<= 2` rounds | **PASS** (0) |
| 4 | Detection delay `<= 4` rounds | **PASS** (2) |
| 5 | Zero honest persistent false positives | **PASS** (0) |
| 6 | No FedTrace parameter changed | **PASS** |

## 7. Baselines

| Detector | Predicted client | Confirmation | Onset | Honest FP | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| A. Peer-cosine baseline | none | — | — | 0 | — | 0.0 | — |
| B. Magnitude-only | `client_002` | 22 | 20 | 0 | 1.0 | 1.0 | 1.0 |
| C. Temporal-residual-only (v3) | `client_002` | **22** | **20** | 0 | 1.0 | 1.0 | 1.0 |
| D. FedTrace v3 (`max(z_m, z_t_res)`) | `client_002` | 22 | 20 | 0 | 1.0 | 1.0 | 1.0 |

**A** never flags anyone (`naive_flag` is `False` for all 250 rows).

**Notably, C matches B exactly at this seed** (`onset=20`, `confirm=22`) — unlike `EXP_016`, where temporal-residual-only lagged magnitude-only by one round (`onset=21`, `confirm=23`). This is a real, seed-dependent difference in the raw `z_t_res(client_002, round 20)` value: at seed 2026 it already reaches `3.90` (just above threshold) at round 20 itself, whereas at seed 123 it only reached `1.32` at round 20 and did not cross 3 until round 21. Both are genuine, unmodified computations from each run's own trace — this document does not average or otherwise reconcile the difference, only reports it.

## 8. Signal Dominance

At every one of `client_002`'s 30 anomalous rounds, the winning component was determined exactly (both `z_m` and `z_t_res` computed and compared round by round, not sampled):

| Round range | Winner |
|---|---|
| 20 | `z_m` (6.81 vs. 3.90) |
| 21–39 | `z_t_res` (peaks 16.40 at round 21, declining to 9.08 by round 39) |
| 40–42 | `z_m` (8.71–8.59 vs. 8.02–8.51) |
| 43 | `z_t_res` (9.03 vs. 8.95, narrowly) |
| 44–49 | `z_m` (climbing to 9.82 by round 49, vs. `z_t_res` declining to 7.41) |

**`z_m` dominance count: 10/30. `z_t_res` dominance count: 20/30.**

**This is qualitatively different from `EXP_016`**, where `z_m` dominated 100% of the 30 anomalous rounds and `z_t_res` never won. At seed 2026, `z_t_res` is the larger component for the majority of the post-attack window (rounds 21–39 and 43), with `z_m` only regaining the lead at the very start (round 20) and the later tail (rounds 40–49, except 43) as its own upward trend (client_002's `update_l2_norm` climbs steadily across the run) eventually overtakes `z_t_res`'s gently declining trend. **The primary detection outcome (§4) is unaffected by this — both components already exceed 3 at every anomalous round regardless of which one is larger, so onset/confirmation timing is identical either way** — but the *mechanism* by which FedTrace v3 stays above threshold differs materially between the two seeds. This is reported as a genuine, unmodified cross-seed finding, not smoothed over: **the Gaussian attack family's dominant signal is not fixed across seeds** — magnitude and temporal-residual trade off which one is momentarily larger, even though the combiner's coverage property (§14 of prior Stage B analyses: max() picks up whichever is informative) continues to produce the correct, timely detection at both seeds.

## 9. Global Performance

| Metric | `EXP_017` (attacked) | `EXP_009` (honest) | Δ (017 − 009) |
|---|---|---|---|
| Test loss | 1.476440 | 1.446614 | **+0.029826** (worse) |
| Test Top-1 | 0.5808 | 0.5826 | **−0.18 pp** |
| Test Top-5 | 0.8807 | 0.8856 | **−0.49 pp** |
| Macro precision | 0.638091 | 0.623734 | +1.44 pp |
| Macro recall | 0.5808 | 0.5827 | −0.19 pp |
| Macro F1 | 0.562424 | 0.564105 | −0.17 pp |
| Balanced accuracy | 0.5808 | 0.5827 | −0.19 pp |
| Best validation Top-1 | 0.5818 (round 48) | 0.5808 (round 49) | **+0.10 pp** |

All differences are small (`<1.5` percentage points in every field), replicating `EXP_014`'s and `EXP_016`'s "essentially neutral" finding for this attack family at `ρ=1.0`, now confirmed at a second seed. Validation Top-1 trajectories for rounds 14–26 (inspected directly from both runs' `round_history.csv`) show no discontinuity at round 20 in either run — both continue the same steady, noisy improvement through the attack-onset window.

**Best validation Top-1 is nominally higher for the attacked run (+0.10 pp), and macro precision is nominally higher (+1.44 pp) — as with `EXP_016`'s macro-precision fluctuation, neither is interpreted as a beneficial effect of the attack.** Both are small, single-metric fluctuations (the remaining six metrics all move in the expected, unfavorable direction), consistent with ordinary cross-run GPU/training noise already documented throughout this project, not a reproducible pattern across independent measures.

## 10. Independent Validation Status

**`EXP_017` was unseen during FedTrace v3 development** — every formula and threshold in §3 was frozen before this run's trace was read, exactly as for `EXP_016`. No result in this document may modify, and did not modify, the frozen detector.

**Decision: PASS**, using only the pre-registered criteria in §6 — no reinterpretation after seeing the result.

## 11. Gaussian Family Synthesis

| | `EXP_016` (seed 123) | `EXP_017` (seed 2026) |
|---|---|---|
| Predicted client | `client_002` | `client_002` |
| Attribution correct | Yes | Yes |
| Onset estimate | 20 | 20 |
| Onset error | 0 | 0 |
| Confirmation round | 22 | 22 |
| Detection delay | 2 | 2 |
| Honest persistent FP | 0 | 0 |
| Dominant signal | `z_m` (30/30) | mixed (`z_m` 10/30, `z_t_res` 20/30) |
| Per-run decision | PASS | PASS |

Per `research_plans/FEDTRACE_STAGE_B_INDEPENDENT_VALIDATION_PLAN.md` §11: the cross-seed criterion requires **both** unseen seeds to individually pass §10's per-run criteria, with no averaging away of a failure. **Both `EXP_016` and `EXP_017` pass individually, on every criterion, with identical primary outcomes** (attribution, onset, delay, zero honest FP) despite a materially different signal-dominance mechanism underneath.

**GAUSSIAN FAMILY VALIDATED.**

This conclusion is scoped exactly as the Stage B plan requires: it states that the Gaussian update-noise attack family, at `ρ=1.0`, `client_002`, round-20 onset, is independently validated across the two pre-registered unseen seeds. **It does not generalize beyond this attack family** — the norm-preserving directional family (`EXP_018`/`EXP_019`) remains completely untested by this document, and nothing here predicts how it will perform.

## 12. Next Step

`EXP_017` is a technically valid run: attack activation, deactivation, trace schema, and honest-phase validation trajectory all match the established pattern with no divergence from `EXP_009` requiring explanation. No implementation failure occurred at any point.

Per the frozen Stage B protocol, with the Gaussian family now validated (§11) and both `EXP_016`/`EXP_017` technically valid, **the Stage B protocol permits moving to `EXP_018`, the first unseen directional-attack validation run** (seed 123, paired with `EXP_008`), to begin the second attack family's cross-seed matrix. **`EXP_018`'s config is not created by this document**, per this task's explicit restriction — that remains a separate, subsequent step.
