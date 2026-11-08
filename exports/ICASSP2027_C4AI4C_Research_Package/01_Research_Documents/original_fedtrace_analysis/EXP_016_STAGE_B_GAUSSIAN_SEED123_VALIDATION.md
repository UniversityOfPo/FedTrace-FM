# EXP_016 — Stage B Independent Validation: Gaussian Attack, Seed 123

**Status:** Independent validation analysis. `EXP_016` was **not** used to design, diagnose, or tune FedTrace v3 — it is the first genuinely unseen run this frozen method has ever been scored against. FedTrace v3 was applied here **exactly as frozen** (`research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md` §16, `src/analysis/fedtrace_detector.py`'s v3 section); nothing was tuned after inspecting the trace, regardless of what the result turned out to be.

---

## 1. Verify Pairing

Confirmed directly from `EXP_016`'s and `EXP_008`'s own `summary.json`:

| Field | `EXP_016` | `EXP_008` | Match |
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

**The Gaussian attack is confirmed to be the only scientific difference** — `EXP_016`'s `summary.json` has an `"attack"` block entirely absent from `EXP_008`'s; every other tracked setting matches exactly, replicating the same single-variable-change confirmation already established for `EXP_014`/`EXP_007` at seed 42.

## 2. Verify Attack Correctness

From `EXP_016`'s `summary.json`: `attack_type=gaussian_update_noise`, `malicious_client_id=client_002`, `attack_start_round=20`, `attack_strength_rho=1.0` — matches the frozen Stage B attack parameters exactly.

From the full `client_round_trace.csv` (250 rows, read in full): `client_002`'s `attack_active` is `False` for every row with `round_id < 20` and `True` for every row with `round_id >= 20`. **No other client's `attack_active` is ever `True`, at any round** (confirmed by inspecting all 200 non-`client_002` rows).

**`update_l2_norm` before/after onset (`client_002`):** pre-attack (rounds 0–19) mean `5.5097`, sample stdev `0.3139` (20 observations). Post-attack (rounds 20–49): jumps immediately to `7.6277` at round 20 and climbs to `8.6048` by round 49 — every one of the 30 post-attack values exceeds the pre-attack mean by more than `6` standard deviations (`z_m` ranges `6.75`–`9.86` across the whole post-attack window, computed in §7 below). This is the same immediate, sustained magnitude signature already established for `EXP_014` at seed 42, now reproduced at seed 123.

## 3. Apply Frozen FedTrace v3

Applied exactly as frozen — `z_m` (per-client longitudinal, `T_ref=20`), `c(r)`/`d(k,r)`/`D_k`/`z_t_res` (per §1 of `research_plans/FEDTRACE_STAGE_B_INDEPENDENT_VALIDATION_PLAN.md`), `FedTraceV3Score = max(z_m, z_t_res)`, threshold `> 3`, `P=3`. No attack ground truth was used at any point during scoring — `attack_start_round`/`malicious_client_id` are read only in this document's post-hoc evaluation tables, never inside the calibration or persistence computation itself.

## 4. Primary Stage B Result

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

## 5. Honest False Positives

All 5 clients' `z_m` and `z_t_res` were computed and inspected at **every one of the 49 rounds with a defined signal** (rounds 1–49 for `z_t_res`, rounds 0–49 for `z_m`) — not a representative sample.

- **`z_m`:** no honest client ever exceeds `z=1.85` (maximum: `client_000`, round 47) at any round, pre- or post-attack.
- **`z_t_res`:** no honest client ever exceeds `z=2.53` (maximum: `client_000`, round 1 — inside its own calibration window, not post-attack) at any round. Post-attack, the maximum honest `z_t_res` observed anywhere is `0.0` (`client_000`, `client_001`, and `client_004` all sit at or below their own historical center for the entire post-attack window; `client_003` is the round median itself, `d=0`, at all 30 post-attack rounds).
- **`FedTraceV3Score`:** consequently, no honest client ever exceeds `3` at any round, pre- or post-attack.

**No honest client is confirmed before `client_002` — no honest client is ever confirmed at all.**

- Honest persistent false-positive count: **0**
- Honest persistent false-positive rate: **0/4 (0%)**

## 6. Per-Run Pre-Registered Pass Criteria

| # | Criterion | Result |
|---|---|---|
| 1 | `predicted_client == client_002` | **PASS** |
| 2 | No honest client confirmed before `client_002` | **PASS** (no honest client confirmed at all) |
| 3 | Onset error `<= 2` rounds | **PASS** (0) |
| 4 | Detection delay `<= 4` rounds | **PASS** (2) |
| 5 | Zero honest persistent false positives | **PASS** (0) |
| 6 | No FedTrace parameter changed | **PASS** (all values used exactly as frozen in §1 of the Stage B plan) |

## 7. Baseline Detectors

| Detector | Predicted client | Confirmation round | Onset estimate | Honest FP | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| A. Peer-cosine baseline | none | — | — | 0 | — | 0.0 | — |
| B. Magnitude-only | `client_002` | 22 | 20 | 0 | 1.0 | 1.0 | 1.0 |
| C. Temporal-residual-only (v3) | `client_002` | **23** | **21** | 0 | 1.0 | 1.0 | 1.0 |
| D. FedTrace v3 (`max(z_m, z_t_res)`) | `client_002` | 22 | 20 | 0 | 1.0 | 1.0 | 1.0 |

**A** never flags anyone — `naive_flag` is `False` for every one of the 250 rows in the trace (confirmed by direct inspection), replicating the peer-cosine baseline's already-established failure mode against this attack family.

**C is evaluable and correct, but slower and less precisely localized than B or D on this run**, and reveals a genuine, real (not fabricated) geometric difference from the `EXP_015` directional attack: `client_002`'s `z_t_res` only exceeds 3 at rounds 21–25 (peaking at `3.47`, round 22), then drops back under 3 for the remainder of the run. This is expected: `gaussian_update_noise` **adds** independent noise to the honest delta rather than **replacing** its direction outright, so `self_prev_round_cosine` degrades only partially, not to the near-zero collapse the norm-preserving directional attack produced. Temporal-residual-only still crosses threshold and confirms (rounds 21/22/23), but one round later than magnitude-only (onset 21 vs. 20; delay 3 vs. 2).

**D (FedTrace v3) matches B (magnitude-only) exactly** on every metric — `z_m` dominates the combined score at every single anomalous round (§8).

## 8. Signal Dominance

At every one of `client_002`'s 30 anomalous (`FedTraceV3Score > 3`) rounds, `z_m` is the maximum component — `z_m` ranges `6.75`–`9.86` across rounds 20–49, while `z_t_res` peaks at `3.47` (round 22) and is otherwise below 3 for most of the window. **`z_m` dominance count: 30/30 (100%). `z_t_res` dominance count: 0/30 (0%).**

**This independent seed confirms the development-run finding** (`analysis/EXP_014_Gaussian_Attack_Traceability_Analysis.md`, `analysis/FEDTRACE_V3_RETROSPECTIVE_ANALYSIS.md` §13) **that magnitude is the dominant signal for the Gaussian attack family.** Per `research_plans/FEDTRACE_STAGE_B_INDEPENDENT_VALIDATION_PLAN.md` §8, this is reported as a single additional confirming data point for the Gaussian family specifically — **not treated as proof of universal FedTrace v3 behavior**, and not evidence about the directional family, which this run does not test.

## 9. Global Performance

| Metric | `EXP_016` (attacked) | `EXP_008` (honest) | Δ (016 − 008) |
|---|---|---|---|
| Test loss | 1.498936 | 1.465440 | **+0.033496** (worse) |
| Test Top-1 | 0.5650 | 0.5683 | **−0.33 pp** |
| Test Top-5 | 0.8734 | 0.8779 | **−0.45 pp** |
| Macro precision | 0.645422 | 0.633761 | +1.17 pp |
| Macro recall | 0.5651 | 0.5683 | −0.32 pp |
| Macro F1 | 0.545689 | 0.548573 | −0.29 pp |
| Balanced accuracy | 0.5651 | 0.5683 | −0.32 pp |
| Best validation Top-1 | 0.5748 (round 48) | 0.5826 (round 47) | −0.78 pp |

All differences are small (magnitude `<1.2` percentage points in every field), matching `EXP_014`'s own "essentially neutral" finding for this attack family at `ρ=1.0` — now replicated at a second seed. Validation loss/Top-1 trajectories for rounds 14–26 (inspected directly from both runs' `round_history.csv`) show no discontinuity or divergence at round 20; both runs continue the same steady, noisy improvement through the attack-onset window. **Macro precision is nominally higher under the attacked run** — this is not claimed as a benefit of the attack: it is a single, small, plausibly noise-level fluctuation in one derived metric, not corroborated by any of the other seven metrics (all of which move in the expected direction, however slightly), and this document does not interpret it as a real effect.

## 10. Independent Validation Discipline

**`EXP_016` was not used to design FedTrace v3.** Every formula, threshold, and rule applied in §3 was frozen in `research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md` before this run's trace was ever read. **This result is therefore genuine Stage B independent-validation evidence** — the first FedTrace v3 has ever received. Per the Stage B plan's own §13 protocol: regardless of the outcome recorded above, **FedTrace v3 was not modified in the course of producing this document**, and would not have been modified even had any §6 criterion failed.

## 11. EXP_016 Decision

**PASS.** All six pre-registered per-run criteria (§6) are satisfied, with no reinterpretation after seeing the result — the criteria were fixed in `research_plans/FEDTRACE_STAGE_B_INDEPENDENT_VALIDATION_PLAN.md` §10 before this run's trace was ever inspected.

## 12. Next Step

`EXP_016` is a technically valid run: the attack activated correctly, deactivated for every other client, the trace schema is complete and consistent with `EXP_014`'s established pattern, and the honest-phase validation trajectory tracks `EXP_008` closely with no unexplained divergence. **No implementation failure occurred** — there is nothing here to distinguish from a detector failure, because no failure occurred at all (§11).

Per the frozen Stage B protocol (`research_plans/FEDTRACE_STAGE_B_INDEPENDENT_VALIDATION_PLAN.md` §16), the next step in sequence is the **implementation-correctness checkpoint** for `EXP_016`, now satisfied (attack correctness §2, trace integrity, honest-phase fidelity — all confirmed above), followed by `EXP_017` (Gaussian, seed 2026) to complete the Gaussian-family cross-seed matrix. **Proceeding to `EXP_017` is permitted by the frozen Stage B protocol** — one seed's PASS is necessary but not sufficient for a family-level validation claim (`research_plans/FEDTRACE_STAGE_B_INDEPENDENT_VALIDATION_PLAN.md` §11 requires both seeds within a family); it does not, by itself, authorize any conclusion about the Gaussian family as a whole, and it says nothing about the directional family (`EXP_018`/`EXP_019`), which remains entirely untested. **`EXP_017`'s config is not created by this document** — per this task's explicit restriction, that remains a separate, subsequent step.
