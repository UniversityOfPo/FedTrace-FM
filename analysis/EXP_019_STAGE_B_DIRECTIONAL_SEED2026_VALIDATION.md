# EXP_019 — Stage B Independent Validation: Directional Attack, Seed 2026

**Status:** Independent validation analysis. `EXP_019` was **not** used to design, diagnose, or tune FedTrace v3. FedTrace v3 was applied here **exactly as frozen** — nothing was tuned after inspecting the trace, regardless of outcome. This is the second and final directional-family Stage B run, and the fourth and final Stage B run overall.

---

## 1. EXP_019 ↔ EXP_009 Pairing Verification

| Field | `EXP_019` | `EXP_009` | Match |
|---|---|---|---|
| `seed` | 2026 | 2026 | ✅ |
| `partition` | dirichlet | dirichlet | ✅ |
| `dirichlet_alpha` (config) | 0.1 | 0.1 | ✅ |
| `client_sample_counts` | `{000:5233, 001:12318, 002:9171, 003:6556, 004:11722}` | identical, value-for-value | ✅ (same partition realization) |
| `model` / `total_parameters` | resnet18_cifar / 11,220,132 | same | ✅ |
| `optimizer` / `learning_rate` / `momentum` / `weight_decay` | sgd / 0.01 / (config-identical) | same | ✅ |
| `scheduler` / `mixed_precision` | (config-identical: none / true) | same | ✅ |
| `local_epochs` | 1 | 1 | ✅ |
| `communication_rounds_completed` | 50/50 | 50/50 | ✅ |
| `aggregation_strategy` | fedavg | fedavg | ✅ |
| `batch_size` | 32 | 32 | ✅ |
| `num_clients` / `clients_per_round` / `minimum_updates` | 5 / 5 / 5 | same | ✅ |

**Pairing is valid.** As an additional cross-check beyond `summary.json`: `EXP_019`'s pre-attack (`round_id < 20`) `update_l2_norm` and `self_prev_round_cosine` values were compared directly against `EXP_017`'s (the other seed-2026 attacked run) and found bit-identical row for row — e.g. `client_002` round 0 `update_l2_norm = 7.38376577122356` in both — confirming the honest pre-attack trajectory at seed 2026 is reproduced exactly regardless of attack family. **The only scientific difference is activation of the frozen directional attack.**

## 2. Attack Correctness

`summary.json`: `attack_type=norm_preserving_directional`, `malicious_client_id=client_002`, `attack_start_round=20`, `directional_target_cosine=0.0`.

**`attack_strength_rho=0.0`** is confirmed to be the inert, backward-compatible Gaussian-only metadata field — not the operative parameter for this attack (`directional_target_cosine` is), consistent with `EXP_015`'s and `EXP_018`'s own documented convention.

Full `client_round_trace.csv` (250 rows): `client_002`'s `attack_active` is `False` for `round_id < 20` and `True` for `round_id >= 20`. **No other client's `attack_active` is ever `True`, at any round** (confirmed by inspecting all 200 non-`client_002` rows).

## 3. Norm-Preservation Verification

Not inferred from the attack's name — verified from the trace's own dedicated fields, all 30 active rounds:

| Field | Min | Max | Mean |
|---|---|---|---|
| `delta_attack_norm_ratio` | `0.9999999998546052` | `1.0000000001380283` | `≈1.0` (deviations of order `1e-10`) |
| `realized_cosine_to_honest` | `-5.478639902121074e-10` | `5.182390302136504e-10` | `≈0.0` (deviations of order `1e-10`) |

Both frozen directional invariants are satisfied to floating-point precision at every active round — this is a genuinely norm-preserving, target-cosine-`0.0` attack, not merely assumed to be so.

**`update_l2_norm(client_002)` itself** shows the same nuance already documented for `EXP_018`: pre-attack (rounds 0–19) mean `6.0220`, sample stdev `0.3510` (reused from `EXP_017`'s identical seed-2026 calibration). Post-attack, the value starts close to this baseline (`5.9478` at round 20) and drifts upward across the run to `7.2763` by round 49 — the same honest-counterfactual-drift mechanism already established for this attack family (§3 of `EXP_018`'s analysis), reproduced at a second seed, again *not* a contradiction of the exact per-round norm preservation confirmed above.

## 4–5. FedTrace v3 Predicted Malicious Client / Primary Detection Result

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

## 6. Exhaustive Honest False-Positive Check

All 5 clients' `z_m` and `z_t_res` were computed and inspected at every round with a defined signal — every client-round in the complete 250-row trace.

| Client | Raw anomaly ever? | First raw-anomaly round | Max `z_m` | Max `z_t_res` | `P=3` confirmed? |
|---|---|---|---|---|---|
| `client_000` | No | — | 1.150 (round 49) | 0.0 (never above its own historical center) | No |
| `client_001` | No | — | 1.245 (round 49) | 0.041 (round 26, negligible) | No |
| `client_003` | No | — | 0.875 (round 20) | 0.0 (never above its own historical center) | No |
| `client_004` | No | — | 1.078 (round 48) | 0.0 (literal round median, `d=0`, at all 30 post-attack rounds) | No |

No honest client ever produces even a single raw anomaly (`score > 3`), let alone a `P=3` persistent run — there is no isolated-calibration-window anomaly to distinguish from a confirmed detection here, because none occurred at any point, pre- or post-attack.

- **No honest client is confirmed before `client_002` — no honest client is ever confirmed at all.**
- Honest persistent false-positive count: **0**
- Honest persistent false-positive rate: **0/4 (0%)**

## 7. Peer-Cosine Baseline

`naive_flag` is `False` for all 250 rows of the trace. **No one is flagged; there is no predicted client, no onset, no confirmation.** Honest persistent FP count: 0 (trivially, since it never flags anyone). Same failure mode already documented against every prior attack/seed in this series.

## 8. Magnitude-Only Baseline (`z_m`)

| | Value |
|---|---|
| Predicted client | `client_002` |
| First anomalous round | **44** |
| Confirmation round | **46** |
| Estimated onset | **44** |
| Onset error | 24 |
| Detection delay | 26 |
| Honest persistent FP | 0 |
| Precision / Recall / F1 | 1.0 / 1.0 / 1.0 |

Not assumed to fail in advance — checked against the actual trace. `z_m(client_002)` stays below `3` for the first 24 post-attack rounds (rising gradually as the honest-counterfactual drift of §3 accumulates: `0.21` at round 20 to `2.88` at round 43), crosses `3` at round 44 (`z=3.13`) and remains above `3` for the rest of the run (6 consecutive rounds, 44–49) — confirming at round 46. **Attribution is still eventually correct**, but detection is very late. This is a materially different (earlier, more sustained) magnitude-only result than `EXP_018`'s (which confirmed only at round 49, the very last round, on a bare 3-round streak) — a genuine seed-to-seed difference in how quickly the honest-counterfactual drift accumulates enough magnitude signal, reported exactly as found.

## 9. Temporal-Residual-Only Baseline (`z_t_res`)

| | Value |
|---|---|
| Predicted client | `client_002` |
| First anomalous round | **20** |
| Confirmation round | **22** |
| Estimated onset | **20** |
| Onset error | 0 |
| Detection delay | 2 |
| Honest persistent FP | 0 |
| Precision / Recall / F1 | 1.0 / 1.0 / 1.0 |

`z_t_res(client_002)` is `40.83` at round 20 itself and remains far above `3` for the entire post-attack window (minimum `21.82`, round 49) — an even sharper separation than at seed 123 (`EXP_018`: range `5.62`–`9.19`), driven by this seed's especially tight `client_002` historical calibration scale (`std_d = 0.005142`, identical to `EXP_017`'s, reused since pre-attack rounds are bit-identical).

## 10. Full FedTrace v3 (`max(z_m, z_t_res)`)

| | Value |
|---|---|
| Predicted client | `client_002` |
| First anomalous round | 20 |
| Confirmation round | 22 |
| Onset estimate | 20 |
| Onset error | **0** |
| Detection delay | **2** |
| Honest persistent FP | 0 |
| Precision / Recall / F1 | 1.0 / 1.0 / 1.0 |

Identical to temporal-residual-only in every respect (§11).

## 11. Signal Dominance

For every one of `client_002`'s 30 post-attack anomalous rounds, `z_m` and `z_t_res` were compared directly:

| Comparison | Count | Percentage |
|---|---|---|
| `z_t_res > z_m` | **30** | **100%** |
| `z_m > z_t_res` | 0 | 0% |
| Tied | 0 | 0% |

**Values at rounds 20/21/22** (both signals, exactly as computed from this run's own trace):

| Round | `z_m` | `z_t_res` |
|---|---|---|
| 20 | 0.211 | **40.827** |
| 21 | 0.372 | **40.594** |
| 22 | 0.456 | **38.078** |

`z_t_res` produces the earliest detection outright; `z_m` never approaches threshold until round 44 (§8), by which point `z_t_res` has already been confirmed for 22 rounds. **`z_t_res` dominates every single anomalous round; `z_m` never wins, exactly matching `EXP_018`'s 30/30 pattern.** Per this task's explicit instruction, this match is not required or assumed in advance — it is what the data independently shows for this second directional-family seed, and it is reported as a finding (a second, consistent data point), not as a property `z_t_res` was guaranteed to have.

## 12. Pre-Registered Per-Run Criteria

| # | Criterion | Result |
|---|---|---|
| 1 | `predicted_client == client_002` | **PASS** |
| 2 | No honest client confirmed before `client_002` | **PASS** |
| 3 | Onset error `<= 2` rounds | **PASS** (0) |
| 4 | Detection delay `<= 4` rounds | **PASS** (2) |
| 5 | Zero honest persistent false positives | **PASS** (0) |
| 6 | No FedTrace parameter changed | **PASS** |

No criterion was redefined or softened.

## 13. Global Performance vs. `EXP_009`

| Metric | `EXP_019` (attacked) | `EXP_009` (honest) | Δ (019 − 009) |
|---|---|---|---|
| Test loss | 1.713880 | 1.446614 | **+0.267266** |
| Test Top-1 | 0.5456 | 0.5826 | **−3.70 pp** |
| Test Top-5 | 0.8387 | 0.8856 | **−4.69 pp** |
| Macro precision | 0.577611 | 0.623734 | **−4.61 pp** |
| Macro recall | 0.5455 | 0.5827 | **−3.72 pp** |
| Macro F1 | 0.509870 | 0.564105 | **−5.42 pp** |
| Balanced accuracy | 0.5455 | 0.5827 | **−3.72 pp** |
| Best validation Top-1 | 0.5444 (round 49) | 0.5808 (round 49) | **−3.64 pp** |

**No metric improves under the attack** — a real, substantial, consistently unfavorable degradation, matching `EXP_018`'s finding in kind and magnitude, and again independently reproducing (at a second seed) the "real, sustained, boundary-localized global-model degradation" already documented for this attack family at seed 42 (`analysis/EXP_015_Directional_Attack_Traceability_Analysis.md`). Validation Top-1 trajectories for rounds 14–26 (`round_history.csv`, both runs) confirm the divergence is visible starting right at attack onset: pre-attack (rounds 14–19) the two runs track closely (`EXP_019`: `0.4472` at round 19; `EXP_009`: `0.4482`); at round 20 a modest gap opens (`0.4380` vs. `0.4536`); by round 26 the gap has grown to roughly 4 percentage points (`0.4592` vs. `0.4862`, reading `round_history.csv`'s own round-26 value). No accuracy increase occurs anywhere in this comparison, so no such increase requires the "not an attack benefit" caveat here.

## 14. EXP_019 Independent Validation Verdict

**PASS.** All six pre-registered criteria (§12) are satisfied. There is no near-pass category, and none was needed.

## 15. Directional Family Cross-Seed Synthesis

| | `EXP_018` (seed 123) | `EXP_019` (seed 2026) |
|---|---|---|
| Predicted client | `client_002` | `client_002` |
| Attribution correct | Yes | Yes |
| Onset estimate / error | 20 / 0 | 20 / 0 |
| Confirmation round / delay | 22 / 2 | 22 / 2 |
| Honest persistent FP | 0 | 0 |
| Magnitude-only confirms at | round 49 (barely, final round) | round 46 |
| Dominant signal | `z_t_res` (30/30) | `z_t_res` (30/30) |
| Per-run verdict | PASS | PASS |

Per `research_plans/FEDTRACE_STAGE_B_INDEPENDENT_VALIDATION_PLAN.md` §11: **both unseen seeds individually PASS every criterion, with no averaging away of a failure (none occurred).**

**DIRECTIONAL FAMILY VALIDATED.**

## 16. Current Stage B Status (All Four Runs)

| Run | Attack family | Seed | Verdict |
|---|---|---|---|
| `EXP_016` | Gaussian | 123 | PASS |
| `EXP_017` | Gaussian | 2026 | PASS |
| `EXP_018` | Directional | 123 | PASS |
| `EXP_019` | Directional | 2026 | PASS |

**Gaussian family: VALIDATED** (`analysis/EXP_017_STAGE_B_GAUSSIAN_SEED2026_VALIDATION.md` §11). **Directional family: VALIDATED** (§15, above). **All four Stage B independent-validation runs have now completed, each individually PASSing its six pre-registered criteria, and both pre-registered attack families are validated.** Per `research_plans/FEDTRACE_STAGE_B_INDEPENDENT_VALIDATION_PLAN.md` §18's decision categories, this combination of outcomes is consistent with an eventual Stage B **PASS** determination — **the frozen protocol now permits proceeding to the final Stage B synthesis document**, which is explicitly not produced in this task (§19 of this task's own restrictions) and remains a separate, subsequent step.

## 17. Scientific Interpretation

The flagship question — does frozen FedTrace v3 provide traceability across both a magnitude-revealing Gaussian attack and a norm-preserving directional attack, under unseen strong-Non-IID seeds — is answered by the combined four-run evidence, with the distinction this task requires drawn explicitly:

**Cross-attack coverage (established):** a single, frozen `FedTraceV3Score = max(z_m, z_t_res)`, with no attack-specific tuning and no knowledge of which family a given run belongs to, correctly attributed `client_002` with `onset error = 0` and `delay = 2` on all four Stage B runs. It did this by relying on `z_m` for the Gaussian family (`EXP_016`: `z_m` 30/30 dominant; `EXP_017`: mixed, `z_m` 10/30 and `z_t_res` 20/30, still correct) and on `z_t_res` for the directional family (`EXP_018`/`EXP_019`: `z_t_res` 30/30 dominant both times) — automatically, via the `max()` combiner, without ever being told in advance which signal would matter for a given run.

**Not established, and not claimed, as superiority over the attack-specific single detector:** in every one of the four runs, `FedTraceV3Score`'s onset/delay is numerically identical to whichever single component dominates that run (magnitude-only on the Gaussian runs where it wins fastest; temporal-residual-only on the directional runs). The combiner has never, across all four Stage B runs, produced a detection outcome better than its best individual component for that run — this is the same "coverage, not fusion" property already established throughout the Stage A/development series (`analysis/FEDTRACE_V3_RETROSPECTIVE_ANALYSIS.md` §10, `EXP_018`'s analysis §20) and now confirmed, not merely carried forward by assumption, across all four independent validation seeds. **The demonstrated value is a single, unified, frozen detector that does not need to know the attack type in advance and performs, on each of the two tested families, as well as the best single signal for that family — not a claim that combining signals outperforms knowing the right one ahead of time.**

## 18. Validation Discipline

**`EXP_019` is independent validation data. It was not used to develop FedTrace v3.** No result in this document — including the eventual-but-very-late magnitude-only confirmation in §8, and the extreme (`>40`) `z_t_res` values in §9/§11 — modified, or would have modified, any formula, calibration, threshold, persistence rule, signal definition, or attribution rule. `src/analysis/fedtrace_detector.py` was not touched in the course of producing this document. No weakness requiring a repair was found in this run (§14: clean PASS on every criterion); had one been found, per this task's explicit instruction, it would have been recorded here and not repaired.

---

## Final Report

1. **File created:** `analysis/EXP_019_STAGE_B_DIRECTIONAL_SEED2026_VALIDATION.md`.
2. **Files modified:** none — all source artifacts were read-only.
3. **EXP_019 ↔ EXP_009 pairing verification:** valid — seed 2026, identical partition realization (`client_sample_counts` matches value-for-value), identical training/federated settings; only the directional attack differs (§1).
4. **Attack correctness:** confirmed — `norm_preserving_directional`, `client_002`, round 20, `target_cosine=0.0`; `attack_active` correctly `False` rounds 0–19 and for every non-`client_002` client throughout; `attack_strength_rho=0.0` confirmed inert/backward-compatible, not evidence of no attack (§2).
5. **Norm-preservation verification:** confirmed genuinely norm-preserving — `delta_attack_norm_ratio ≈ 1.0` and `realized_cosine_to_honest ≈ 0.0` to `~1e-10` precision at every active round (§3).
6. **FedTrace v3 predicted malicious client:** `client_002`.
7. **First anomaly / confirmation / onset estimate:** round 20 / round 22 / round 20.
8. **Onset error:** 0.
9. **Detection delay:** 2.
10. **Honest raw-anomaly summary:** zero honest raw anomalies for any client, at any round, pre- or post-attack (max `z_m`=1.245, max `z_t_res`=0.041, both far below 3) — §6.
11. **Honest persistent FP count:** 0 (rate 0/4, 0%).
12. **Peer baseline result:** flags no one; no predicted client, onset, or confirmation.
13. **Magnitude-only result:** `client_002`, first anomaly round 44, confirm round 46, onset 44 (onset error 24, delay 26) — eventually correct but very late.
14. **Temporal-residual-only result:** `client_002`, first anomaly/confirm/onset = 20/22/20 (onset error 0, delay 2).
15. **Full FedTrace v3 result:** identical to temporal-residual-only — onset 20, confirm 22, delay 2, 0 honest FP, precision/recall/F1 = 1.0/1.0/1.0.
16. **Signal-dominance counts:** `z_t_res` wins 30/30 (100%); `z_m` wins 0/30; no ties.
17. **`z_m`/`z_t_res` at rounds 20/21/22:** `z_m` = 0.211 / 0.372 / 0.456; `z_t_res` = 40.827 / 40.594 / 38.078.
18. **EXP_019 vs EXP_009 performance differences:** substantial, consistent degradation — Top-1 −3.70pp (test), −3.64pp (best validation); every metric checked moves unfavorably; visible divergence beginning at attack onset in the validation trajectory.
19. **Six pre-registered criteria:** all PASS.
20. **Overall EXP_019: PASS.**
21. **EXP_018 + EXP_019 directional-family verdict: DIRECTIONAL FAMILY VALIDATED** — both seeds individually pass every criterion, no failure averaged away.
22. **Current EXP_016–EXP_019 Stage B status:** all four runs complete, all four PASS; Gaussian family validated; directional family validated.
23. **Whether final Stage B synthesis is permitted next:** Yes — the frozen protocol now permits proceeding to the final Stage B synthesis document; it is not produced in this task, per its explicit restriction.
24. **Confirmation FedTrace v3 unchanged:** confirmed — `src/analysis/fedtrace_detector.py` not touched.
25. **Confirmation no experiment/config/code/test/database changed:** confirmed — only this one analysis document was created; no `EXP_020` config or any other file was created.
26. **Confirmation whether any commands were executed:** none — this task used only `Read` and `Grep` (read-only) plus manual arithmetic; no shell/PowerShell/Bash command was run.
