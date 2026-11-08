# EXP_027 — Onset Generalization: Directional Attack, Seed 123, `client_002`, Onset Round 30

**Status:** Analysis-only document. `src/analysis/fedtrace_detector.py` (frozen FedTrace v3), `src/attacks/*` (attack implementation), every existing config, test, README, and Research Database file were read but **not modified**. No new experiment was run. FedTrace v3's formula, threshold (`>3`), persistence (`P=3`), `T_ref=20`, and calibration rules were applied exactly as already frozen — nothing was tuned after seeing `EXP_027`'s result. Produced by directly reading `EXP_027`'s own result artifacts and applying the unmodified `src/analysis/fedtrace_detector.py` module (imported read-only; no repository file was written to by this process), the same procedure used for `EXP_016`–`EXP_019` and `EXP_026`.

---

## 1. Pairing Verification

| Field | `EXP_027` | `EXP_018` | Match |
|---|---|---|---|
| `seed` | 123 | 123 | ✅ |
| `partition` / `dirichlet_alpha` | dirichlet / 0.1 | dirichlet / 0.1 | ✅ |
| `num_clients` / `clients_per_round` | 5 / 5 | 5 / 5 | ✅ |
| `communication_rounds_completed` | 50/50 | 50/50 | ✅ |
| `model` / `optimizer` / `learning_rate` / `batch_size` | resnet18_cifar / sgd / 0.01 / 32 | same | ✅ |
| `aggregation_strategy` | fedavg | fedavg | ✅ |
| `attack_type` | norm_preserving_directional | norm_preserving_directional | ✅ |
| `malicious_client_id` | `client_002` | `client_002` | ✅ (unchanged, deliberately) |
| `directional_target_cosine` | 0.0 | 0.0 | ✅ |
| `attack_start_round` | **30** | 20 | **the one scientific change** |
| `client_sample_counts` | identical, value-for-value | — | ✅ (same partition realization, seed 123) |

**`attack_start_round` is confirmed to be the only scientific difference between `EXP_027` and `EXP_018`**, directly from `summary.json` and by direct comparison of `configs/cifar100_fedavg_dirichlet_a01_direction_attack_client002_round30_seed123.yaml` against `configs/cifar100_fedavg_dirichlet_a01_direction_attack_client002_round20_seed123.yaml`.

**Trace-level pairing check, done correctly for this specific onset shift:** a naive round-by-round `update_l2_norm` comparison across the *entire* pre-onset window (`round < 30`) between `EXP_027` and `EXP_018` finds 46 of 150 checked cells differing — this is **expected, not an error**, and is explained precisely rather than dismissed: `EXP_018`'s own attack is already active from round 20 onward, so `EXP_018`'s global-model trajectory has already diverged from the honest counterfactual by round 21; `EXP_027`, whose attack does not begin until round 30, has no such divergence during rounds 20–29. **Restricting the comparison to the genuinely honest-in-both-runs overlap window (`round < 20`) finds 100/100 cells bit-identical, for all 5 clients** — this is the correct pairing check, and it passes completely. A finer per-round breakdown of rounds 20–29 confirms the mechanism exactly: all four honest clients (`client_000/001/003/004`) still match at round 20 itself (their round-20 local training starts from the round-19 global model, identical in both runs) but diverge from round 21 onward (once the round-20 *aggregate* — poisoned in `EXP_018`, clean in `EXP_027` — becomes each client's new starting point); `client_002` itself matches at **no** round in `[20,29]` (its own round-20 submission is already replaced by the attack transform in `EXP_018`, honest in `EXP_027`). This is a positive, mechanistic confirmation that the two runs' divergence begins exactly where each run's own attack begins, not a pairing failure.

**Diagnostics integrity:** `EXP_027`'s `client_round_trace.csv` contains exactly **250 rows** (50 rounds × 5 clients), 50 unique `round_id` values, 5 unique `client_id` values — matches the expected shape exactly.

## 2. Attack Correctness

`client_002`'s `attack_active` is `True` for exactly its 20 rows with `round_id ∈ [30,49]`, and `False` for all 30 of its rows with `round_id < 30`. **No other client's `attack_active` is ever `True`, at any round.**

**Norm preservation** (`delta_attack_norm_ratio`, 20 active rows): min `0.999999999820`, max `1.000000000219`, mean `1.000000000034` — **maximum absolute error from `1.0`: `2.19 × 10⁻¹⁰`**.

**Directional targeting** (`realized_cosine_to_honest`, 20 active rows): min `-0.000000000479`, max `0.000000000532`, mean `0.000000000059` — **maximum absolute error from `0.0`: `5.32 × 10⁻¹⁰`**.

**The attack is confirmed genuinely norm-preserving and correctly targeted, at the same floating-point precision already established for `EXP_018`, `EXP_019`, `EXP_024`, `EXP_025`, and `EXP_026`.** Moving the onset from round 20 to round 30 did not alter the attack transform's own fidelity.

## 3. FedTrace v3 Applied Exactly

Applied via `src/analysis/fedtrace_detector.py::evaluate_all_detectors_v3`/`evaluate_detector_v3`, unmodified.

| Client | First raw anomaly | Confirmation round | Onset estimate | Max `z_m` | Max `z_t_res` | Persistent detection? |
|---|---|---|---|---|---|---|
| `client_000` | 0 | — | — | 3.8943 (round 0) | 2.5303 (round 1) | **No** |
| `client_001` | 0 | — | — | 3.2558 (round 0) | 0.3234 (round 1) | **No** |
| **`client_002`** | 0 | **32** | **30** | 3.9303 (round 0) | 7.6271 (round 30) | **Yes** |
| `client_003` | 0 | — | — | 3.6073 (round 0) | 2.1534 (round 21) | **No** |
| `client_004` | 0 | — | — | 3.1892 (round 0) | 1.0338 (round 27) | **No** |

**Answers to the required questions:**

- **(A) Predicted malicious client:** `client_002`.
- **(B) True malicious client:** `client_002`.
- **(C) First anomaly round:** the raw, project-wide `first_anomalous_round` field reports **round 0** for every client, including `client_002` — this is the pre-existing round-0 structural calibration artifact (§6), not attack-related. **The first anomaly belonging to, and leading into, the confirmed persistent run is round 30** — exactly the true onset.
- **(D) Confirmation round:** **32**.
- **(E) Estimated attack onset round:** **30**.
- **(F) True attack onset round:** **30**.
- **(G) Onset error:** **0**.
- **(H) Detection delay:** **2** (confirmation round 32 − true onset 30).
- **(I) Is the delay the theoretical minimum under `P=3`?** **Yes** — three consecutive anomalous rounds starting at the true onset (30, 31, 32) confirm at `onset + (P−1) = 32`, which is exactly what occurred; no `P=3` rule could confirm faster.
- **(J) Persistent honest-client false positives:** **0** — `client_000`/`client_001`/`client_003`/`client_004` are never confirmed (`confirmation_round = None` for all four).
- **(K) Dominant signal:** `z_t_res`, in **20/20 (100%)** of `client_002`'s post-attack (`round ≥ 30`) anomalous rounds — `z_m` never overtakes it in this window (§4).
- **(L) See §4.**
- **(M) Does any detector behavior occur before true onset round 30?** Only the two pre-existing, cross-run-consistent artifacts documented in §6 (round 0's structural `z_m` crossing for all five clients, and `client_002`'s own round-1 isolated `z_t_res = 3.2888` crossing) — **nothing else, for any client, at any round in `[2, 29]`** (directly verified, §6). Neither forms, or contributes to, a persistent confirmed run.
- **(N) See §6.**

## 4. Malicious-Client (`client_002`) Signal Analysis, Rounds 27–36

| Round | `z_m` | `z_t_res` | Combined score | Dominant |
|---|---|---|---|---|
| 27 (pre-attack) | 0.0537 | 0.0 | 0.0537 | `m` |
| 28 (pre-attack) | 0.0299 | 0.0 | 0.0299 | `m` |
| 29 (pre-attack) | 0.2153 | 0.0 | 0.2153 | `m` |
| **30 (onset)** | 0.1662 | **7.6271** | 7.6271 | `t_res` |
| 31 | 0.5691 | 7.3534 | 7.3534 | `t_res` |
| 32 (confirm) | 1.0871 | 7.2360 | 7.2360 | `t_res` |
| 33 | 1.3168 | 7.2090 | 7.2090 | `t_res` |
| 34 | 1.6426 | 6.8257 | 6.8257 | `t_res` |
| 35 | 1.4409 | 6.9745 | 6.9745 | `t_res` |
| 36 | 1.6559 | 6.7868 | 6.7868 | `t_res` |

**`z_t_res` drives detection, decisively and immediately at the new onset**, jumping from `0.0` (rounds 27–29, pre-attack, `t_res` clipped to zero — no evidence of unusually *high* self-consistency, per the frozen one-sided convention) to `7.63` at round 30 itself. Across the full 20-round post-attack window (`round ≥ 30`), `client_002`'s `z_t_res` ranges `5.54`–`7.63`, never dropping below threshold even once.

**`z_m` does eventually become informative, but very late and only marginally** — it climbs steadily across the post-attack window (from `0.17` at round 30 to `3.33` at round 49, the run's final round), driven by the same accumulated honest-counterfactual global-model drift mechanism already documented for `client_002` in `EXP_018`. **Magnitude-only alone does confirm `client_002`** in this run — first crossing threshold at round 47 (`z=3.03`), confirming at round 49 (`onset_est=47`, `delay=19`) — a striking numeric echo of `EXP_018`'s own magnitude-only result (`onset_est=47`, `confirm=49`), even though the true onset moved by 10 rounds. This coincidence (the identical final confirmation round, 49, in both runs) is reported as observed and is **not** interpreted as evidence of any fixed relationship between onset round and magnitude-only confirmation timing — one data point in each condition is not sufficient to establish such a pattern, and this document does not claim one.

**Dominance count:** `z_t_res > z_m` in **20/20 (100%)** of `client_002`'s post-attack anomalous rounds — identical in kind to `EXP_018`'s own 30/30 (100%) `z_t_res` dominance, and to `EXP_026`'s 30/30 (100%).

## 5. Full FedTrace v3 vs. Component-Only Detectors

| Detector | Predicted | Correct? | Onset err | Delay | Honest FP |
|---|---|---|---|---|---|
| `A_magnitude_only` | `client_002` | Yes | 17 | 19 | 0 |
| `E_temporal_residual_only` | `client_002` | Yes | **0** | **2** | 0 |
| `H_fedtrace_v3` (full) | `client_002` | Yes | **0** | **2** | 0 |

**Full FedTrace v3's result is identical to `z_t_res`-only** — the combiner neither helps nor hurts here, because `z_t_res` dominates every single anomalous round in the confirmed run (§4). This is the same "coverage, not fusion" pattern already established in every prior Stage B and generalization run (`analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md` §8, `analysis/EXP_026_ATTACKER_IDENTITY_GENERALIZATION_ANALYSIS.md` §4). Magnitude-only alone would have satisfied attribution correctness but would have badly failed the pre-registered onset-error (`≤2`) and delay (`≤4`) criteria on its own — exactly the scenario the multi-signal design exists to cover automatically, without knowing in advance which signal would be informative.

## 6. Honest-Client Safety and the Round-0/Round-1 Artifacts

All 5 clients' `z_m` and `z_t_res` were computed and inspected at every round with a defined signal — the complete 250-row trace.

**Raw anomaly counts (score `> 3`):**

| Client | Raw anomalous rounds | Persistent (`P=3`) runs |
|---|---|---|
| `client_000` | 1 — round 0 (`z_m = 3.8943`) | **0** |
| `client_001` | 1 — round 0 (`z_m = 3.2558`) | **0** |
| `client_003` | 1 — round 0 (`z_m = 3.6073`) | **0** |
| `client_004` | 1 — round 0 (`z_m = 3.1892`) | **0** |
| `client_002` (attacker) | round 0 (`z_m = 3.9303`), round 1 (`z_t_res = 3.2888`), then rounds 30–49 (all `z_t_res`, the genuine detection) | **1** (the correct, attack-driven run) |

**No honest client (`000`/`001`/`003`/`004`) produces any raw anomaly anywhere outside round 0.** `client_002`'s own round-0/round-1 crossings are pre-attack and isolated (not contiguous with each other or with round 30, 28 rounds later) — they do not contribute to, or interfere with, its later genuine detection.

**These are not new occurrences.** Both values are **bit-identical** to the round-0/round-1 values already reported for `client_002` in `analysis/EXP_026_ATTACKER_IDENTITY_GENERALIZATION_ANALYSIS.md` §5 (where `client_002` was honest throughout) and consistent with the round-0 pattern reported there for the other four clients — a direct consequence of §1's confirmed bit-identical rounds-`[0,20)` trajectory shared by `EXP_018`, `EXP_026`, and `EXP_027` alike (all three runs are honest for every client through at least round 19). **This is reassuring, not concerning: it demonstrates the artifact is a deterministic, fully reproducible property of this seed/partition's own round-0/round-1 data, not noise, and not something newly introduced by moving the attack onset.**

**As instructed, no detector logic was changed because of this artifact.** It remains exactly what it was diagnosed as in `EXP_026`'s analysis: a self-inclusive `n=20` calibration-window boundary effect (round 0 is the largest `update_l2_norm` value in every client's own `[0,20)` window, a structural property of round 0 being each client's very first local-training step from the shared initialization), bounded in magnitude (`≤(n−1)/√n ≈ 4.25` for `n=20`), never persistent, and without effect on any attribution, onset, or false-positive conclusion in this document. **It does not affect the round-30 onset conclusion**: the artifact is confined to rounds 0–1, twenty-eight to twenty-nine rounds before the true onset, with a clean, anomaly-free gap (`rounds [2,29]`, verified exhaustively for all 5 clients) in between.

## 7. `EXP_027` vs. `EXP_018` — Onset Generalization

| | `EXP_018` (onset 20) | `EXP_027` (onset 30) |
|---|---|---|
| Malicious client | `client_002` | `client_002` |
| True onset | 20 | 30 |
| FedTrace v3 predicted client | `client_002` | `client_002` |
| Onset estimate | 20 | 30 |
| Onset error | 0 | 0 |
| Confirmation round | 22 | 32 |
| Detection delay | 2 | 2 |
| Honest persistent FP | 0 | 0 |
| Dominant signal | `z_t_res`, 30/30 (100%) | `z_t_res`, 20/20 (100%) |
| Did magnitude-only eventually confirm? | Yes — round 49 | Yes — round 49 |
| Final test Top-1 | 53.10% | 53.17% |
| Final test Top-5 | 81.40% | 81.98% |

**The central scientific question — does frozen FedTrace v3 preserve correct attacker attribution and exact onset localization when the attack onset moves from round 20 to round 30, with no retuning of any kind — is answered yes, within this one controlled comparison.** Every pre-registered criterion (attribution, onset error, delay, honest FP) is identical between the two runs; the confirmation round shifts from 22 to 32, tracking the true onset exactly (`onset + 2` in both cases) rather than remaining fixed at some absolute round number — direct evidence that the detector's decision is genuinely onset-relative, not an artifact of the specific round-20 value used in every prior run through `EXP_025`.

**An important structural boundary this comparison does not test, stated explicitly:** round 30 remains safely outside the fixed `T_ref = 20` magnitude-calibration window (rounds `[0,20)`), so `client_002`'s honest-reference calibration in `EXP_027` is, like every prior run, computed entirely from genuinely pre-attack data. **An onset earlier than round 20 (e.g., round 10 or round 15) would test a structurally different and harder regime** — one where the calibration window itself would contain some already-attacked rows, a case no run in this project (including `EXP_027`) has ever exercised. This is noted as an explicit scope boundary, not evaluated here.

## 8. Performance Interpretation

| Comparison | Top-1 Δ | Top-5 Δ | Test loss Δ | Best validation Top-1 Δ |
|---|---|---|---|---|
| `EXP_027` vs. `EXP_008` (attacked-round30 vs. clean) | **−3.66 pp** | **−5.81 pp** | +0.3390 | **−3.40 pp** |
| `EXP_027` vs. `EXP_018` (round30 vs. round20, both attacked) | +0.07 pp | +0.58 pp | −0.0538 | −0.24 pp |
| (reference) `EXP_018` vs. `EXP_008` | −3.73 pp | −6.39 pp | +0.3927 | −3.16 pp |

**Moving the attack onset from round 20 to round 30 (10 fewer post-attack rounds, 10 more honest pre-attack rounds) produced essentially the same overall global-model degradation relative to the clean baseline** (−3.66pp vs. −3.73pp Top-1) — `EXP_027` vs. `EXP_018` are close to neutral relative to each other (+0.07pp Top-1). This is reported descriptively; this document does not attempt to disentangle "fewer poisoned rounds" from "more honest pre-attack convergence" as competing explanations, since doing so would require additional runs not authorized in this task.

**This performance comparison is kept strictly separate from the forensic attribution/onset claim, as instructed.** FedTrace v3's correct attribution and exact onset localization (§3) were computed entirely from `update_l2_norm`/`self_prev_round_cosine` trace signals, independent of the eventual global-model accuracy impact; the numbers in this section characterize the attack's *damage*, not FedTrace's ability to *detect* it.

---

## Final Report

1. **Files created:** `analysis/EXP_027_ONSET_GENERALIZATION_ANALYSIS.md`, `analysis/FEDTRACE_FINAL_GENERALIZATION_GATE_STATUS.md`.
2. **Pairing verdict:** confirmed — `EXP_027` differs from `EXP_018` scientifically only in `attack_start_round` (`20`→`30`); every other config field is identical. Trace-level check confirms bit-identical honest-phase data for rounds `[0,20)` across both runs (100/100 cells); the expected divergence in rounds `[20,29]` is mechanistically explained by `EXP_018`'s own earlier attack onset, not a pairing defect (§1).
3. **Attack correctness:** confirmed — `client_002` active only for rounds 30–49, no other client ever active; `delta_attack_norm_ratio=1.0` to within `2.19×10⁻¹⁰`; `realized_cosine_to_honest=0.0` to within `5.32×10⁻¹⁰` (§2).
4. **Diagnostics integrity:** confirmed — 250 rows, 50 unique rounds, 5 unique clients (§1).
5. **Attribution result:** predicted = true = `client_002`, correct (§3).
6. **Onset result:** onset estimate = 30, exactly the true onset (§3).
7. **Onset error:** **0** (§3).
8. **Detection delay:** **2** — confirmed to be the theoretical minimum under `P=3` (§3).
9. **Honest FP result:** **0** — no honest client ever confirmed (§3, §6).
10. **Dominant signal:** `z_t_res`, 20/20 (100%) of the confirmed run's anomalous rounds; magnitude-only eventually, separately, also confirmed correctly but far too late (onset error 17, delay 19) to satisfy the pre-registered per-run criteria on its own (§4, §5).
11. **`EXP_027` vs. `EXP_018` comparison:** identical outcome on every pre-registered criterion; confirmation round tracked the shifted true onset exactly (`onset+2` in both cases), demonstrating the detector's decision is onset-relative, not tied to the specific round-20 value used in every prior run (§7).
12. **Identity-generalization status (carried from `EXP_026`):** narrowed/supported by one new identity (`client_004`); not proof of arbitrary-identity invariance.
13. **Onset-generalization status:** narrowed/supported by one new onset (round 30); not proof of arbitrary-onset invariance. An onset earlier than `T_ref=20` remains a structurally distinct, untested case (§7).
14. **Combined evidence-gap status:** both axes the ICML/ICLR audit identified as the single essential remaining gap have now each been tested once, and both passed every pre-registered criterion with no retuning — see `analysis/FEDTRACE_FINAL_GENERALIZATION_GATE_STATUS.md` for the full synthesis and gate decision.
15. **Limitations:** one new identity and one new onset, each tested in isolation (never jointly), both at seed 123 only, both within the directional attack family only, both with onset kept safely later than `T_ref=20` (an earlier-than-`T_ref` onset was not tested); see the companion gate-status document for the complete, consolidated limitations list.
16. **Final gate A/B/C:** see `analysis/FEDTRACE_FINAL_GENERALIZATION_GATE_STATUS.md`.
17. **Whether another GPU experiment is needed:** see `analysis/FEDTRACE_FINAL_GENERALIZATION_GATE_STATUS.md`.
18. **Confirmation FedTrace v3 unchanged:** confirmed — `src/analysis/fedtrace_detector.py` was read and imported for computation only, never edited.
19. **Confirmation attack code unchanged:** confirmed — no file under `src/attacks/` was read or touched.
20. **Confirmation no experiment was run:** confirmed — only `EXP_027`'s already-existing result artifacts were read.
21. **Confirmation no code/test/README/result/database/master-workbook file changed:** confirmed — no existing file was created, edited, or deleted; only the two analysis documents listed in item 1 were created.
22. **Commands executed:** disclosed — `Bash` ran a read-only Python analysis script (written to the scratchpad directory, not the repository) importing the unmodified `src/analysis/fedtrace_detector.py` module and computing scores from already-collected `client_round_trace.csv`/`summary.json` files. No repository file was written by it, and no training, experiment-runner, or modification command of any kind was executed.
