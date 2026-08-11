# EXP_024 — First Directional Attacked Multi-Krum Run: Seed 123

**Status:** Analysis document only. No code, test, config, README, FedTrace, attack, or Research Database file was created or modified in producing it. No experiment was run. Multi-Krum's `f=1`/`m=4`, tie-breaking, weighting, buffer handling, and the pre-registered `P_krum=3` rejection proxy were not modified or tuned. FedTrace v3 remains frozen. Multi-Krum's exclusion pattern is referred to throughout as the **"Multi-Krum rejection proxy"**, never as an attribution detector.

Paired experiments used: `EXP_020` (clean Multi-Krum, seed 123), `EXP_018` (directional-attacked FedAvg, seed 123, including its already-frozen FedTrace v3 result), `EXP_008` (clean FedAvg, seed 123, for the 2×2). Cross-attack-family comparison uses `EXP_022` (Gaussian-attacked Multi-Krum, seed 123).

---

## 1. Pairing Verification

| Field | EXP_018 (attacked FedAvg) | EXP_024 (attacked Multi-Krum) | EXP_020 (clean Multi-Krum) | Match? |
|---|---|---|---|---|
| `dataset` | CIFAR-100 | CIFAR-100 | CIFAR-100 | ✅ |
| `model` | resnet18_cifar | resnet18_cifar | resnet18_cifar | ✅ |
| `seed` | 123 | 123 | 123 | ✅ |
| `dirichlet_alpha` | 0.1 | 0.1 | 0.1 | ✅ |
| `num_clients`/`clients_per_round` | 5/5 | 5/5 | 5/5 | ✅ |
| `communication_rounds_completed` | 50/50 | 50/50 | 50/50 | ✅ |
| `local_epochs` | 1 | 1 | — | ✅ |
| `optimizer`/`learning_rate`/`batch_size` | sgd/0.01/32 | sgd/0.01/32 | sgd/0.01/32 | ✅ |
| `client_sample_counts` | `{000:6024,001:9628,002:7202,003:10361,004:11785}` | **identical, integer-for-integer** | **identical, integer-for-integer** | ✅ same seed-123 partition realization used throughout this entire program |
| `attack_type` | norm_preserving_directional | norm_preserving_directional | none | ✅ |
| `malicious_client_id` | client_002 | client_002 | — | ✅ |
| `attack_start_round` | 20 | 20 | — | ✅ |
| `directional_target_cosine` | 0.0 | 0.0 | — | ✅ |
| `aggregation_strategy` | fedavg | **multikrum** | multikrum | **The sole scientific difference between EXP_024 and EXP_018** |

**No discrepancy found.** EXP_024 differs from EXP_018 scientifically only by `aggregation: fedavg → multikrum`. EXP_024's rounds 0–19 are byte-identical to EXP_020's (verified in Part 3), confirming the shared partition realization directly, not merely inferred from matching sample counts.

## 2. Directional Attack Correctness

`client_round_trace.csv` (250 rows) read in full. `client_002`'s `attack_active` is `False` for all `round_id < 20` and `True` for all `round_id ≥ 20`; no other client's `attack_active` is ever `True`.

**Norm preservation (`delta_attack_norm_ratio`), all 30 active rounds:** ranges from `0.9999999997244672` (round 30) to `1.0000000001989686` (round 33) — deviation from exact norm preservation is on the order of `1e-10`, i.e. floating-point precision, not a real deviation. **Mean deviation from 1.0: ≈ 3×10⁻¹¹.**

**Directional target (`realized_cosine_to_honest`), all 30 active rounds:** magnitudes range from `~3×10⁻¹²` to `9.69×10⁻¹⁰` (e.g. round 20: `-6.28×10⁻¹¹`; round 26: `-9.69×10⁻¹⁰`, the largest deviation observed; round 49: `-6.10×10⁻¹¹`) — indistinguishable from exactly `0.0`, the configured `directional_target_cosine`.

**The attack remained genuinely norm-preserving and correctly configured throughout the entire post-attack window**, confirmed directly from the trace's own dedicated verification fields, not assumed from the attack's name — consistent with `EXP_018`'s own already-frozen verification (`analysis/EXP_018_STAGE_B_DIRECTIONAL_SEED123_VALIDATION.md` §3).

## 3. Multi-Krum Diagnostics Audit

`multikrum_diagnostics.csv` read in full: 251 lines (1 header + 250 data rows). **Integrity — all checks pass:** 50 rounds × 5 clients = 250 rows; ranks 1–5 assigned exactly once per round in every round; exactly `m=4` selected/1 rejected every round, no exception; no missing or malformed rows.

**Determinism cross-check:** rounds 0–19 are identical, to the full floating-point digit, to `EXP_020`'s corresponding rounds (e.g. round 0: `client_004` score `200.57255083212604`, rank 5, in both).

**Full rejection/rank statistics (exact, computed from the CSV):**

| Client | Total rejections (of 50) | Rejections R0–R19 | Rejections R20–R49 | Pre-attack rate | Post-attack rate | Longest streak | First rejection round | Mean rank pre | Mean rank post | Median rank pre | Median rank post |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `client_000` | 0 | 0 | 0 | 0% | 0% | 0 | never | 1.00 | 1.00 | 1 | 1 |
| `client_001` | 0 | 0 | 0 | 0% | 0% | 0 | never | 3.00 | 2.07 | 3 | 2 |
| `client_002` (malicious) | **0** | 0 | **0** | 0% | **0%** | 0 | never | 2.00 | 2.93 | 2 | 3 |
| `client_003` | 1 | 1 | 0 | 5% | 0% | 1 (R1, isolated) | 1 | 4.05 | 4.00 | 4 | 4 |
| `client_004` | **49** | 19 | **30** | 95% | **100%** | **48 (R2–R49, unbroken)** | 0 | 4.95 | 5.00 | 5 | 5 |

**Mean Krum score, pre- vs. post-attack (`client_002`):** rises from a pre-attack range of ≈103–115 (mean ≈110) to a post-attack range of 113.68 (round 20) → 146.46 (round 49), a real, gradual, roughly linear increase (≈1.1/round) — elevated relative to its own pre-attack trend, but never remotely approaching `client_004`'s post-attack range (158–190, mean ≈170).

## 4. Critical Question: Does Multi-Krum Reject the Directional Attacker?

- `client_002` rank at round 19 (last clean round): **2**.
- `client_002` rank at round 20 (attack onset): **2** — unchanged.
- `client_002` rank trajectory, rounds 20–49: **2 (rounds 20–21), then 3 (rounds 22–49)** — a real, measurable, one-position elevation, but never below rank 3.
- Post-attack rounds rejected: **0 of 30 (0%)**.
- Post-attack rounds selected: **30 of 30 (100%)**.
- Longest rejection streak (post-attack): **0** — `client_002` is never rejected at any point in the entire 50-round run.
- Longest selection streak: **50 rounds** (the entire run, pre- and post-attack combined).

**Classification: C — the attacker is not rejected at all; it evades Multi-Krum's exclusion completely and continuously.** This is not "mostly selected" as a matter of degree — `client_002` is selected in literally every single round of the run, with no exception. This is determined entirely from the diagnostics counts above, not inferred from model accuracy.

## 5. Rounds 15–25

| Round | Rejected client | `client_002` rank | `client_002` score | `client_004` rank | `client_004` selected/rejected | `attack_active` |
|---|---|---|---|---|---|---|
| 15 | `client_004` | 2 | 113.27 | **5 (rejected)** | rejected | False |
| 16 | `client_004` | 2 | 112.47 | **5 (rejected)** | rejected | False |
| 17 | `client_004` | 2 | 112.37 | **5 (rejected)** | rejected | False |
| 18 | `client_004` | 2 | 111.64 | **5 (rejected)** | rejected | False |
| 19 | `client_004` | 2 | 114.89 | **5 (rejected)** | rejected | False |
| **20 (attack starts)** | **`client_004`** | **2** | **113.68** | **5 (rejected)** | **rejected** | **True** |
| 21 | `client_004` | 2 | 118.19 | 5 (rejected) | rejected | True |
| 22 | `client_004` | **3** | 120.16 | 5 (rejected) | rejected | True |
| 23 | `client_004` | 3 | 122.38 | 5 (rejected) | rejected | True |
| 24 | `client_004` | 3 | 124.01 | 5 (rejected) | rejected | True |
| 25 | `client_004` | 3 | 124.25 | 5 (rejected) | rejected | True |

**No switch occurs.** The rejected client at round 20 is `client_004` — identical to round 19, and identical to every round from 2 through 49. `client_002`'s rank moves modestly, from 2 to 3, over rounds 20–22, but never comes close to rank 5.

**Comparison with `EXP_022` (Gaussian, seed 123):** there, `client_002` jumped from rank 2 (score 114.89, round 19) to rank 5 (score 170.80, round 20) in a single round — a +55.9 (+48.7%) score increase and an instantaneous, complete regime switch. Here, `client_002`'s score increases much more modestly (113.68 at round 20, versus its own pre-attack round-19 value of 114.89 — **initially no increase at all**, then a gradual climb to 146.46 by round 49) and **no regime switch occurs at any point.** This is a direct, quantified, same-seed contrast between the two attack families' effect on Multi-Krum's raw distance metric.

## 6. Full-Run Rejection Proxy (`P_krum=3`, Not Reset at Round 20)

- **`client_004`:** rejected round 0 (streak 1), selected round 1 (reset), rejected rounds 2–49 (48 consecutive, unbroken by the attack). First reaches 3 consecutive at round 4 — **persistently confirmed at round 4**. Onset estimate = round 2. Remains confirmed continuously for the rest of the run.
- **`client_003`:** rejected only once (round 1) — never confirmed.
- **`client_002`:** **never rejected at any round — never confirmed, at any point in the run.**
- **`client_000`, `client_001`:** never rejected — never confirmed.

**Predicted suspicious client over the full run: `client_004`** — confirmed at round 4. **This is incorrect** — `client_004` is honest.

**Malicious client's own confirmation: none, ever.** This is a materially *worse* outcome for the proxy than `EXP_022`'s Gaussian case, where the true attacker was at least eventually confirmed (round 22, 18 rounds after the honest false positive). Here, the proxy provides **zero signal distinguishing `client_002` from any of the other three never-rejected honest clients (`client_000`, `client_001`, `client_003` after round 1)** — it does not merely misattribute *first*, it never attributes the true attacker **at all**.

**Confirmation ordering:** `client_004` (round 4) — the only confirmation event in the entire run. `client_002` has no confirmation round to compare against.

## 7. EXP_024 vs. EXP_020 (Directional Attack Effect Under Multi-Krum)

| Metric | EXP_020 (clean MK) | EXP_024 (attacked MK) | Difference (024 − 020) |
|---|---|---|---|
| Final test loss | 2.1901 | 2.9546 | **+0.7644** (much worse) |
| Test Top-1 | 49.91% | 44.36% | **−5.55 pp** |
| Test Top-5 | 80.06% | 70.26% | **−9.80 pp** |
| Macro precision | 54.31% | 44.59% | **−9.72 pp** |
| Macro recall | 49.91% | 44.36% | **−5.55 pp** |
| Macro F1 | 46.16% | 38.50% | **−7.66 pp** |
| Balanced accuracy | 49.91% | 44.36% | **−5.55 pp** |
| Best validation Top-1 | 51.44% (round 48) | 45.68% (round 47) | **−5.76 pp** |

**This is the opposite direction from the Gaussian result** (`EXP_022` outperformed `EXP_020` by +4.52 pp Top-1). Here, the attacked run is substantially *worse* than the clean control on every metric.

**Trajectory around round 20:** rounds 0–19 are byte-identical between `EXP_024` and `EXP_020` (round 19 validation Top-1: 38.72% in both). At round 20 itself, `EXP_024` already trails `EXP_020` (38.52% vs. 39.86%, **−1.34 pp**) — **immediate**, not delayed, degradation. The gap widens over the following rounds (round 25: 37.9% vs. 42.42%, −4.52 pp; round 40: 43.32% vs. 48.22%, −4.90 pp) and settles into a roughly stable **−4.9 to −5.8 pp plateau** for the remainder of the run — it does not recover.

**Classification: immediate degradation, settling into a persistent plateau.** No recovery is observed by round 49.

## 8. EXP_024 vs. EXP_018 (Multi-Krum vs. FedAvg Under the Same Directional Attack)

| Metric | EXP_018 (attacked FedAvg) | EXP_024 (attacked Multi-Krum) | Difference (024 − 018) |
|---|---|---|---|
| Final test loss | 1.8582 | 2.9546 | **+1.0964** (much worse) |
| Test Top-1 | 53.10% | 44.36% | **−8.74 pp** |
| Test Top-5 | 81.40% | 70.26% | **−11.14 pp** |
| Macro precision | 58.90% | 44.59% | **−14.31 pp** |
| Macro recall | 53.11% | 44.36% | **−8.75 pp** |
| Macro F1 | 50.61% | 38.50% | **−12.11 pp** |
| Balanced accuracy | 53.11% | 44.36% | **−8.75 pp** |
| Best validation Top-1 | 55.10% (round 48) | 45.68% (round 47) | **−9.42 pp** |

**Does Multi-Krum mitigate the directional attack better than FedAvg? No — stated directly: Multi-Krum performs substantially worse than FedAvg under this attack, by −8.7 to −14.3 percentage points across every metric — a much larger gap than the Gaussian case's −2.1 to −5.5 pp.**

**Separating robust-exclusion behavior from resulting global utility:**
- **Robust-exclusion behavior:** Multi-Krum completely fails to exclude `client_002` (0/30 post-attack rejections, Part 4) — it provides *no* exclusion-based protection against this attack at all.
- **Resulting global utility:** independent of the exclusion failure, Multi-Krum's aggregation rule itself compounds the damage. `client_002` holds only 7,202/45,000 = 16.0% of total training samples; under FedAvg's sample-weighted average, its poisoned update receives exactly that 16.0% weight every round it participates. Under Multi-Krum's **unweighted** average of the `m=4` selected clients, `client_002`'s poisoned update instead receives a full **25% (1/4) weight** every one of the 30 post-attack rounds it is selected (i.e. all of them) — objectively *more* influence over the aggregate than FedAvg would give it, on top of providing zero exclusion benefit. This weighting-rule fact is directly confirmed from `src/federated/aggregation/multikrum.py` and Part 4's selection counts (§12 below separates this into confirmed fact vs. supported interpretation vs. hypothesis).

## 9. Complete Seed-123 Directional 2×2

|  | CLEAN | DIRECTIONAL ATTACK |
|---|---|---|
| **FedAvg** | EXP_008 | EXP_018 |
| **Multi-Krum** | EXP_020 | EXP_024 |

**Top-1:** EXP_008=56.83%, EXP_018=53.10%, EXP_020=49.91%, EXP_024=44.36%.
1. Clean aggregation effect (EXP_020−EXP_008): **−6.92 pp**
2. Attack effect under FedAvg (EXP_018−EXP_008): **−3.73 pp**
3. Attack effect under Multi-Krum (EXP_024−EXP_020): **−5.55 pp**
4. Aggregation effect under attack (EXP_024−EXP_018): **−8.74 pp**

**Top-5:** EXP_008=87.79%, EXP_018=81.40%, EXP_020=80.06%, EXP_024=70.26%.
1. Clean aggregation effect: **−7.73 pp**
2. Attack effect under FedAvg: **−6.39 pp**
3. Attack effect under Multi-Krum: **−9.80 pp**
4. Aggregation effect under attack: **−11.14 pp**

**Interpretation:** unlike the Gaussian family, where the attack effect flipped sign between FedAvg (−0.33 pp, near-neutral) and Multi-Krum (+4.52 pp, apparently positive), **here the attack effect is negative under both aggregation strategies, and substantially larger in magnitude under Multi-Krum than under FedAvg** (−5.55 pp vs. −3.73 pp Top-1; −9.80 pp vs. −6.39 pp Top-5). This is a genuine, measured interaction — the directional attack is not "neutral" for either method here, and Multi-Krum's own aggregation/exclusion behavior actively makes the outcome worse than FedAvg's simple averaging, not better. **This is a single-seed result; no statistical-significance claim is made.**

## 10. Gaussian vs. Directional Multi-Krum, Same Seed

| | `EXP_022` (Gaussian) | `EXP_024` (Directional) |
|---|---|---|
| Attacker post-attack rejection rate | **30/30 (100%)** | **0/30 (0%)** |
| Attacker rank behavior post-attack | 5 (rejected), every round | 2→3 (selected), every round |
| Rejection-target switch at round 20? | Yes, instantaneous | **No — no switch at all** |
| Full-run proxy attribution | `client_004` (wrong), attacker confirmed later (round 22) | `client_004` (wrong), attacker **never** confirmed |
| Test Top-1 | 54.43% | **44.36%** |
| Test Top-5 | 82.59% | **70.26%** |
| Attacked vs. clean Multi-Krum (Top-1) | **+4.52 pp** | **−5.55 pp** |
| Attacked Multi-Krum vs. attacked FedAvg (Top-1) | −2.07 pp | **−8.74 pp** |
| Validation trajectory vs. clean control | Ahead from round 20 onward | Behind from round 20 onward, growing |

**Is norm-preserving directional manipulation materially harder for Multi-Krum than Gaussian update noise? Yes — decisively, and confirmed by direct experimental comparison, not merely predicted.** `EXP_022`'s own trace analysis (`analysis/EXP_022_MULTIKRUM_GAUSSIAN_SEED123_ANALYSIS.md` Part H) had offered, as a hypothesis, that Multi-Krum's raw Euclidean-distance metric separates the Gaussian attacker primarily via its magnitude inflation (`delta_attack_norm_ratio` is unconstrained/large under Gaussian noise, ≈1.0 under the directional attack by design). **This run directly tests and confirms that hypothesis's converse**: an attack engineered to leave magnitude essentially unchanged (`delta_attack_norm_ratio ≈ 1.0` throughout, Part 2) produces a real but small score elevation (Part 3) — never large enough to overtake `client_004`'s much larger, heterogeneity-driven baseline distance, and Multi-Krum's rejection mechanism provides **zero** exclusion-based protection. This is potentially central to the paper's argument, exactly as the task anticipated.

## 11. FedTrace vs. Multi-Krum

Using the already-frozen, already-validated FedTrace v3 result for `EXP_018` (not re-run on `EXP_024`'s own trace in this task):

| | FedTrace v3 (`EXP_018`) | Multi-Krum rejection proxy (`EXP_024`, full run) |
|---|---|---|
| Predicted attacker | `client_002` | `client_004` (honest) |
| Onset | 20 (error 0) | 2 (the honest false positive's own onset) |
| Confirmation delay | 2 (round 22) | Not applicable — the true attacker is never confirmed at all |
| Honest persistent FP | **0** | **1** (`client_004`) |
| Signal used | `z_t_res` (temporal-residual), dominant in 100% of `client_002`'s 30 anomalous rounds per `EXP_018`'s own analysis (`analysis/EXP_018_STAGE_B_DIRECTIONAL_SEED123_VALIDATION.md` §15) | Raw per-round Euclidean distance to the other 4 clients' updates, no per-client history |
| Requires known `f`? | No | Yes (`f=1`) |
| Actual selection/rejection behavior (this run) | N/A (FedTrace does not aggregate) | `client_002` selected 30/30 post-attack rounds; `client_004` rejected 30/30 |

`EXP_024`'s own trace corroborates the mechanism: `client_002`'s `self_prev_round_cosine` collapses from a pre-attack range of ≈0.24–0.35 to essentially `0` (magnitudes `<0.001`, e.g. round 20: `-6.6×10⁻⁵`; round 49: `-7.7×10⁻⁴`) at every post-attack round — this is exactly the signal `z_t_res` is built to detect, and exactly the signal Multi-Krum's Euclidean-distance-only scoring has no analog for.

**Does the directional experiment provide stronger evidence that robust aggregation and forensic traceability solve different problems? Yes — stronger than the Gaussian evidence alone.** In the Gaussian case, Multi-Krum's raw per-round scoring *did* separate the attacker geometrically (100% rejection); the failure was confined to the *full-run attribution* level (the earlier honest false positive winning the "earliest confirmed" race). Here, Multi-Krum's raw per-round scoring **fails to separate the attacker at all, at any round** — a categorical, not merely a proxy-application, failure — while FedTrace, via its temporal-residual component, succeeds with identical timing to the Gaussian case (onset 20, confirm 22, zero honest FP). **No superiority claim beyond this evidence is made**: this shows FedTrace's cross-attack-family coverage handles a case Multi-Krum's exclusion mechanism cannot handle at all in this configuration; it does not show FedTrace would succeed against every conceivable attack, and it says nothing about Multi-Krum's native aggregation-robustness value against other, more magnitude-revealing attacks (where, per `EXP_022`, it performs a real exclusion function, even if that exclusion did not translate to net utility gain over FedAvg).

## 12. Why Did Performance Collapse?

**CONFIRMED FACT:**
- `client_002` is selected (never rejected) in 100% of post-attack rounds (Part 4).
- `client_004` is rejected in 100% of post-attack rounds and 95% of pre-attack rounds (Part 3) — unchanged or slightly worsened relative to the clean control.
- Multi-Krum's selected-client averaging is unweighted (`1/m = 1/4` per selected client), confirmed directly from `src/federated/aggregation/multikrum.py`; `client_002` holds 16.0% (7,202/45,000) of total training samples.
- `client_002`'s Krum score is measurably elevated post-attack relative to its own pre-attack trend (Part 3), but never approaches `client_004`'s score range.
- The directional attack causes real, substantial damage under plain FedAvg too (`EXP_018` vs. `EXP_008`: −3.73 pp Top-1, already established and re-confirmed here, not merely "essentially neutral" as the Gaussian family was).
- EXP_024 underperforms both EXP_020 (clean Multi-Krum) and EXP_018 (attacked FedAvg) on every metric, by a substantially larger margin than EXP_022 showed relative to EXP_016.

**SUPPORTED INTERPRETATION** (evidence-grounded, not a full mathematical decomposition of exact causal contribution):
- The combination of (a) complete failure to exclude the attacker and (b) `client_002`'s poisoned update receiving *more* proportional weight under Multi-Krum's unweighted scheme (25%) than it would under FedAvg's sample-weighted scheme (16%) whenever selected, is the most direct, evidence-consistent explanation for why Multi-Krum's total degradation exceeds FedAvg's under this specific attack — both effects point in the same, harmful direction, and both are independently confirmed facts (above).
- The norm-preserving attack's near-zero magnitude change produces too small an increase in Euclidean distance for Multi-Krum's raw-distance scoring to separate `client_002` from the honest cluster, especially given `client_004`'s pre-existing, much larger heterogeneity-driven distance already occupying the single `f=1` exclusion slot every round (Part 4/5).

**HYPOTHESIS (not directly tested in this task, and not to be tested by tuning per this task's restrictions):**
- Whether a stronger directional rotation (a `target_cosine` further from the honest direction than `0.0`) would eventually produce enough distance increase to trigger exclusion — untested; no such config exists or is proposed here.
- Whether norm-preserving, direction-only manipulation is a general blind spot of raw-Euclidean-distance robust aggregation across neural architectures generally, beyond this one configuration — plausible given this result, not established beyond it.

**Explicitly not supported by this evidence:** that Multi-Krum's exclusion mechanism was "confused" by client_004's heterogeneity into somehow protecting `client_002` — the two facts (client_004's exclusion; client_002's evasion) are independently, mechanistically explained (Part 3/4) and their co-occurrence does not require or imply any interaction beyond the `m=4`/`f=1` slot being fully occupied by `client_004` every round, leaving no exclusion capacity free to catch a geometrically weaker anomaly even if one existed.

## 13. Paper-Level Verdict

**CONFIRMED (this run, directly measured):** attack correctness and norm-preservation to floating-point tolerance (Part 2); Multi-Krum diagnostics integrity (Part 3); `client_002`'s complete evasion of rejection (Part 4); the absence of any regime switch (Part 5); the full-run proxy's misattribution with the true attacker never confirmed (Part 6); every metric delta in Parts 7–9; the Gaussian-vs-directional contrast (Part 10, both runs' own diagnostics directly compared); FedTrace's success on `EXP_018` (cited, already independently Stage-B-validated) and Multi-Krum's categorical failure on the same attack under the same seed (Part 11).

**SINGLE-SEED:** every quantitative finding in this document — seed 123 only; no seed-2026 directional Multi-Krum run exists yet to compare against.

**HYPOTHESIS:** the exact causal decomposition of *how much* of the performance gap is attributable to the unweighted-vs-sample-weighted averaging difference versus the exclusion failure itself (Part 12); whether a stronger directional attack would eventually trigger exclusion; whether this is a general property of raw-Euclidean-distance robust aggregation beyond this configuration.

**UNSUPPORTED (claims not to be made):** that Multi-Krum's native robust-aggregation function is defective in general (only shown here, in this one configuration, to fail against this specific attack geometry while performing a real, if net-unhelpful, exclusion function against a different one, `EXP_022`); that this result generalizes to seed 2026, other `target_cosine` values, other client counts, or other attack strengths; that FedTrace would succeed against every conceivable attack; that Multi-Krum is "worse than FedAvg" as a general claim rather than specifically under this attack/aggregation-weighting interaction.

**Regarding the paper's central contribution:**
- **Malicious-client traceability:** strengthened — this run adds a case where Multi-Krum's rejection proxy provides not merely a *misleading* signal (as in the Gaussian case) but **no signal at all** about the true attacker, while FedTrace succeeds with identical, already-established timing.
- **Onset localization:** FedTrace's onset estimate (round 20, error 0) is directly confirmed again on an attack family Multi-Krum's raw scoring cannot separate at all — reinforcing that onset localization is a property of FedTrace's per-client temporal calibration, not a byproduct available from any distance-based exclusion rule.
- **Attack-geometry complementarity:** **directly, empirically demonstrated for the first time in this program** (Part 10) — the same seed, same attacker, same onset, same client, under two attack families that differ only in whether magnitude is preserved, produce opposite qualitative outcomes for Multi-Krum (100% exclusion vs. 0% exclusion) while FedTrace's temporal-residual/magnitude coverage design handles both with materially the same success.
- **Limits of robust aggregation as an attribution substitute:** strengthened further and in a new way — not only does the repurposed proxy misattribute (as in the Gaussian case), here it provides **zero evidence of any kind** about the true attacker, which is arguably a *stronger* limitation than a misattribution (a misattribution at least produces a wrong-but-present signal; here there is no signal to evaluate for the true attacker at all).

## 14. Next Gate

**Decision: A — EXP_024 is technically valid and scientifically interpretable; create/run the seed-2026 directional Multi-Krum replication next.**

Justification: no implementation or methodological problem was found (diagnostics fully intact, attack verified genuinely norm-preserving to floating-point tolerance, pairing confirmed exactly). The result is striking and, if anything, *more* informative for the paper than a confirmatory replication would be uninformative to skip — but it is a single seed, and the entire Gaussian-family branch of this program only reached its two-seed replicated conclusions after testing both seeds; the directional family's single most important finding (complete attacker evasion, §4/§10) has not yet been tested for seed-dependence at all. Per this task's explicit restriction, **EXP_025 (the seed-2026 directional Multi-Krum config) is not created in this task.**

**Exact next experiment:** the Multi-Krum analog of `EXP_019` — norm-preserving directional attack, `client_002`, round-20 onset, `directional_target_cosine=0.0`, seed 2026, paired against `EXP_009` (clean FedAvg), `EXP_019` (attacked FedAvg, directional, seed 2026), and `EXP_021` (clean Multi-Krum, seed 2026) — mirroring the Gaussian family's own seed-123-then-seed-2026 sequence exactly.

---

## Final Report

1. **Files created:** `analysis/EXP_024_MULTIKRUM_DIRECTIONAL_SEED123_ANALYSIS.md`, `analysis/GAUSSIAN_VS_DIRECTIONAL_MULTIKRUM_SEED123_SYNTHESIS.md`.
2. **Pairing verdict:** valid — EXP_024 differs from EXP_018 only by `aggregation: fedavg→multikrum`; identical partition to EXP_020/EXP_018/EXP_008 (byte-identical pre-attack diagnostics/trajectories).
3. **Attack-correctness verdict:** confirmed — genuinely norm-preserving (`delta_attack_norm_ratio≈1.0` to ~1e-10) and correctly targeted (`realized_cosine_to_honest≈0.0` to ~1e-9) at every one of the 30 active rounds.
4. **Diagnostics integrity:** pass — 250/250 rows, fully structurally valid.
5. **Pre/post rejection counts:** `client_004`=19 pre/**30 post**; `client_003`=1 pre/0 post; `client_002`=**0 pre/0 post**; `client_000`/`client_001`=0/0.
6. **`client_002` post-attack rejection/selection rate:** **0/30 rejected (0%), 30/30 selected (100%)** — complete evasion.
7. **Attacker rank behavior:** rank 2 (rounds 20–21), rank 3 (rounds 22–49) — never below rank 3, never rejected.
8. **Regime-switch behavior:** **none** — `client_004` remains the rejected client every round, uninterrupted by the attack.
9. **Full-run `P_krum` attribution result:** misattributes `client_004` (confirmed round 4); `client_002` is **never** confirmed at any point in the run.
10. **EXP_024 vs. EXP_020:** −5.55 pp Top-1, −9.80 pp Top-5 (attacked worse than clean) — opposite direction from the Gaussian result.
11. **EXP_024 vs. EXP_018:** −8.74 pp Top-1, −11.14 pp Top-5 (Multi-Krum much worse than FedAvg) — a substantially larger gap than the Gaussian case.
12. **Directional seed-123 2×2:** clean-agg effect −6.92 pp; attack-under-FedAvg −3.73 pp; attack-under-Multi-Krum −5.55 pp; aggregation-under-attack −8.74 pp (Top-1) — attack is harmful under both aggregators here, more so under Multi-Krum.
13. **Gaussian vs. directional Multi-Krum result:** decisively different — 100% vs. 0% attacker rejection rate, same seed, same attacker, same onset; directional manipulation is materially harder for Multi-Krum to catch.
14. **FedTrace vs. Multi-Krum interpretation:** FedTrace succeeds (onset 20, delay 2, zero honest FP, via `z_t_res`); Multi-Krum's rejection proxy provides zero evidence about the true attacker at any point — a stronger demonstration of the robust-aggregation/traceability distinction than the Gaussian case alone.
15. **Explanation of performance degradation:** confirmed facts (complete evasion; unweighted 25%>16% relative over-weighting of the poisoned update) plus a supported interpretation connecting them; exact causal decomposition remains a hypothesis.
16. **Paper-level contribution verdict:** strengthened across all four dimensions asked about (traceability, onset localization, attack-geometry complementarity, limits of robust aggregation as an attribution substitute) — see Part 13 for exact scope.
17. **Limitations:** single seed (123); one directional severity (`target_cosine=0.0`); FedTrace v3 not re-run on EXP_024's own trace (EXP_018's frozen result used); causal weighting-vs-exclusion decomposition not directly measured; no statistical significance claimed from n=1.
18. **Next-gate decision:** **A** — technically valid; proceed to seed-2026 directional Multi-Krum replication.
19. **Exact next experiment if A:** Multi-Krum analog of `EXP_019` (directional attack, seed 2026, paired with `EXP_009`/`EXP_019`/`EXP_021`). **Not created in this task.**
20. **Confirmation no implementation/config/test/result changed:** confirmed — `src/federated/aggregation/multikrum.py`, `src/analysis/fedtrace_detector.py`, `src/attacks/traceability.py`, all configs, all tests, `README.md`, Research_Database, and every existing analysis/result file were untouched; only pre-existing result artifacts and prior analysis/plan documents were read.
21. **Confirmation no experiment was run:** confirmed — no training script was executed.
22. **Confirmation whether commands were executed:** none — this task used only read-only file inspection (`Read`) and file-pattern search (`Glob`) tool calls; no shell, PowerShell, Bash, or Python command was executed.
