# EXP_022 — First Attacked Multi-Krum Run: Gaussian Attack, Seed 123

**Status:** Analysis document only. No code, test, config, README, FedTrace, attack, or Research Database file was created or modified in producing it. No experiment was run. Multi-Krum's `f=1`/`m=4`, tie-breaking, weighting, buffer handling, and the pre-registered `P_krum=3` rejection proxy were not modified or tuned. FedTrace v3 remains frozen. Multi-Krum's exclusion pattern is referred to throughout as the **"Multi-Krum rejection proxy"**, never as an attribution detector in its own right.

Paired experiments used: `EXP_020` (clean Multi-Krum, seed 123, the aggregation-matched control), `EXP_016` (Gaussian-attacked FedAvg, seed 123, the attack-matched aggregation comparison, including its already-frozen FedTrace v3 result from `analysis/EXP_016_STAGE_B_GAUSSIAN_SEED123_VALIDATION.md`), `EXP_008` (clean FedAvg, seed 123, used only to complete the 2×2 factorial in Part F).

---

## A. Pairing Verification

Confirmed directly from each run's own `summary.json`:

| Field | EXP_016 (attacked FedAvg) | EXP_022 (attacked Multi-Krum) | EXP_020 (clean Multi-Krum) | Match? |
|---|---|---|---|---|
| `dataset` | CIFAR-100 | CIFAR-100 | CIFAR-100 | ✅ |
| `model` | resnet18_cifar | resnet18_cifar | resnet18_cifar | ✅ |
| `seed` | 123 | 123 | 123 | ✅ |
| `dirichlet_alpha` | 0.1 | 0.1 | 0.1 | ✅ |
| `num_clients`/`clients_per_round` | 5/5 | 5/5 | 5/5 | ✅ |
| `communication_rounds_completed` | 50/50 | 50/50 | 50/50 | ✅ |
| `local_epochs` | 1 | 1 | — | ✅ |
| `optimizer`/`learning_rate`/`batch_size` | sgd/0.01/32 | sgd/0.01/32 | sgd/0.01/32 | ✅ |
| `client_sample_counts` | `{000:6024,001:9628,002:7202,003:10361,004:11785}` | **identical, integer-for-integer** | **identical, integer-for-integer** | ✅ same partition realization across all three runs |
| `attack_type` | gaussian_update_noise | gaussian_update_noise | none | ✅ (EXP_020 clean, both attacked runs identical) |
| `malicious_client_id` | client_002 | client_002 | — | ✅ |
| `attack_start_round` | 20 | 20 | — | ✅ |
| `attack_strength_rho` | 1.0 | 1.0 | — | ✅ |
| `aggregation_strategy` | fedavg | **multikrum** | multikrum | **The sole scientific difference between EXP_022 and EXP_016** |

**No discrepancy found.** EXP_022 differs from EXP_016 scientifically only by `aggregation: fedavg → multikrum`; EXP_022 and EXP_020 share an identical partition realization (confirmed both by the sample-count match above and, more strongly, by the fact — established in Part B — that EXP_022's own round-0-through-19 Multi-Krum diagnostics and validation trajectory are **byte-identical** to EXP_020's, since the attack is inactive for those rounds).

## B. Multi-Krum Diagnostics Audit

`multikrum_diagnostics.csv` (`results/federated/cifar100_multikrum_dirichlet_a01_gaussian_attack_client002_round20_seed123/`) read in full: 251 lines (1 header + 250 data rows).

**Integrity — all checks pass:** 50 rounds × 5 clients = 250 rows, exactly as expected; ranks 1–5 assigned exactly once per round in every round; exactly `m=4` selected and exactly 1 rejected per round, every round, no exception; no missing or malformed rows.

**Determinism cross-check (not requested but directly verifiable and important):** rounds 0–19's `krum_score`/`krum_rank`/`multikrum_selected` values are **identical, to the full floating-point digit, to EXP_020's** corresponding rounds (e.g. round 0: `client_004` score `200.57255083212604`, rank 5, in both runs; round 19: `client_004` score `154.46062117972684`, rank 5, in both). This is expected — the attack is inactive before round 20, so with identical seed/partition/model initialization, EXP_022's honest training and EXP_020's honest training are the same computation — but it is a strong, directly-observed confirmation that everything diverging after round 20 is attributable to the attack activating, not to any other confound.

**Rejection counts and rank statistics (exact, computed from the CSV):**

| Client | Total rejections (of 50) | Rejections R0–R19 | Rejections R20–R49 | Rejection rate pre-attack | Rejection rate post-attack | Longest streak | Mean rank pre | Mean rank post | Median rank pre | Median rank post |
|---|---|---|---|---|---|---|---|---|---|---|
| `client_000` | 0 | 0 | 0 | 0% | 0% | 0 | 1.00 | 1.00 | 1 | 1 |
| `client_001` | 0 | 0 | 0 | 0% | 0% | 0 | 3.00 | 2.00 | 3 | 2 |
| `client_002` (malicious) | **30** | **0** | **30** | **0%** | **100%** | **30 (R20–R49, unbroken)** | 2.00 | **5.00** | 2 | 5 |
| `client_003` | 1 | 1 | 0 | 5% | 0% | 1 (R1, isolated) | 4.05 | 3.00 | 4 | 3 |
| `client_004` | 19 | 19 | 0 | 95% | **0%** | 18 (R2–R19) | 4.95 | **4.00** | 5 | 4 |

**Preliminary observations, verified exactly, not assumed:**
- **R0–R19: `client_004` receives 19/20 rejections, `client_003` receives the remaining 1 (round 1).** **Confirmed exactly** — `client_004` is rejected at R0 and R2–R19 (19 rounds); `client_003` is rejected only at R1; `client_000`, `client_001`, `client_002` are never rejected pre-attack. 19 + 1 = 20, matching the 20 pre-attack rounds exactly.
- **R20–R49: `client_002` receives 30/30 rejections and rank 5 in every attacked round.** **Confirmed exactly** — every one of the 30 rows for `client_002` from round 20 through round 49 shows `krum_rank=5`, `multikrum_selected=False`. No exception anywhere in the post-attack window.

## C. Quantifying the Rejection Regime Switch (Rounds 15–25)

| Round | Rejected client | `client_002` rank | `client_002` score | `client_004` rank | `client_004` score |
|---|---|---|---|---|---|
| 15 | `client_004` | 2 | 113.27 | **5 (rejected)** | 150.91 |
| 16 | `client_004` | 2 | 112.47 | **5 (rejected)** | 150.58 |
| 17 | `client_004` | 2 | 112.37 | **5 (rejected)** | 153.01 |
| 18 | `client_004` | 2 | 111.64 | **5 (rejected)** | 152.62 |
| 19 | `client_004` | 2 | 114.89 | **5 (rejected)** | 154.46 |
| **20 (attack starts)** | **`client_002`** | **5 (rejected)** | **170.80** | **4** | **158.17** |
| 21 | `client_002` | 5 (rejected) | 175.91 | 4 | 152.31 |
| 22 | `client_002` | 5 (rejected) | 178.68 | 4 | 150.26 |
| 23 | `client_002` | 5 (rejected) | 181.69 | 4 | 146.72 |
| 24 | `client_002` | 5 (rejected) | 182.65 | 4 | 150.78 |
| 25 | `client_002` | 5 (rejected) | 183.06 | 4 | 148.04 |

**The switch is instantaneous — one round, zero lag.** At round 19 (last clean round), `client_002` sits at rank 2 (score 114.89), well inside the trusted majority, while `client_004` sits at rank 5 (rejected, score 154.46). At round 20 — exactly `attack_start_round`, the first round the attack is active — `client_002`'s score jumps to 170.80 (a **+55.9, +48.7% relative** single-round increase) and it becomes the rejected client, while `client_004`'s score barely moves (154.46→158.17, a normal round-to-round drift) but is no longer the *worst*-scoring client because `client_002` now scores far higher. **`client_004` immediately stops being rejected, and `client_002` immediately becomes the rejected client, in the same round the attack activates — there is no transition period.** Both trends hold, unbroken, for all 30 remaining rounds (R20–R49); rank assignment among all 5 clients is in fact *more* stable post-attack than pre-attack (zero rank swaps among the 4 selected clients across all 30 post-attack rounds, versus one minor swap observed pre-attack in the clean-control analog, `EXP_020`'s round-47 `client_001`/`client_002` crossover) — the attack signal is strong enough to leave no ambiguity about the ranking order.

This is accurately described, for this controlled experiment specifically, as a **"heterogeneity-dominated clean rejection → attack-dominated rejection regime switch."** This characterization is **not** claimed as universal; it describes what was directly observed in this one seed/attack-strength/partition combination.

## D. Clean Multi-Krum (EXP_020) vs. Attacked Multi-Krum (EXP_022)

| Metric | EXP_020 (clean MK) | EXP_022 (attacked MK) | Difference (022 − 020) |
|---|---|---|---|
| Final test loss | 2.1901 | 1.7933 | **−0.3968** (lower/better) |
| Test Top-1 | 49.91% | 54.43% | **+4.52 pp** |
| Test Top-5 | 80.06% | 82.59% | **+2.53 pp** |
| Macro precision | 54.31% | 59.09% | **+4.78 pp** |
| Macro recall | 49.91% | 54.43% | **+4.52 pp** |
| Macro F1 | 46.16% | 51.61% | **+5.46 pp** |
| Balanced accuracy | 49.91% | 54.43% | **+4.52 pp** |
| Best validation Top-1 | 51.44% (round 48) | 55.84% (round 48) | **+4.40 pp** |

**EXP_022 (attacked) performs *better* than EXP_020 (clean) on every single metric.** This fact is reported exactly as observed — it is not an artifact of a typo or a mismatched comparison; both final-metrics files were read directly.

**Did the Gaussian attack hurt Multi-Krum, or did Multi-Krum successfully exclude the malicious update?** Rejection alone does not answer this (Part B/C already show `client_002` is rejected in all 30 post-attack rounds — perfect exclusion, by the rejection-count evidence alone). Performance evidence tells a more specific story: the round-history trajectories for EXP_020 and EXP_022 are byte-identical through round 19 (validation Top-1 at round 10: 28.4% in both; round 19: 38.72% in both), then **diverge starting exactly at round 20** — EXP_022's validation Top-1 at round 20 is 41.2% versus EXP_020's 39.86% (already +1.34 pp at the very first attacked round), and the gap grows to +4.40 pp by the best-validation round. The divergence begins at the same round the rejection *target* switches (Part C), not merely at the round the attack starts in the abstract — this is a "changed effective training mixture" effect, not a "damage" effect: nothing in the attacked run's post-round-20 trajectory resembles degradation.

**Hypothesis (not directly demonstrated, offered as the most parsimonious explanation consistent with the data):** in EXP_020, `client_004` — the largest partition (11,785/45,000 samples, 26.2% of all training data) — is excluded from the aggregate in 49 of 50 rounds, meaning its substantial local training signal is essentially never incorporated into the global model for the entire run. In EXP_022, the attack causes the rejection *target* to switch from `client_004` to `client_002` (a smaller partition, 7,202 samples, 16.0% of the total) starting exactly at round 20. For rounds 20–49 (60% of the run), EXP_022's aggregate therefore incorporates `client_004`'s larger, previously-excluded local signal instead of `client_002`'s smaller one. Since Multi-Krum's selected-client averaging is unweighted (equal 1/4 weight regardless of sample count — confirmed in `src/federated/aggregation/multikrum.py`), this is not a re-weighting effect; it is a *which distinct local update gets incorporated at all* effect. If `client_004`'s larger local dataset yields a more broadly representative (less overfit) local gradient direction than `client_002`'s smaller one, restoring `client_004` to the aggregate — a side effect of the attack changing *who* gets excluded, not of the attack's content being informative — would plausibly explain the improvement, entirely independent of the attack being "beneficial" in any causal sense. **This mechanism was not directly measured in this task** (e.g., no controlled ablation re-running clean Multi-Krum with `client_004` forcibly included, or with `client_002` forcibly excluded absent any attack, exists yet) and is reported as a hypothesis, not an established fact. See Part K for how this could be tested directly without introducing a new attack or seed.

## E. FedAvg (EXP_016) vs. Multi-Krum (EXP_022) Under the Same Attack

Same attack, same seed, same partition; only aggregation differs.

| Metric | EXP_016 (attacked FedAvg) | EXP_022 (attacked Multi-Krum) | Difference (022 − 016) |
|---|---|---|---|
| Final test loss | 1.4989 | 1.7933 | **+0.2943** (worse) |
| Test Top-1 | 56.50% | 54.43% | **−2.07 pp** |
| Test Top-5 | 87.34% | 82.59% | **−4.75 pp** |
| Macro precision | 64.54% | 59.09% | **−5.45 pp** |
| Macro recall | 56.51% | 54.43% | **−2.08 pp** |
| Macro F1 | 54.57% | 51.61% | **−2.95 pp** |
| Balanced accuracy | 56.51% | 54.43% | **−2.08 pp** |
| Best validation Top-1 | 57.48% (round 48) | 55.84% (round 48) | **−1.64 pp** |

**Despite perfectly and completely excluding the malicious client for the entire attacked window, Multi-Krum's final model is worse than FedAvg's on every metric.** This is the central comparison of this analysis. Two already-established facts jointly explain it, without needing to invoke any new mechanism:

1. **The attack barely hurts FedAvg to begin with.** Per `analysis/EXP_016_STAGE_B_GAUSSIAN_SEED123_VALIDATION.md` §9 (already-frozen result, not re-derived here): `EXP_016` vs. its own honest baseline `EXP_008` shows only a −0.33 pp Top-1 / −0.45 pp Top-5 difference — "essentially neutral," because FedAvg's sample-count weighting dilutes one attacker's contribution (`client_002` carries only 7,202/45,000 = 16.0% of the aggregation weight) enough that a `rho=1.0` magnitude perturbation does little visible damage. Checking `EXP_016`'s own round history directly (not inferred): validation Top-1 climbs smoothly through round 20 (43.72%→46.24%, a normal round-to-round gain) with no discontinuity or dip at onset.
2. **Multi-Krum's own clean-performance cost (already established: `analysis/EXP_020_CLEAN_MULTIKRUM_SEED123_ANALYSIS.md`, ≈−6.9 pp Top-1 relative to `EXP_008`) is much larger than the damage FedAvg experiences from this attack.** Even though Multi-Krum recovers roughly +4.5 pp of its own clean-run handicap once the rejection target switches at round 20 (Part D), that partial recovery is not large enough to overcome either its own remaining structural cost or FedAvg's own near-zero attack damage. `EXP_022`'s Top-1 (54.43%) remains below both `EXP_008` (56.83%, clean FedAvg) and `EXP_016` (56.50%, attacked FedAvg).

**There is no visible attack-onset degradation to "mitigate" under FedAvg in the first place** — so Multi-Krum's perfect rejection of `client_002` cannot be credited with preventing damage that was never substantial. Robustness benefit and net final-model quality are two different questions, and this run cleanly separates them.

## F. Complete 2×2 Factorial Interpretation

|  | CLEAN | GAUSSIAN ATTACK |
|---|---|---|
| **FedAvg** | EXP_008 | EXP_016 |
| **Multi-Krum** | EXP_020 | EXP_022 |

**Top-1:**

| Cell | Value |
|---|---|
| EXP_008 | 56.83% |
| EXP_016 | 56.50% |
| EXP_020 | 49.91% |
| EXP_022 | 54.43% |

1. Clean aggregation effect (EXP_020 − EXP_008): **−6.92 pp**
2. Attack effect under FedAvg (EXP_016 − EXP_008): **−0.33 pp**
3. Attack effect under Multi-Krum (EXP_022 − EXP_020): **+4.52 pp**
4. Aggregation effect under attack (EXP_022 − EXP_016): **−2.07 pp**

**Top-5:**

| Cell | Value |
|---|---|
| EXP_008 | 87.79% |
| EXP_016 | 87.34% |
| EXP_020 | 80.06% |
| EXP_022 | 82.59% |

1. Clean aggregation effect: **−7.73 pp**
2. Attack effect under FedAvg: **−0.45 pp**
3. Attack effect under Multi-Krum: **+2.53 pp**
4. Aggregation effect under attack: **−4.75 pp**

**Interpretation:** the attack's *effect* has opposite sign depending on aggregation strategy — mildly harmful under FedAvg, apparently beneficial under Multi-Krum (rows 2 vs. 3) — a genuine interaction between attack and aggregation strategy in this run, not an additive combination of a fixed "clean aggregation cost" and a fixed "attack damage." Per Part D, the most likely driver is the exclusion-target switch changing the effective training mixture, not any protective mechanism acting on the attack's content. **This is a single-seed, single-attack-strength observation; no claim of statistical significance or generalization beyond this exact configuration is made.**

## G. Multi-Krum Rejection vs. FedTrace Attribution

This section uses `EXP_016`'s already-frozen, already-validated FedTrace v3 result (`analysis/EXP_016_STAGE_B_GAUSSIAN_SEED123_VALIDATION.md`) as the FedTrace evidence for this attack scenario (same attack, same seed, same partition, same malicious client) — FedTrace v3 was **not** re-run on `EXP_022`'s own trace in this task (that would be a new, un-pre-registered analysis outside Stage B's frozen FedAvg-only validation matrix, and is flagged in Part J's limitations, not attempted here).

| | FedTrace v3 (from `EXP_016`) | Multi-Krum rejection proxy, applied to the **full 50-round run** | Multi-Krum rejection proxy, applied **only to the post-round-20 window** |
|---|---|---|---|
| Predicted suspicious client | `client_002` | **`client_004`** (confirmed round 4) | `client_002` |
| Onset estimate | 20 (error 0) | 2 | 20 (error 0) |
| Confirmation round | 22 | 4 | 22 |
| Detection delay | 2 rounds | — (16 rounds *before* the real attack starts) | 2 rounds |
| Honest false positives (whole run) | **0** | — | — |
| Attribution correctness | **Correct** | **Incorrect (wrong client)** | Correct, but only by construction (window chosen with foreknowledge of onset) |

**This is the central finding of this section.** Applying the pre-registered Multi-Krum rejection proxy's own rule ("predicted suspicious client = earliest persistently rejected client," `P_krum=3`) to the *entire* run — the only way it could honestly be applied in a real deployment, since the true onset round is exactly what an attribution method is supposed to discover, not something to condition on in advance — **misattributes `client_004`**, confirmed at round 4, sixteen rounds before the actual attack even begins. This is not a new phenomenon: it is `EXP_020`'s clean-run false positive (`analysis/EXP_020_CLEAN_MULTIKRUM_SEED123_ANALYSIS.md`), reproduced inside `EXP_022` itself (rounds 0–19 are identical to `EXP_020`, Part B), now sitting **earlier in the same run** than the true attacker's own confirmation (round 22). Under the proxy's own "earliest confirmed" tie-break rule, `client_004` wins.

**Only if the analysis window is artificially restricted to rounds ≥20** — i.e., only if the analyst already knows the attack started at round 20, which is precisely the fact an attribution method exists to discover — does the proxy correctly identify `client_002`, and its timing then exactly matches FedTrace's (confirmation round 22, onset error 0, delay 2 rounds in both). This is a genuine, fair, positive observation about Multi-Krum's *raw per-round geometric separation* once isolated from the pre-existing confound (Part H explains the likely mechanism); it is not evidence that the rejection proxy is a usable attribution method as-is.

**Answering the eight questions directly:**

1. **Does Multi-Krum identify `client_002` at attack onset?** Only if the pre-onset window is excluded from consideration; applied honestly to the whole run, no — it names `client_004` first.
2. **Does FedTrace identify `client_002`?** Yes, correctly, with zero honest false positives anywhere in the run.
3. **Onset/confirmation delays:** FedTrace — onset error 0, confirmed round 22, delay 2. Multi-Krum proxy (post-onset window only) — identical numbers. Multi-Krum proxy (full run) — no meaningful "delay" applies; it names the wrong client 16 rounds before the true onset.
4. **What happens in clean rounds?** FedTrace: nothing (0 honest FP, per `EXP_016`'s validated result, computed over the same partition/heterogeneity conditions). Multi-Krum's rejection proxy: `client_004` is persistently, confidently rejected — a standing false positive throughout the clean phase.
5. **Which method produces clean false positives?** Multi-Krum's rejection proxy (yes, `client_004`). FedTrace v3 (no, per `EXP_016`'s already-established zero-FP result).
6. **Is Multi-Krum rejection equivalent to attacker attribution?** No — demonstrated directly by the full-run misattribution above.
7. **Does Multi-Krum provide persistent temporal forensic evidence, or only per-round geometric exclusion?** Only the latter. `MultiKrum.aggregate()` (`src/federated/aggregation/multikrum.py`) is stateless across rounds — every call scores only that round's five submitted updates, with no memory of any prior round's scores or selections. The `P_krum=3` persistence rule is an *offline*, post-hoc construction applied afterward to the sequence of per-round CSV outputs; it is not a capability the aggregation strategy itself possesses. FedTrace v3, by contrast, has an explicit per-client calibration window (`T_ref=20`) and longitudinal, client-specific baselines built directly into its scoring formula (`research_plans/MULTIKRUM_EXTERNAL_BASELINE_IMPLEMENTATION_PLAN.md` §5's already-disclosed asymmetry: Multi-Krum needs a known/assumed `f`; FedTrace needs no such thing, and also needs no assumption that "whichever client is most different this round" is the same concept as "whichever client is behaving abnormally *relative to its own history*").
8. **Does FedTrace add information beyond aggregation robustness?** Yes, decisively, per this exact experiment: it avoids precisely the failure mode (mistaking a chronic, heterogeneity-driven outlier for a newly-anomalous one) that the repurposed rejection proxy falls into when applied the only way a real deployment could apply it.

**No FedTrace-superiority conclusion is forced beyond what this evidence shows.** This result does not claim Multi-Krum's aggregation-robustness function is defective (Part E already treats that separately and on its own terms), nor that FedTrace would necessarily succeed under every possible attack/partition configuration — only that, in this exact experiment, the conceptual distinction the project's design is built around — *"which updates should influence the global model this round"* is a different question from *"which client exhibits persistent, newly-emerging anomalous behavior, and when did it begin"* — is not an abstract distinction but an empirically demonstrated one: the two methods give different, and in the naive-application case, contradictory, answers on the identical data.

## H. Client Trace Signal Analysis

`client_round_trace.csv` for `EXP_022` (250 rows) was read in full. Two Gaussian-attack-irrelevant fields, `delta_attack_norm_ratio` and `realized_cosine_to_honest`, are blank for every row of every client, including `client_002`'s attacked rounds — **this is by design, not a data gap**: those two fields are populated only for `attack_type == "norm_preserving_directional"` runs (per `src/attacks/traceability.py`'s trace-row construction, confirmed already in `analysis/EXP_016_STAGE_B_GAUSSIAN_SEED123_VALIDATION.md`'s own handling of the identical field set), and `EXP_022` is a Gaussian-family run.

**`client_002`'s `update_l2_norm`:** pre-attack (R1–R19, excluding R0's universal initial-round outlier) ranges ≈5.19–5.44, essentially flat. At round 20 it jumps to **7.636** (a +2.2, +41% jump from R19's 5.44) and climbs steadily to **9.263** by round 49 — an immediate, sustained, monotonically-growing magnitude elevation, matching the signature already established for this attack family (`analysis/EXP_014_Gaussian_Attack_Traceability_Analysis.md`; `EXP_016`'s own §2).

**`client_002`'s `peer_mean_cosine_similarity`:** declines smoothly from ≈0.12 (R0) to ≈0.05 (R19) pre-attack — a gradual downward drift shared by every client as training converges (not attack-specific) — and continues in the same ≈0.03–0.06 band post-attack, with **no sharp discontinuity at round 20**. This is consistent with the naive peer-cosine baseline detector's `naive_flag` staying `False` for `client_002` in all 50 rows — **the naive detector never flags the actual attacker at all**, in this run exactly as in `EXP_016`. (A separate, incidental observation: the naive detector's `naive_flag` *does* turn `True` for `client_004` starting round 23 — a third, independent false positive alongside the Multi-Krum rejection proxy's `client_004` false positive — corroborating, from an entirely different signal family, that this specific client's naturally large/distinct honest update is a recurring source of spurious flags across more than one simple detector, not an artifact unique to Multi-Krum.)

**`client_002`'s `self_prev_round_cosine`:** pre-attack ≈0.24–0.35 (declining, like the other clients). At round 20 it drops to **0.168** and stays in a lower ≈0.14–0.19 band for the rest of the run — a partial, not total, degradation. This matches `EXP_016`'s own finding that `gaussian_update_noise` **adds** independent noise to the honest direction rather than **replacing** it, so temporal self-consistency degrades but does not collapse to near-zero the way the norm-preserving directional attack does.

**Why can Multi-Krum's raw distance metric separate `client_002` geometrically after round 20, when the naive peer-cosine detector cannot?** The most direct, data-supported explanation: Multi-Krum's Krum score is computed from **raw squared Euclidean distance** (`multikrum.py::_pairwise_squared_distances`), which is sensitive to vector *magnitude* as well as direction, whereas peer-mean cosine similarity is direction-only and magnitude-invariant by construction. `rho=1.0` means the injected Gaussian noise's own L2 norm is set equal to the honest update's norm — a substantial, deliberately magnitude-scaled perturbation — and that magnitude increase shows up directly and strongly in Euclidean distance (client_002's own `update_l2_norm` overtaking even `client_004`'s naturally-elevated norm by round 20: 7.64 vs. 7.50, and pulling further ahead every round after — 9.26 vs. 7.40 by round 49) while barely registering in a normalized cosine-similarity signal. This is offered as a plausible, data-consistent mechanism for why a raw-distance-based robust-aggregation rule can succeed at excluding a magnitude-revealing attacker precisely where a normalized-similarity-based naive detector fails — **noted here as a candidate explanation worth further investigation, not established beyond what these two signal comparisons directly show.**

## I. Partition-Size / Fairness Analysis

`client_partition_summary.csv` (identical across `EXP_008`/`EXP_016`/`EXP_020`/`EXP_022`, confirming the same partition realization throughout): `client_000`=6,024, `client_002`=7,202, `client_001`=9,628, `client_003`=10,361, `client_004`=11,785.

- **`client_004` is the largest client:** confirmed (11,785, largest of the five).
- **`client_002`'s sample-size rank:** 2nd-smallest of five (7,202) — neither the largest nor the smallest client.
- **Relationship between sample-size rank and clean Krum rank:** during R0–R19, the correspondence is **exact and monotonic** — ascending sample count (`client_000` < `client_002` < `client_001` < `client_003` < `client_004`) matches ascending Krum rank (1→5) exactly, identically to `EXP_020`'s already-established finding (§Part 4 of `analysis/EXP_020_CLEAN_MULTIKRUM_SEED123_ANALYSIS.md`), since these rounds are byte-identical between the two runs.
- **Does this clean association remain visible during R0–19 of EXP_022?** Yes, exactly (Part B, Part C).
- **How does the attack override that relationship after R20?** Completely: `client_002` — despite being a *smaller*-than-median partition, and despite occupying the *2nd-most-trusted* Krum rank throughout the entire clean phase — becomes the single worst-ranked client the instant the attack activates, while `client_004` (the *largest* partition, previously always worst-ranked) is simultaneously promoted out of rejection. The injected attack signal is strong enough to fully supersede the partition-size-driven ordering that governs the clean phase, in a single round, with no gradual transition.

As in `EXP_020`'s analysis, the underlying **mechanism** connecting partition size to clean-phase Krum rank (hypothesized there as unequal local SGD step counts under fixed `local_epochs=1` across differently-sized partitions) is used here as a **candidate explanation**, not a directly measured or causally established fact — no additional measurement (e.g. raw per-client update norms independent of Krum scoring) was performed in this task to confirm it beyond what was already noted in `EXP_020`'s own analysis.

## J. Paper-Level Interpretation

1. **Multi-Krum attack mitigation:** **inconclusive/negative as net protection.** Multi-Krum completely and immediately excludes the malicious client from round 20 onward (confirmed fact), and the attacked run outperforms the clean Multi-Krum control (confirmed fact, Part D) — but the attacked run still underperforms *both* FedAvg baselines, clean and attacked (confirmed fact, Part E/F). There is no attack-onset damage under FedAvg for Multi-Krum to have prevented in the first place (confirmed, citing `EXP_016`'s already-frozen result). **Single-run observation**, one seed, one attack strength.
2. **Multi-Krum clean-performance cost:** **confirmed, substantial, and replicated** — already established across two seeds in `analysis/MULTIKRUM_CLEAN_CONTROL_CROSS_SEED_SYNTHESIS.md` (~6.8–8.9 pp Top-1/Top-5 cost) and directly relevant to why Multi-Krum cannot outperform FedAvg here even with perfect exclusion. **Replicated finding.**
3. **Multi-Krum clean false-positive behavior:** **confirmed and reproduced inside this very run.** `client_004`'s clean-phase persistent rejection (rounds 2–19, confirmed round 4 under `P_krum=3`) is identical to `EXP_020`'s finding, now shown to *outlast* into an attacked run and to *precede* the true attacker's own confirmation when the proxy is applied naively (Part G). **Replicated finding, now with a new, single-run consequence (misattribution) not previously observable from clean-only data.**
4. **Malicious-client rejection accuracy after attack:** **confirmed fact, single run** — 30/30 (100%) post-attack rejection rate, immediate onset (round 20, zero lag), stable for the entire post-attack window. This is a strong, clean, unambiguous result for the narrow claim "does Multi-Krum's raw per-round score separate the attacker from the honest clients once it is active" — it does, decisively.
5. **FedTrace attribution value beyond robust aggregation:** **confirmed, single run, directly demonstrated (Part G).** FedTrace correctly attributes `client_002` with zero honest false positives across the whole run; the Multi-Krum rejection proxy, applied the only way a real (non-oracle) deployment could apply it, misattributes `client_004`. The conceptual distinction ("which updates should count this round" vs. "which client has a persistently changed baseline, and since when") is not asserted abstractly here — it is the direct, sufficient explanation for why the two methods disagree on this exact data.
6. **Fairness/heterogeneity implication:** **confirmed and replicated** — the same client (or, across seeds, the same-*role* client: the largest partition) that `EXP_020`/`EXP_021` already showed is falsely, persistently flagged under clean conditions is shown here to remain a live confound *inside an attacked run*, not merely a clean-control artifact that disappears once a real attacker is present.
7. **Does this strengthen the paper's central contribution?** **Yes, on the evidence available.** The clean-control findings (established previously) show the fairness risk exists; this run shows the fairness risk **persists specifically at the moment it would matter most for a naive attribution use of Multi-Krum** — inside the same run as a genuine attacker, competing directly for "earliest confirmed" status, and losing that competition to the true attacker only by 18 rounds' margin (round 4 vs. round 22) in this one configuration. This is exactly the kind of concrete, quantified evidence a "why not just use Krum's rejection pattern instead of FedTrace" objection would need to be answered with.

**Classification of findings:**
- **Confirmed facts (this run, directly measured):** pairing validity (Part A); diagnostics integrity and the exact rejection counts/streaks/regime-switch timing (Part B/C); the exact metric deltas in Parts D/E/F; the exact FedTrace-vs-proxy attribution disagreement (Part G); the trace-signal values in Part H; the partition/rank correspondence in Part I.
- **Replicated findings (consistent with prior runs, now reproduced):** Multi-Krum's clean-performance cost and clean false-positive behavior (originally `EXP_020`/`EXP_021`, now visible again inside `EXP_022`'s own R0–19).
- **Single-run observations (not yet replicated at a second seed or attack instance):** the attacked-vs-clean-Multi-Krum performance improvement (Part D); the specific magnitude of the FedAvg-vs-Multi-Krum attacked gap (Part E); the exact regime-switch shape (Part C).
- **Hypotheses requiring additional evidence:** the "exclusion-target-switch changes effective training mixture" explanation for Part D's counter-intuitive result (Part D, Part K); the "unequal local SGD steps under fixed `local_epochs`" mechanism for the partition-size/Krum-rank association (Part I, inherited from `EXP_020`); the "raw Euclidean distance vs. normalized cosine similarity" explanation for why Multi-Krum separates the attacker geometrically where the naive detector does not (Part H).

## K. Next-Experiment Gate

**Candidates considered:**
- **A. Seed-2026 Gaussian Multi-Krum replication** — would test whether Part D's counter-intuitive "attacked outperforms clean" result and Part E's "Multi-Krum underperforms FedAvg despite perfect exclusion" result replicate at the second already-validated seed, using existing, already-frozen attack parameters — no new seed, attack family, or parameter is introduced, matching the frozen plan's scope discipline exactly.
- **B. Seed-123 directional Multi-Krum experiment** — would test the norm-preserving directional attack family instead, a different open question (whether Multi-Krum's Euclidean-distance metric, shown here to respond strongly to a magnitude-revealing attack, responds at all to a norm-*preserving* one).
- **C. Additional analysis before another GPU run** — e.g., directly measuring per-client raw update norms/SGD-step counts to test the "unequal local steps" hypothesis (Part I) without any new training run.
- **D. Stop this branch** — not justified; the evidence so far (Parts D, E, G) raises specific, testable, single-run questions that more data would directly resolve, and the frozen plan's own minimal matrix (`research_plans/MULTIKRUM_EXTERNAL_BASELINE_IMPLEMENTATION_PLAN.md` §8) already calls for both Gaussian seeds before moving to the directional family.

**Recommendation: A — seed-2026 Gaussian Multi-Krum replication.** This is the highest-information-gain next step: it directly tests whether *every* finding in this document (the regime switch, the counter-intuitive clean-vs-attacked comparison, the FedAvg-vs-Multi-Krum attacked gap, and — most importantly for the paper's central contribution — the rejection-proxy misattribution under naive full-run application) is specific to seed 123's particular partition draw or is a more general property of this attack/aggregation/heterogeneity combination. It follows the frozen plan's own pre-registered sequence (Gaussian×123 before Gaussian×2026, before either directional seed) and introduces no new seed, attack family, or `f`/`m` value. **Its config is not created in this task.**

---

## Final Report

1. **Files created:** `analysis/EXP_022_MULTIKRUM_GAUSSIAN_SEED123_ANALYSIS.md`, `analysis/ROBUST_AGGREGATION_VS_TRACEABILITY_SYNTHESIS.md`.
2. **Pairing verdict:** valid — EXP_022 differs from EXP_016 scientifically only by `aggregation: fedavg→multikrum`; EXP_022 shares an identical partition realization with EXP_020 and EXP_016 (confirmed both by sample counts and by byte-identical pre-attack diagnostics/trajectories).
3. **Diagnostics-integrity verdict:** pass — 250/250 rows, 50 rounds × 5 clients, ranks 1–5 exactly once per round, exactly 4 selected/1 rejected every round, no missing or malformed rows; rounds 0–19 independently verified identical to EXP_020.
4. **Rejection counts pre/post attack:** `client_004`=19 pre-attack/0 post; `client_003`=1 pre/0 post; `client_002`=0 pre/**30 post**; `client_000`/`client_001`=0/0.
5. **Malicious-client rejection result:** `client_002` rejected in 30/30 (100%) post-attack rounds, rank 5 every time, immediately from round 20 onward.
6. **Exact regime-switch result:** instantaneous, zero-lag switch at round 20 exactly — `client_004` (rejected every pre-attack round but R1) is immediately selected, `client_002` (rank 2 every pre-attack round) is immediately rejected, in the same round the attack activates.
7. **Clean-vs-attacked Multi-Krum metric differences:** EXP_022 outperforms EXP_020 on every metric (+4.52 pp Top-1, +2.53 pp Top-5, +4.40 pp best-validation) — reported exactly as observed; explained as a hypothesis (exclusion-target switch restoring the largest client's training signal), not a causal "attack improves learning" claim.
8. **FedAvg-vs-Multi-Krum attacked differences:** EXP_022 underperforms EXP_016 on every metric (−2.07 pp Top-1, −4.75 pp Top-5, −1.64 pp best-validation) despite perfect malicious-client exclusion, because the Gaussian attack barely damages FedAvg to begin with and Multi-Krum's own clean-performance cost dominates.
9. **2×2 interpretation:** clean aggregation cost −6.92 pp Top-1; attack effect under FedAvg −0.33 pp; attack effect under Multi-Krum +4.52 pp; aggregation effect under attack −2.07 pp — a genuine attack×aggregation interaction, single-seed, not claimed as statistically significant.
10. **FedTrace-vs-Multi-Krum interpretation:** FedTrace correctly attributes `client_002` (onset error 0, delay 2, zero honest FP); the Multi-Krum rejection proxy, applied honestly to the whole run, misattributes `client_004` (confirmed round 4, sixteen rounds before the true attacker's own confirmation at round 22) — a direct, empirical demonstration that per-round exclusion is not equivalent to persistent-behavior attribution.
11. **Fairness/partition finding:** the clean-phase Krum rank ordering matches the sample-count ordering exactly (ascending); the attack fully overrides this ordering in a single round; the mechanism connecting partition size to clean rank remains a hypothesis, not directly measured in this task.
12. **Paper-level contribution verdict:** the evidence strengthens the project's central contribution — it shows the fairness risk pre-registered and confirmed in the clean controls is not merely a clean-run artifact but persists, and actively competes with the true attacker for "earliest confirmed" status, inside a genuinely attacked run.
13. **Limitations:** single seed (123), single attack strength (rho=1.0), FedTrace v3 was not re-run on EXP_022's own trace (EXP_016's frozen result was used as the representative FedTrace evidence, per this task's own framing of EXP_016 as "the paired Gaussian attack experiment"); the exclusion-target-switch and partition-size mechanisms are hypotheses, not directly measured; no statistical significance is claimed from n=1.
14. **Exact next-experiment recommendation:** seed-2026 Gaussian Multi-Krum replication (the Multi-Krum analog of EXP_017, paired with EXP_009/EXP_017/EXP_021) — not created in this task.
15. **Confirmation no implementation/config/experiment was changed:** confirmed — `src/federated/aggregation/multikrum.py`, `src/analysis/fedtrace_detector.py`, `src/attacks/traceability.py`, all configs, all tests, `README.md`, and Research_Database were untouched; only pre-existing result artifacts and prior analysis/plan documents were read; no experiment was run; no shell/PowerShell/Bash command was executed.
