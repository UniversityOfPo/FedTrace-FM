# Multi-Krum External Baseline: Implementation Plan

**Status:** Planning document only. No code, config, test, README, or Research Database file was created or modified in producing it. No experiment was run. FedTrace v3 remains frozen and untouched. This document fully specifies the Multi-Krum baseline so that a future task can implement it without further open design decisions; it does not implement anything itself.

---

## 1. Scientific Role

**Multi-Krum will serve primarily as an external, Byzantine-robust *aggregation* baseline.**

**Primary question:** does a standard, literature-established Byzantine-robust aggregation rule reduce global-model damage under the exact same attack scenarios (`gaussian_update_noise`, `norm_preserving_directional`; `client_002`; round-20 onset; seeds 123/2026) already used to independently validate FedTrace v3?

**Secondary, explicitly exploratory question:** can Multi-Krum's per-round selected/rejected-client pattern — a byproduct of its aggregation decision, not something it was designed to produce — offer any useful client-level suspiciousness signal?

**The secondary proxy is never treated as equivalent to FedTrace attribution.** FedTrace v3 was purpose-built, frozen, and independently validated specifically for the "which client, and when" question (`analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md`). Multi-Krum was not. Every reference to its exclusion pattern in this document and any future one built from it is prefixed **"Multi-Krum rejection proxy"** — never "Multi-Krum malicious-client detector" or any phrasing implying it is a native capability.

## 2. Original Method Definition

Following Blanchard, El Mhamdi, Guerraoui, Stainer, *"Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent,"* NeurIPS 2017 (verified in `research_plans/FEDTRACE_EXTERNAL_BASELINE_SELECTION_PLAN.md` §5), exactly, with no invented variant:

**Krum score, per round, per client `i`:**
```
d(i, j) = || u_i - u_j ||_2^2          (squared Euclidean distance between client i's and j's update vectors)

score(i) = sum over the (n - f - 2) smallest values of { d(i, j) : j != i }
```
i.e., for each client, sum the squared distances to its `n − f − 2` *nearest* peers (excluding itself), discarding the `f + 1` most-distant peers from that sum. This is the exact formulation that gives Krum its resilience guarantee: it never contaminates a client's score with distances to more than `f` potential Byzantine outliers plus itself.

**Krum (single-update) selection:** the client with the lowest `score(i)` is chosen; its update alone becomes that round's aggregate.

**Multi-Krum:** compute `score(i)` for all `n` clients exactly as above, rank ascending, and select the `m` lowest-scoring clients. Their updates are combined (§4) into that round's aggregate — Multi-Krum, not single-update Krum, is what this plan specifies for implementation, because it is the standard choice when more than one honest update is available to average (this project's `n=5` easily supports it) and because it produces a *set* of selected/excluded clients each round, which is what the secondary rejection-proxy analysis (§11) needs.

**Parameters, fixed for this project, not tuned per attack family or seed:**
```
n = 5   (num_clients = clients_per_round, this project's fixed configuration)
f = 1   (assumed upper bound on Byzantine clients -- matches this project's own,
         already-established single-attacker threat model, PHASE5_Malicious_Client_Traceability_Plan.md §3;
         not a free parameter to be swept, per this task's explicit restriction)
```

**Feasibility check:** Krum's resilience proof requires `n ≥ 2f + 3`. At `n=5, f=1`: `2(1)+3 = 5`, so `5 ≥ 5` — satisfied, but exactly at the boundary, not with margin. This is stated plainly, not hidden: this project's client count is the smallest value at which `f=1` Krum resilience is even theoretically claimed to hold.

**Score computation at these parameters:** `n − f − 2 = 5 − 1 − 2 = 2` — each client's score is the sum of squared distances to its **2 nearest** peers among the other 4.

**Multi-Krum selection count `m`: fixed at `m = n − f = 4`.** Justification: `m = n − f` is the standard, parameter-minimal choice in the literature — it directly encodes "assume at most `f` Byzantine clients exist; average everyone except the `f` most suspicious" and introduces no additional free parameter beyond the already-fixed `f`. **This single value of `m` is used for every run in this plan — Gaussian and directional, both seeds, clean and attacked — with no per-scenario adjustment**, per this task's explicit instruction.

## 3. Model Update Representation

**Exactly one representation is used: the flattened delta over trainable parameters only** — `u_i = flatten(client_i_model_state[k] - global_state[k] for k in trainable_keys)`, sorted-key-concatenated exactly as `src/attacks/traceability.py::_flatten_sorted`/`flattened_l2_norm` already do, and exactly matching `trainable_keys = frozenset(name for name, _ in model_fn().named_parameters())` as already computed in `src/experiments/run_federated_cifar100.py`.

**This is the scientifically correct choice, not an arbitrary one**, because it is exactly the vector every attack in this project's series already manipulates: both `apply_gaussian_update_noise` and `apply_norm_preserving_directional_attack` operate on `honest_model_state[k] - global_state[k]` for `k in trainable_keys`, and FedTrace v3's own `z_m` is computed on `update_l2_norm`, itself `flattened_l2_norm` over the same `trainable_keys`-restricted delta. Using the identical vector definition means Multi-Krum's Euclidean distance and FedTrace's magnitude/temporal signals are being computed over the same underlying "update," making any later comparison between them meaningful rather than an artifact of differing definitions of "update."

**Explicitly excluded from the Euclidean distance computation:** BatchNorm `running_mean`, `running_var` (floating buffers, but not trainable parameters), `num_batches_tracked` (integer buffer), and any other non-trainable buffer. None of these are included in `u_i` — silently folding a non-trainable buffer into a Euclidean-distance-based robustness score would let ordinary, honest, architecture-level bookkeeping variance (e.g. `num_batches_tracked` legitimately differing across clients under Non-IID partition sizes, already documented and deliberately handled specially by `FedAvg._max_across_clients`) contaminate a score that is supposed to measure malicious *training* deviation, not accounting artifacts.

**These excluded buffers still need a value in the final aggregated state dict** (§4 addresses how).

## 4. Weighting Question

**Original Krum/Multi-Krum aggregation is unweighted** — it averages the `m` selected clients' full parameter tensors with equal weight `1/m` each, with no dependence on sample count. This plan preserves that convention exactly for the trainable-parameter aggregation, **not** modified to use FedAvg's sample-count weighting, because doing so would silently turn "Multi-Krum" into an unpublished hybrid, undermining the entire point of comparing against an externally-defined, literature-faithful method.

**This is explicitly documented as a difference from FedAvg**, and is expected to matter under this project's already-established strong client-size imbalance (`analysis/EXP_009_Three_Seed_Baseline_Analysis.md` §7: sample-count coefficient of variation up to ≈30.9% across seeds) — an unweighted average could over-represent a small client and under-represent a large one relative to FedAvg's sample-proportional rule, entirely independent of any attack. This is treated as a real, reportable property of the external baseline (§6), not a flaw to correct.

**Non-trainable floating buffers** (BatchNorm `running_mean`/`running_var`): aggregated by unweighted-averaging the **same `m` selected clients'** values — reusing the trainable-parameter selection decision, not a second, separate distance computation over buffers (which the original method never defines at all). This is the minimal, explicitly-flagged extension needed to produce a complete state dict for an architecture (ResNet-18/BatchNorm) the 2017 paper never addressed.

**Non-floating buffers** (`.num_batches_tracked` and any other integer/boolean buffer): reuse FedAvg's own already-implemented, already-justified rules unchanged — `_max_across_clients` for `.num_batches_tracked`-suffixed keys, `_require_identical` for everything else — applied across **all** participating clients' submitted values (not only the `m` selected ones), since these rules already have nothing to do with which clients Multi-Krum's robustness logic trusts; they are pure accounting rules, reused via a shared helper rather than reimplemented, to avoid two subtly-different copies of the same non-trivial buffer-reconciliation logic.

## 5. Known-Attacker-Count Assumption

**Multi-Krum requires `f`, a known or assumed upper bound on the number of Byzantine clients, as an explicit input parameter.** FedTrace v3 requires no such input at any point — its calibration and persistence rules operate purely on each client's own historical/cross-sectional signal statistics, with no assumption about how many clients (if any) are malicious.

**This is an important fairness caveat, stated prominently, not hidden:** Multi-Krum is given information (`f=1`, matching the exact single-attacker ground truth this project's own benchmark happens to use) that FedTrace itself is never given and does not need. Any future comparison that reads Multi-Krum's mitigation performance as evidence of general superiority or inferiority relative to FedTrace without noting this asymmetry would be scientifically misleading. This caveat must be repeated wherever Table A/B (§12) results are eventually reported.

## 6. Non-IID Fairness Risk

**The key risk, pre-registered now, before any run:** under this project's already-established strong Non-IID condition (Dirichlet `α=0.1`), *honest* clients' updates are themselves naturally far apart in Euclidean update-space — this project's own three-seed baseline (`analysis/EXP_009_Three_Seed_Baseline_Analysis.md`) already demonstrates substantial, seed-dependent heterogeneity in both partition composition and training dynamics. Krum/Multi-Krum's entire mechanism assumes that *honest* updates cluster together and a *Byzantine* update is the outlier; if honest heterogeneity itself produces updates as far apart as (or farther apart than) the actual attacker's update, Multi-Krum may **reject a genuinely honest client instead of, or in addition to, the true attacker** — a direct conflict between distance-based Byzantine robustness and this project's own already-documented statistical heterogeneity.

**This is pre-registered as one of the scientifically important comparisons this baseline study exists to make, and any such behavior — Multi-Krum rejecting an honest client, failing to reject the true attacker, or both — will be reported exactly as observed.** It will not be tuned away (e.g. by adjusting `f`, `m`, or the distance metric after seeing an unfavorable result), consistent with every prior Stage A/B document's discipline in this project.

## 7. Clean Baseline Requirement

**Yes — clean (honest-only, no attack) Multi-Krum control runs are scientifically necessary, at both seeds.** Krum is already known in the literature to impose an accuracy cost even with zero Byzantine clients present, because it structurally discards `f` clients' updates every single round regardless of whether an attack is occurring — under this project's own strong Non-IID condition, discarding one client's update each round also discards that round's only representation of whatever local class distribution that client alone holds. Without a clean control, any accuracy difference observed on an attacked run cannot be separated into "cost of robustness" versus "residual damage the attack still caused" — exactly the "clean-performance cost" metric already pre-registered in §10 of `research_plans/FEDTRACE_EXTERNAL_BASELINE_SELECTION_PLAN.md`.

**Both seed 123 and seed 2026 clean Multi-Krum controls are required** (paired against the same already-existing `EXP_008`/`EXP_009` honest FedAvg baselines this project already has), for the same paired-seed discipline already used throughout this project — a single clean-control seed would not distinguish a real, general clean-performance cost from one seed's particular partition realization. **This adds exactly 2 runs, kept to the minimum needed to make the comparison meaningful** — not a broader clean-control sweep.

## 8. Attacked Experiment Matrix

The minimum fair matrix, reusing the already-frozen attack scenarios exactly, with only the aggregation strategy changed:

| Attack family | Seed | Attack parameters (identical to `EXP_016`–`EXP_019`) | Paired honest FedAvg | Paired clean Multi-Krum control |
|---|---|---|---|---|
| Gaussian update-noise | 123 | `client_002`, round 20, `ρ=1.0` | `EXP_008` | Clean Multi-Krum, seed 123 |
| Gaussian update-noise | 2026 | `client_002`, round 20, `ρ=1.0` | `EXP_009` | Clean Multi-Krum, seed 2026 |
| Norm-preserving directional | 123 | `client_002`, round 20, `target_cosine=0.0` | `EXP_008` | Clean Multi-Krum, seed 123 |
| Norm-preserving directional | 2026 | `client_002`, round 20, `target_cosine=0.0` | `EXP_009` | Clean Multi-Krum, seed 2026 |

**No new attack is introduced.** All attack settings (attacker identity, onset round, severity) remain byte-for-byte identical to `EXP_016`–`EXP_019`'s already-frozen configuration — the only difference from those four runs is `aggregation: fedavg → multikrum`. **No configs are created in this task** (per explicit restriction) — this table specifies what a future task would create, six new configs total (4 attacked + 2 clean), paired against the exact same seed-123/seed-2026 honest FedAvg baselines already on disk.

## 9. Robustness Metrics (Native Purpose)

Pre-registered for every Multi-Krum run (clean and attacked): final test loss, Top-1, Top-5, best validation Top-1, degradation relative to the paired honest FedAvg baseline (same computation already used for `EXP_016`–`EXP_019` vs. `EXP_008`/`EXP_009`, `analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md` §13), improvement/recovery relative to the corresponding *attacked* FedAvg run (i.e. does attacked Multi-Krum close some, all, or none of the Top-1 gap `EXP_016`–`EXP_019` each showed relative to their honest baseline), and clean-performance cost (the clean Multi-Krum control's own Top-1 gap relative to its paired honest FedAvg baseline, isolating the cost of robust aggregation absent any attack, §7).

## 10. Selection/Rejection Diagnostics

**Required logging, per round, for every run (clean and attacked):** each client's Krum score, the set of selected clients (the `m=4` lowest-scoring), the set of rejected clients (the remaining `n − m = 1`), and each client's rank by score (1 = lowest/most-trusted, 5 = highest/most-suspicious). This is exactly analogous in spirit to `client_round_trace.csv`'s existing per-round diagnostic logging, but as a new, separate artifact (§13) — not merged into or overwriting FedTrace's own trace schema.

**No ground truth is used in selection at any point** — the score and selection are computed purely from the `n=5` clients' own submitted update vectors that round, exactly as the original algorithm specifies; `attack_start_round`/`malicious_client_id` are never read by the selection logic itself, only afterward, by whichever analysis task eventually evaluates the results against ground truth (mirroring FedTrace v3's own, already-established discipline).

## 11. Repurposed Attribution Proxy

**Definition, pre-registered now, not tuned after seeing any attacked result:**

- **"Multi-Krum rejection proxy — first anomalous round":** the first round a given client is rejected (excluded from the `m=4` selected clients).
- **"Multi-Krum rejection proxy — persistent rejection":** a client is rejected in `P_krum = 3` consecutive rounds — the same persistence length FedTrace v3 uses (`P=3`), chosen for direct comparability, not independently re-derived or tuned for Multi-Krum's own score distribution.
- **"Multi-Krum rejection proxy — onset estimate":** the first round of that confirmed 3-round persistent-rejection run (mirroring FedTrace v3's own onset-estimate definition exactly, for comparability).
- **"Multi-Krum rejection proxy — predicted client":** the earliest-persistently-rejected client, if any; ties broken by rejection frequency (most-often-rejected across the whole run), then by `client_id` ascending — the same tie-break *shape* as FedTrace v3's, adapted to rejection frequency in place of mean anomaly score since Multi-Krum produces no continuous score comparable to `FedTraceV3Score`.

**This proxy is defined completely now, before any attacked run exists, specifically so it cannot be adjusted after seeing whether it happens to track the true attacker.** Every occurrence of it in any future document must use the exact phrase **"Multi-Krum rejection proxy"**, never **"Multi-Krum malicious-client detector"** or equivalent — reinforcing §1's framing at the point where it would be easiest to blur.

## 12. FedTrace Comparison Boundary

Two separate tables, never merged into one ranking:

**Table A — Attribution / traceability** (FedTrace v3 vs. the Multi-Krum rejection proxy, §11): predicted suspicious client, onset estimate (if the proxy confirms at all), detection delay, honest persistent FP (a rejected honest client counts here exactly as an honest FedTrace false positive would). **A strong caveat must accompany this table wherever it appears**: only FedTrace v3 was purpose-built, frozen, and independently validated for this task; Multi-Krum's entry in this table is a repurposed byproduct of a method designed for a different question (§1, §5), evaluated here only because it is the best available proxy, not because it is a fair native competitor.

**Table B — Robust aggregation** (attacked FedAvg vs. Multi-Krum, §9): test loss, Top-1, Top-5, clean-performance cost, recovery of attacked performance. This table never includes any FedTrace metric — FedTrace has no aggregation-side effect to measure (§8 of `research_plans/FEDTRACE_EXTERNAL_BASELINE_SELECTION_PLAN.md`'s own invalid-comparison warning applies directly here).

## 13. Implementation Architecture

**Multi-Krum plugs into the existing `AggregationStrategy` interface (`src/federated/aggregation/base.py`) directly — this is architecturally valid and requires no change to that interface.** `AggregationStrategy.aggregate(global_state, updates)` already receives the full raw `ClientUpdate.model_state` for every participating client (confirmed by direct inspection of `src/federated/aggregation/fedavg.py`, which already operates on exactly this data), which is all Multi-Krum's distance computation needs.

**Files expected to change (in a future implementation task, not this one):**
- **New:** `src/federated/aggregation/multikrum.py` — the `MultiKrum(AggregationStrategy)` class implementing §2–§4 exactly.
- **Modified:** `src/experiments/run_federated_cifar100.py` — extend `_SUPPORTED_AGGREGATIONS` (currently `("fedavg",)`, line 81) to include `"multikrum"`, and add a dispatch (currently `strategy = FedAvg()` unconditionally, line 748) selecting `MultiKrum()` when `config.aggregation == "multikrum"`. `f` is fixed at `1` in code (§2), not exposed as a new YAML field, to foreclose an accidental parameter sweep.
- **New:** `tests/test_multikrum_aggregation.py`, parallel in style and scope to the existing `tests/test_aggregation.py` (§14).
- **New (a future task, not this one):** six experiment configs (§8), following the exact naming/pairing convention already established for `EXP_016`–`EXP_019`.

**Explicitly untouched by this addition, confirmed by design:** `src/attacks/traceability.py` (attack injection is orthogonal to aggregation choice — `update_transform_callback` fires before the server ever sees an update, regardless of which `AggregationStrategy` is later applied to it), `src/federated/runner.py`, `src/federated/server.py`, `src/federated/client.py`, `src/analysis/fedtrace_detector.py`, and every existing config, test, and analysis document. FedTrace v3 remains frozen throughout.

## 14. Test Plan (Pre-Registered, Not Implemented Now)

- **Pairwise distance correctness** — hand-computable tiny vectors (e.g. 5 clients, 2-dimensional trainable-parameter deltas with exactly known squared distances), asserting the computed distance matrix matches hand arithmetic exactly.
- **Krum-score correctness** — a hand-computable `n=5, f=1` example (score = sum of the 2 smallest of each client's 4 pairwise distances), asserting exact numeric agreement.
- **`f` validation** — rejects a non-integer, negative, or zero-when-required `f`.
- **`n`/`f` feasibility validation** — rejects a client set violating `n ≥ 2f+3` (e.g. fewer than 5 participating clients at `f=1`) with a clear error, never silently proceeding with an infeasible configuration.
- **Selection-count (`m`) validation** — confirms `m = n − f` is derived, not independently settable to a value that would silently change the resilience assumption.
- **Deterministic tie-breaking** — two clients with numerically identical scores (constructed synthetically) resolve to the same selection every call, via a defined, stable rule (client_id ascending), not iteration-order-dependent.
- **Malicious-outlier rejection in a toy example** — 4 near-identical synthetic "honest" update vectors plus 1 clearly-outlying vector; asserts the outlier is excluded from the selected `m=4`.
- **Honest-update selection in a toy example** — the same setup; asserts all 4 near-identical vectors are the ones selected and the aggregated output equals their unweighted mean exactly.
- **Buffer exclusion** — a synthetic model with a trainable float parameter, a non-trainable float buffer (e.g. a `running_mean`-style tensor), and a `.num_batches_tracked`-style integer buffer; asserts only the trainable parameter affects the computed Krum scores/selection, while all three keys still appear, correctly aggregated per §4, in the final output.
- **Shape/key-mismatch handling** — reuses `validate_common_update_structure` (already shared with `FedAvg`); asserts a client update with an incompatible key set or tensor shape raises the same clear error `FedAvg` already raises, not a `MultiKrum`-specific divergent error.
- **Backward compatibility with FedAvg** — confirms `FedAvg`'s own behavior and its existing test suite (`tests/test_aggregation.py`) are unaffected by `MultiKrum`'s addition (no shared mutable state, no import-order dependency).
- **Unchanged existing tests** — explicit confirmation that `tests/test_aggregation.py` requires zero modification, since `MultiKrum` is purely additive to the aggregation plugin set.

**No implementation, and no test execution, occurs in this task.**

## 15. Implementation Failure vs. Scientific Failure

**Implementation failure:** the Krum-score formula, neighbor count, selection count, tie-breaking, buffer handling, or logging is computed incorrectly relative to §2–§4/§10's specification — caught by §14's unit tests before any GPU run, or discovered via an implementation-correctness checkpoint mirroring `research_plans/FEDTRACE_STAGE_B_INDEPENDENT_VALIDATION_PLAN.md` §16's discipline (verify against a known-correct toy case before trusting a full CIFAR-100 run's numbers).

**Scientific failure:** a *correctly implemented* Multi-Krum performs poorly — rejects an honest client, fails to reject the true attacker, imposes a large clean-performance cost, or fails to meaningfully recover attacked accuracy — because this project's strong Non-IID heterogeneity or one of the two attack families' specific geometry genuinely violates Multi-Krum's practical operating assumptions (§6). **A scientific failure is evidence to report, exactly as observed, in both Table A and Table B (§12) — never something to fix by adjusting `f`, `m`, the distance metric, or any other parameter after seeing an unfavorable result.** This mirrors, exactly, the "do not tune away a discovered weakness" discipline already applied to every FedTrace v1/v2/v3 finding in this project.

## 16. Success Criteria

**Multi-Krum is not required to beat FedTrace v3 on attribution** — it was never designed for that task (§1, §5), and requiring it to do so would be exactly the kind of invalid comparison §8 of `research_plans/FEDTRACE_EXTERNAL_BASELINE_SELECTION_PLAN.md` already rejects.

**For robust aggregation specifically, meaningful success (evaluated per attack family, not averaged across both) means all of:**
- Attacked Multi-Krum's Top-1 degradation, relative to its own paired honest FedAvg baseline, is measurably smaller than attacked FedAvg's corresponding degradation for the same attack family/seed (`EXP_016`–`EXP_019`'s already-established Δs, `analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md` §13) — i.e. genuine recovery, not merely a different but equally large gap.
- The clean-performance cost (§7, §9) is small enough to be plausibly worth paying for that recovery — this plan does not pre-commit to a specific percentage-point bound, since what counts as "worth it" is a judgment call for the eventual synthesis, not a criterion to be gamed by picking a lenient threshold now; it commits only to *reporting* the exact clean cost measured, unfiltered.
- Selection behavior is stable (not wildly different client-exclusion patterns round to round with no discernible pattern) — reported descriptively, not reduced to a single pass/fail number.

**These criteria are chosen conservatively and are not a promise Multi-Krum will meet them** — §6's fairness risk is real and pre-registered precisely because it might not.

## 17. Publication Interpretation

All four outcomes below are scientifically reportable, and none is designed to be more or less likely by any choice made in this plan:

- **A. Multi-Krum mitigates global-model damage but the rejection proxy tracks attribution poorly** — the expected, most likely outcome given §1/§5's framing: strengthens the paper's case that FedTrace v3 fills a genuine gap (attribution) that a standard robustness method does not close as a byproduct, while still giving a fair, quantified robustness comparison.
- **B. Multi-Krum both mitigates damage and its rejection proxy happens to track the true attacker** — would be reported honestly as a positive result for Multi-Krum's proxy, with the §5/§11 caveats (known `f`, repurposed signal, no onset guarantee) stated exactly as strongly as if the result had gone the other way.
- **C. Multi-Krum struggles under this project's strong Non-IID heterogeneity** (rejects honest clients, fails to isolate the true attacker, or both) — directly confirms the §6 fairness risk pre-registered before any run; this would be reported as a genuine, useful finding about the limits of distance-based Byzantine robustness under the exact heterogeneity level this project has already characterized in depth, not suppressed or reframed.
- **D. Multi-Krum measurably harms clean (honest-only) performance** — would be reported as the clean-performance cost §7/§9 exist specifically to quantify, and would directly inform whether Multi-Krum's mitigation (if any) is worth its cost.

**No outcome is designed to guarantee a favorable comparison for FedTrace.** The experiment matrix, parameters, and metrics are fixed identically regardless of which of A–D eventually occurs.

## 18. Experiment Budget

**Minimal, six new runs total (§7, §8): two clean controls, four attacked runs.** Explicitly **not** to be run alongside or instead of this:

- Krum + Multi-Krum + coordinate-wise median + trimmed mean **all together** — only Multi-Krum, per `research_plans/FEDTRACE_EXTERNAL_BASELINE_SELECTION_PLAN.md` §4/§13's own scope discipline.
- **New seeds** beyond 123/2026 — the same two already-validated seeds are reused, matching FedTrace v3's own Stage B matrix exactly.
- **New attack families** — only the two already-frozen transforms (`gaussian_update_noise`, `norm_preserving_directional`) are used.
- **Parameter sweeps over `f` or `m`** — both are fixed once (§2) and used identically across every run; no sweep of either is performed at this stage.
- **Any FedTrace v3 modification** — FedTrace v3 remains frozen throughout this entire baseline study, exactly as it has been since Stage B began.
- **New datasets or models** — CIFAR-100/ResNet-18/5-clients remains the evaluated configuration.

## 19. Decision Gate

**A. Multi-Krum's implementation is sufficiently specified to implement next.**

Every open design question this task's own instructions raised has a single, justified answer with no remaining ambiguity: the exact score formula and neighbor count (§2), `n=5/f=1/m=4` fixed and justified (§2), the update representation (flattened trainable-parameter delta, §3), the weighting rule (unweighted, matching the original method, explicitly flagged as a difference from FedAvg, §4), buffer handling (§4), the fairness caveats that must accompany any result (§5, §6), the clean-control requirement (§7, exactly 2 additional runs), the attacked-run matrix (§8, exactly 4 runs, reusing already-frozen attack parameters), required logging (§10), a fully pre-registered (not-to-be-tuned) repurposed attribution proxy (§11), the two-table comparison boundary (§12), the exact files expected to change (§13), and a complete test plan (§14). **No implementation occurs in this task.**

---

## Final Report

1. **File created:** `research_plans/MULTIKRUM_EXTERNAL_BASELINE_IMPLEMENTATION_PLAN.md`.
2. **Files modified:** none — every source document and codebase file consulted (`src/federated/aggregation/base.py`, `src/federated/aggregation/fedavg.py`, `src/federated/server.py`, `src/federated/runner.py`, `src/experiments/run_federated_cifar100.py`, `tests/test_aggregation.py`, and the prior plan/synthesis documents) was read-only.
3. **Exact Multi-Krum definition selected:** Blanchard et al. (NeurIPS 2017) Krum score (sum of squared Euclidean distances to the `n−f−2` nearest peer updates), Multi-Krum selection of the `m` lowest-scoring clients, unweighted average of their trainable-parameter deltas — the literature-faithful original, no invented variant (§2, §4).
4. **`n`/`f`/`m` settings:** `n=5`, `f=1` (fixed, matching this project's single-attacker threat model, not a free/YAML parameter), `m = n−f = 4`; feasibility (`n ≥ 2f+3`) satisfied exactly at the boundary (`5 ≥ 5`) — stated as a caveat, not hidden (§2).
5. **Update representation:** flattened delta (`client_model_state − global_state`) over trainable parameters only, identical to the vector every attack in this project's series already manipulates and to what FedTrace v3's own `z_m` is computed from (§3).
6. **Aggregation weighting rule:** unweighted average of the `m` selected clients' trainable-parameter tensors (original Multi-Krum convention, explicitly not FedAvg's sample-count weighting) — non-trainable floating buffers averaged the same way over the same selected set; non-floating buffers reuse FedAvg's existing `_max_across_clients`/`_require_identical` rules unchanged (§4).
7. **Clean-control requirement:** yes — required at both seeds (123 and 2026), 2 runs, to isolate robust-aggregation cost from attack-mitigation effect under this project's own already-documented strong Non-IID heterogeneity (§7).
8. **Attacked-run matrix:** 4 runs — Gaussian × {123, 2026}, directional × {123, 2026} — all attack parameters identical to `EXP_016`–`EXP_019`, no new attack introduced (§8).
9. **Required logging:** per-round Krum score, selected clients, rejected clients, and rank per client, for every run (clean and attacked), with no ground truth used in the selection logic itself (§10).
10. **Rejection-proxy definition:** first-anomalous-round / persistent-rejection (`P_krum=3`, matching FedTrace's own `P=3`) / onset-estimate / predicted-client, all fully pre-registered now and never to be tuned after seeing attacked results; every mention labeled "Multi-Krum rejection proxy," never "Multi-Krum malicious-client detector" (§11).
11. **Robustness metrics:** test loss, Top-1, Top-5, best validation Top-1, degradation vs. paired honest FedAvg, recovery vs. attacked FedAvg, clean-performance cost (§9).
12. **Traceability-proxy metrics:** predicted suspicious client, onset estimate (if confirmed), detection delay, honest persistent FP — reported only in the explicitly-caveated Table A, never merged with Table B's robustness metrics (§12).
13. **Main fairness caveat:** Multi-Krum is given a known/assumed Byzantine-count `f=1` that FedTrace v3 is never given and does not require — any comparison must state this asymmetry every time it is made (§5).
14. **Files expected to change during implementation:** new `src/federated/aggregation/multikrum.py`; modified `src/experiments/run_federated_cifar100.py` (`_SUPPORTED_AGGREGATIONS` and strategy dispatch); new `tests/test_multikrum_aggregation.py`; six new configs in a future task — `src/attacks/traceability.py`, `src/federated/runner.py`, `src/federated/server.py`, `src/federated/client.py`, and `src/analysis/fedtrace_detector.py` remain untouched (§13).
15. **Test plan summary:** pairwise-distance correctness, Krum-score correctness, `f`/`n`-feasibility validation, `m`-derivation validation, deterministic tie-breaking, malicious-outlier rejection and honest-update selection in hand-computable toy examples, buffer exclusion from scoring, shape/key-mismatch handling (shared with `FedAvg`), backward compatibility with `FedAvg`, and explicit confirmation that `tests/test_aggregation.py` needs no modification — none implemented or run in this task (§14).
16. **Success/failure criteria:** robustness success requires measurable recovery relative to attacked FedAvg at a plausibly acceptable clean-performance cost, evaluated per attack family and reported descriptively, not gamed by a lenient pre-chosen threshold; Multi-Krum rejecting an honest client, failing to isolate the true attacker, or imposing a large clean cost are all pre-registered as reportable scientific findings, not failures to tune away (§15–§17).
17. **Implementation justified next:** **A** — sufficiently specified to implement next; not implemented in this task (§19).
18. **What NOT to run:** multiple robust aggregators at once, new seeds, new attack families, `f`/`m` sweeps, any FedTrace v3 modification, new datasets/models (§18).
19. **Confirmation FedTrace v3 unchanged:** confirmed — `src/analysis/fedtrace_detector.py` was not touched (not even read again beyond what prior tasks already established; this task's file list did not require re-reading it, and it was not modified).
20. **Confirmation no experiment/config/code/test/database changed:** confirmed — only this one planning document was created; every codebase file consulted (`src/federated/aggregation/base.py`, `fedavg.py`, `src/federated/server.py`, `src/federated/runner.py`, `src/experiments/run_federated_cifar100.py`, `tests/test_aggregation.py`) was read-only.
21. **Confirmation no commands executed:** confirmed — only read-only file inspection (including one directory-listing check for existing aggregation-related test files) was performed; no shell, PowerShell, or Bash command executed, modified, or ran anything.
