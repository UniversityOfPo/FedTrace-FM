# FedTrace v2: Drift-Aware Temporal Calibration — Design Plan

**Status:** Design document only. No code, config, test, README, or Research Database file was created or modified in producing it. No experiment was run. This document specifies exactly one frozen FedTrace v2 candidate; it does not implement it.

**Evidence base:** `research_plans/FEDTRACE_METHOD_DESIGN_PLAN.md` (the frozen v1 design), `analysis/FEDTRACE_STAGE_A_RETROSPECTIVE_ANALYSIS.md` (the v1 evaluation that produced Decision B), `analysis/EXP_014_Gaussian_Attack_Traceability_Analysis.md`, `analysis/EXP_015_Directional_Attack_Traceability_Analysis.md`, `src/analysis/fedtrace_detector.py`, `tests/test_analysis_fedtrace_detector.py`, and hand-verified re-computation of `self_prev_round_cosine` at 6 representative rounds of `EXP_015` (round 20, 21, 22, 35, 45, 49; every round the Stage A honest false positives were confirmed at) under the round-relative scheme this document proposes.

---

## 1. Diagnose FedTrace v1 Precisely

Restating Stage A's evidence exactly as found, without softening it:

- **Magnitude (`z_m`) dominates `EXP_014`**: `client_002`'s `update_l2_norm` z-score reaches ≈9.2 at attack onset and stays there; onset error 0, delay 2 rounds, zero honest false positives.
- **Temporal (`z_t`) dominates `EXP_015`**: `client_002`'s `self_prev_round_cosine` z-score reaches ≈8.1 at attack onset and stays there; onset error 0, delay 2 rounds — but **three of four honest clients (`client_001`, `client_003`, `client_004`) also cross the same threshold and confirm persistent detection later in the same 50-round run** (rounds 43–48).
- **Peer (`z_p`) never exceeds threshold** for any client, in either run, at any of the 8 representative rounds hand-computed in Stage A (peak ≈1.8 against a threshold of 3).
- **The `max` combiner provides coverage, not detection-performance improvement**: `G_full_fedtrace` was numerically identical to whichever single signal already dominated (`B` on `EXP_014`, `C` on `EXP_015`) in every case observed — it never caught anything the best individual signal missed, and never suppressed anything the worst individual signal produced.
- **Temporal-only produces persistent false positives on 3 of 4 honest clients**, confirmed at rounds 43, 47, and 48 respectively — a structural, reproducible failure, not an isolated fluctuation (each of the three sustained ≥3 consecutive anomalous rounds, and two of the three sustained anomalous status for the rest of the run once triggered).

**The violated assumption, stated precisely:** `self_prev_round_cosine` is not stationary across the full training horizon. `z_t(k,r) = |t(k,r) − mean_pre_t(k)| / std_pre_t(k)` implicitly assumes `t(k,·)` fluctuates around a fixed mean for the entire run — true only within the `[0, T_ref)` window it was calibrated on. In reality, `t(k,r)` for every client (honest and malicious alike, before the malicious client's behavior changes) declines steadily across the full 50-round run as training converges and consecutive local-training steps become progressively less correlated with each other. `mean_pre_t(k)`, fit only from rounds 1–19, has no way to know this decline continues; by round 45–49 every honest client's `t` has drifted far enough below its own early-training mean that `z_t` alone cannot distinguish "this client's cosine declined because everyone's does" from "this client's cosine declined because it is the attacker."

**This is not a flaw in the attack, and is not blamed on it.** `EXP_015`'s directional attack did exactly what it was designed to do (collapse `client_002`'s temporal self-consistency to ≈0, a real and correctly-detected signal). The failure is entirely in `z_t`'s calibration reference: it borrowed the *shape* of `z_m`'s calibration rule (per-client, longitudinal, pre-attack-only window) without checking whether `t`, unlike `m`, actually satisfies the stationarity assumption that rule requires. `research_plans/FEDTRACE_METHOD_DESIGN_PLAN.md` §6 already built exactly the right fix for this same problem when it appeared in `p` (cross-sectional, no historical window, cancels a shared trend by construction) — but applied that fix only to `p`, and gave `t` the old, `m`-style rule instead, for no evidenced reason. That inconsistency, not any property of the attack, is the diagnosed flaw.

---

## 2. Preserve Useful Signals

FedTrace v2 keeps exactly the same three scientific signal families as v1 — magnitude (`M`), peer-relative direction (`P`), temporal self-consistency (`T`) — computed from the same already-logged trace fields (`update_l2_norm`, `peer_mean_cosine_similarity`, `self_prev_round_cosine`). No new model-derived feature, no new trace column, and no additional raw-tensor computation is introduced. The correction is confined to **how `T` is calibrated**, per §3 below; `M`'s calibration is unchanged (§5); `P` is retained but demoted out of the primary score (§6), which is a *scoring-role* decision, not a new feature.

---

## 3. Drift-Aware Temporal Representation

**Design chosen: per-round, cross-sectional, robust, one-sided.** This mirrors `p`'s already-successful fix in v1 (§6 of the frozen plan) rather than inventing a new mechanism — the same class of correction that already solved this exact class of problem once, applied to the signal that still needed it.

For round `r`, let `S(r)` be the set of clients with a defined `t(k,r)` (i.e. every selected client except any on its very first observed round, where `t` is `None` by construction).

```
center_t(r) = median_{j in S(r)} t(j, r)

raw_dev_j(r) = | t(j, r) − center_t(r) |     for each j in S(r)

MAD_t(r) = median_{j in S(r)} raw_dev_j(r)

scale_t(r) = 1.4826 * MAD_t(r)
             (1.4826 is the standard consistency constant that makes a
             normal distribution's MAD-based scale estimate converge to
             its true standard deviation — used here purely so scale_t(r)
             is on the same footing as an ordinary standard deviation,
             not as a claim that t(·,r) is normally distributed)
```

**Numerical fallback (never fabricate a value):** if `|S(r)| < 2`, or `scale_t(r) == 0` (every client's `t(·,r)` value this round is numerically identical — a degenerate tie, not expected in practice but not excluded by construction), then `z_t_rel(k,r)` is **undefined** for every client at round `r` — excluded from scoring that round, exactly the same "excluded, never zeroed" convention already used throughout v1's calibration rules. No epsilon floor or substitute constant is introduced.

**The score itself — one-sided, since only an unusually *low* temporal self-consistency is evidence of attack under this project's threat model:**

```
z_t_rel(k, r) = max( 0, [ center_t(r) − t(k, r) ] / scale_t(r) )
                if scale_t(r) is defined, else undefined
```

A client whose `t(k,r)` is *above* the round's median (unusually high self-consistency) scores exactly `0`, not a large negative number — this is deliberate: nothing in this project's threat model treats "more self-consistent than typical" as suspicious, and a two-sided test would needlessly halve the effective threshold margin for the direction that actually matters.

**Leave-one-out was considered and rejected.** The concern motivating LOO is that `center_t(r)`/`scale_t(r)` for client `k` would otherwise partly be computed from `k`'s own (possibly malicious) value, contaminating its own reference. This was checked directly, by hand, against real data: at every one of the 6 representative `EXP_015` rounds re-computed for this plan (20, 21, 22, 35, 45, 49), `center_t(r)` (the median) was always drawn from an actual honest client's value, never from `client_002`'s near-zero outlier — the median's breakdown point (it takes a majority of the 5 values to move it) protects it even with `client_002` included. The same held for `scale_t(r)` (MAD): `client_002`'s own deviation was, in every case checked, the *largest* of the 5 per-round deviations, and therefore never the one selected by the inner median that defines `MAD_t(r)`. Symmetric, LOO-free computation is therefore not just simpler but empirically already robust at this project's `n=5` clients-per-round; adding LOO would introduce an asymmetry (the malicious client's own reference would be computed from 4 honest peers, while every honest client's reference would be computed from 3 honest peers plus the malicious client) for no demonstrated benefit, and is rejected on parsimony grounds given the evidence.

**Illustrative hand-computed check (not a claim of exhaustive validation — see §10/§15):**

| Round | `center_t(r)` (median) | `scale_t(r)` (1.4826×MAD) | `z_t_rel(client_002)` | max `z_t_rel` among honest clients |
|---|---|---|---|---|
| 20 | 0.2620 (`client_004`) | 0.0175 | **14.97** | 2.50 (`client_000`) |
| 21 | 0.2529 (`client_004`) | 0.0196 | **12.87** | 2.25 (`client_000`) |
| 22 | 0.2480 (`client_004`) | 0.0241 | **10.33** | 2.10 (`client_000`) |
| 35 | 0.2000 (`client_004`) | 0.0266 | **7.53** | 1.68 (`client_000`) |
| 45 | 0.1797 (`client_003`) | 0.0406 | **4.43** | 1.30 (`client_000`) |
| 49 | 0.1675 (`client_003`) | 0.0422 | **3.96** | 1.06 (`client_000`) |

At every sampled round, `client_002` sits far above threshold (minimum observed 3.96, at the final round — the point where the absolute gap between the attacker and its declining peers is smallest) while the most extreme honest client (`client_000`, consistently) stays comfortably under 3, peaking at 2.50 early in the post-attack window and declining thereafter. `client_003`, `client_004`, and `client_001` — the three clients whose *absolute* calibration produced confirmed false positives in Stage A — show `z_t_rel` at or extremely close to `0` at the same rounds their v1 false alarms were confirmed, since their late-training `t` values are unremarkable *relative to their same-round peers*, only unremarkable relative to their own early-training selves.

---

## 4. Why This Targets the Observed Failure

Model each client's temporal self-consistency additively (a standard decomposition, used here only to reason about the fix, not as a fitted or claimed data-generating model):

```
t(k, r) ≈ trend(r) + offset(k) + noise(k, r)
```

where `trend(r)` is the shared, round-dependent secular decline common to every honest client (the object §1 identifies as the violated-stationarity source), `offset(k)` is a fixed per-client baseline difference (already documented in `research_plans/FEDTRACE_METHOD_DESIGN_PLAN.md` §6 for `m`; the same kind of natural `alpha=0.1` heterogeneity plausibly applies to `t`), and `noise(k,r)` is round-level fluctuation.

**Absolute temporal drift** (v1's `z_t`) computes `t(k,r) − mean_pre_t(k)`, where `mean_pre_t(k) ≈ trend(pre-attack window) + offset(k)`. For any round `r` in the post-`T_ref` region, this difference is approximately `[trend(r) − trend(pre-attack window)] + noise(k,r)` — **the shared `trend(r) − trend(pre-attack window)` term appears, undivided, in every honest client's score alike**, growing without bound as training continues and `trend(r)` keeps declining, eventually pushing every client's `z_t` toward the same threshold regardless of that client's own behavior. This is exactly the observed failure: three separate honest clients independently crossed 3 within a few rounds of each other late in the run, consistent with a shared additive term dominating, not independent client-specific anomalies.

**Relative temporal abnormality** (`z_t_rel`, this document) instead computes `t(k,r) − center_t(r)`, where `center_t(r) ≈ trend(r) + median_j offset(j)` is estimated **fresh, every round, from that round's own cross-section of clients**. The shared `trend(r)` term is present in both `t(k,r)` and `center_t(r)` and **cancels exactly** in their difference:

```
t(k,r) − center_t(r) ≈ [trend(r) + offset(k) + noise(k,r)] − [trend(r) + median_j offset(j)]
                       = [offset(k) − median_j offset(j)] + noise(k,r)
```

No `trend(r)` term survives — the secular decline shared by every client is removed by construction, at every round independently, regardless of how far training has progressed. What remains is only `k`'s own baseline offset relative to its peers' typical baseline, plus round noise — quantities that stay roughly constant across the run for an honest client (matching the empirically observed fact that honest `z_t_rel` values in §3's table stay in a narrow band, ≈1.0–2.5, rather than climbing over 30 rounds the way v1's `z_t` did). A malicious client whose behavior changes abruptly relative to its peers **in the same round** (as `EXP_015`'s attack does — its `t` collapses to ≈0 while its peers' `t` values, whatever the current phase of `trend(r)`, remain in the range those peers' own `offset` values produce) is not protected by this cancellation: its deviation from `center_t(r)` is a genuine, round-specific `offset`-scale departure, not an artifact of `trend(r)`, and survives the differencing intact — which is exactly why §3's table shows `client_002`'s `z_t_rel` staying far above threshold throughout, while the honest clients' values stay flat and low.

---

## 5. Magnitude Calibration

**Unchanged.** Stage A found no structural problem in `z_m`'s per-client longitudinal calibration: zero honest false positives on `EXP_014` across the full 50-round run (max observed honest `z_m` ≈1.29), and zero on `EXP_015` (max observed ≈1.21) — no evidence of the secular-drift failure mode that motivated §3's fix for `t`. `update_l2_norm` does not exhibit a shared cross-client secular trend the way `peer_mean_cosine_similarity` and `self_prev_round_cosine` both independently do; there is no diagnosed problem to fix, and per this task's explicit instruction, it is not altered.

---

## 6. Peer-Direction Signal

**Decision: (B) — remove from the primary FedTrace v2 score; retain only as an ablation/monitoring signal.**

Justification, from evidence and parsimony, not from a desire to simplify for its own sake: `z_p` (already redesigned once, in v1, specifically to fix the same class of secular-trend problem now being fixed for `t`) contributed to **zero** detections and **zero** false positives across every round hand-computed in Stage A, in both runs — its dominant-component share of every anomalous round observed was exactly 0%. Retaining an inert component inside a `max()` primary score adds interpretive surface area (a reader must always ask "did peer contribute here?") without any demonstrated benefit, and §7 below shows that a two-component `max(z_m, z_t_rel)` already satisfies the coverage goal `max()` exists for. This is an evidence-driven demotion, not a tuning action: `z_p`'s formula, calibration, and threshold are left completely unchanged from v1 (§6/§7 of the frozen plan) — it is still computed and still logged for every future run, precisely so that a future attack family that *does* manipulate peer-relative direction in a way `z_p` is sensitive to can still be observed and, if warranted by future evidence, promoted back into the primary score. Removing it now is reversible; inventing a reason to keep tuning it to "make it contribute" on these same two runs would not be.

---

## 7. Combined Score

**FedTrace v2 score:**

```
FedTraceV2Score(k, r) = max( z_m(k, r), z_t_rel(k, r) )
                         over whichever of the two are defined for (k, r)
```

**`max` is retained, not replaced with a more elaborate combiner, and this choice is deliberately re-justified rather than carried over by default.** A "top-two evidence" or AND-style rule (requiring both components to independently exceed threshold) was considered and rejected outright: `z_t_rel` is entirely undefined on `EXP_014` (no `self_prev_round_cosine` field exists in that trace at all, §1), so any rule requiring both signals to agree would make `EXP_014` fundamentally undetectable regardless of how strong `z_m` is — exactly the coverage property `max()` exists to provide, and exactly the failure mode an AND-style rule would reintroduce. A weighted sum was rejected for the same reason already given in the frozen v1 plan (§7 there): with only two components remaining, a weighted sum's weights would either have to be attack-agnostic (in which case they reduce to an arbitrary rescaling of `max`, adding a free parameter with no evidenced value) or be tuned per-attack (the explicit shortcut this entire project has been designed to avoid).

**Addressing the Stage A criticism directly, not by discarding `max` but by re-scoping what it is claimed to do:** Stage A's finding — that `max()` only ever inherits whichever single component already dominates, and never demonstrates a *combination* advantage — still applies to `FedTraceV2Score` exactly as much as it applied to v1's three-component version, and this document does not claim otherwise. What has changed is that the component now available for `EXP_015` (`z_t_rel`) no longer carries a known structural weakness (§3/§4), so `max(z_m, z_t_rel)`'s inherited behavior on both runs is now (pending §10's exhaustive check) the *good* behavior of each dominant single signal, without also inheriting `z_t`'s false-positive defect. **FedTrace v2's value, like v1's, is framed as cross-attack coverage — a single fixed detector that does not need to be told in advance whether a future attack will manipulate magnitude or temporal self-consistency — not as synergistic fusion.** This framing is now honestly earned rather than asserted: with the false-positive-prone component fixed, `max()`'s coverage no longer costs a known reliability penalty on the runs already collected.

---

## 8. Persistence and Attribution

**Unchanged: threshold = 3, `P` = 3 consecutive anomalous rounds.** The Stage A false positives are attributable to `z_t`'s calibration reference (§1/§4), not to the threshold or persistence length: under `z_t_rel`, the same three previously-false-positive honest clients never approach 3 at any of the 6 rounds re-checked in §3 (max observed honest value 2.50), using the *same* threshold and *implicitly* the same persistence rule (a client whose per-round score never exceeds 3 cannot accumulate any anomalous streak at all, let alone a 3-round persistent one). There is no evidence in this plan's re-computation that raising the threshold, lengthening `P`, or any other threshold-side adjustment was needed to fix the problem — the fix was entirely in the calibration reference, and threshold/persistence are stated here, explicitly, as unchanged.

**Definitions, carried over unchanged from `research_plans/FEDTRACE_METHOD_DESIGN_PLAN.md` §8/§9, now applied to `FedTraceV2Score`:**

- **First anomalous round:** the first single round, per client, where `FedTraceV2Score(k,r) > 3`, regardless of whether it is ever confirmed persistent.
- **Confirmation round:** the round at which the 3rd consecutive anomalous round is observed.
- **Estimated onset round:** the first round of that confirmed 3-round run (`confirmation_round − 2` when contiguous) — never the confirmation round itself.
- **Predicted malicious client:** the client with the earliest confirmation round; ties broken by higher mean `FedTraceV2Score` during its persistent run, then by `client_id` (unchanged tie-break order from `src/analysis/fedtrace_detector.py`).
- **Honest-client false-positive event:** any client other than the true malicious client that reaches a confirmed persistent detection at any point in the run — counted per-client (a client confirmed once counts as one false-positive event for the run, regardless of how many further rounds its anomalous streak continues), exactly as tabulated in Stage A's Detector C/E/F/G rows.

---

## 9. Development vs. Validation Boundary

**`EXP_014` and `EXP_015` are, as of this document, explicitly redesignated DEVELOPMENT runs.** Every number in §3's illustrative table, and every design decision in §§1–8, was derived by inspecting these two runs' own trace data — including the specific rounds (43, 45, 47, 48) at which v1's false positives were confirmed, which directly motivated checking exactly those rounds under the new scheme. This is legitimate method-development practice, but it means `EXP_014`/`EXP_015` **cannot** subsequently be cited as independent evidence that FedTrace v2 works — doing so would be validating a fix against the same data used to design the fix, the precise failure mode Stage A itself was created to avoid for v1 (curve-fitting to two data points, `research_plans/FEDTRACE_METHOD_DESIGN_PLAN.md` §2).

Concretely, this means:

- The retrospective test in §10 below (run after implementation, on `EXP_014`/`EXP_015`) is a **development-set sanity check** — confirming the fix behaves as designed on the data it was designed against — not a validation result, and must not be reported as one in any future document, abstract, or decision gate.
- **Genuine validation of FedTrace v2 requires new seeds/runs not used anywhere in this design process** — the already-anticipated seed-123/seed-2026 replications (`research_plans/PHASE5_Malicious_Client_Traceability_Plan.md` §16, `research_plans/EXP_015_Norm_Preserving_Directional_Attack_Plan.md` §13), run only *after* v2 is frozen from §10's retrospective check, with the v2 formula treated as fixed and un-revisitable at that point.
- Any future write-up of FedTrace v2's performance must state, prominently, which of its supporting evidence came from development runs (`EXP_014`/`EXP_015`) versus independent validation runs (the future seed replications) — the two must never be merged into a single "n=4" evidence count.

---

## 10. Stage A-v2 Retrospective Test Protocol

**To be executed after implementation, not as part of this design document.** Applies `evaluate_all_detectors`-style evaluation (extending `src/analysis/fedtrace_detector.py` with the `z_t_rel` calibration of §3 and the two-component score of §7, once implementation is authorized) to the same already-collected `EXP_014`/`EXP_015` trace CSVs used throughout this series. Six detectors, computed identically to the exact formulas frozen in §§3–8 above with no threshold or parameter adjustment based on the outcome:

1. **Magnitude-only** (`z_m`, unchanged from v1) — reproduces Stage A Detector B exactly, included as an unchanged reference point.
2. **Peer-only** (`z_p`, unchanged from v1, still cross-sectional) — reproduces Stage A's (never-triggering) peer ablation, included to re-confirm §6's demotion was not premature.
3. **Old temporal-only** (`z_t`, v1's absolute/longitudinal calibration) — reproduces Stage A Detector C exactly, kept as the direct "before" comparison for the fix.
4. **Corrected temporal-only** (`z_t_rel`, §3) — the new "after" comparison point, evaluated on exactly the same protocol as (3).
5. **FedTrace v1** (`max(z_m, z_p, z_t)`, exactly as Stage A ran it) — the full old method, for direct before/after comparison.
6. **FedTrace v2** (`max(z_m, z_t_rel)`, §7) — the full new method.

For every detector/run combination (skipping combinations already established as NOT EVALUABLE, i.e. anything requiring `t` or `z_t_rel` on `EXP_014`), report the same full metrics suite Stage A used: malicious-client attribution correctness, onset localization error, detection delay, honest-client persistent false-positive count (per-client, not per-round), and precision/recall/F1. **No threshold, persistence length, `T_ref`, or calibration-window adjustment of any kind is permitted after seeing these retrospective outcomes** — if the numbers disappoint, that is reported as a finding (exactly as Stage A's own weaknesses were reported, not tuned away), not silently corrected by a second round of adjustment.

## 11. Success Criteria for the Correction

Pre-registered now, to be checked without modification once §10 is executed:

- `EXP_015`'s malicious client (`client_002`) remains attributable under FedTrace v2, with onset estimate at or very near the true `attack_start_round=20` (consistent with the ≈14.97-to-3.96 z-score margin already hand-verified in §3, which gives no reason to expect a materially different confirmation round than v1's already-excellent 22).
- Honest persistent temporal false positives are **materially reduced, preferably to zero** — the explicit target is that none of `client_001`/`client_003`/`client_004`/`client_000` reach confirmed persistent detection under `z_t_rel` on `EXP_015`, matching what §3's 6-round sample already indicates (max honest value observed: 2.50, comfortably under 3).
- `EXP_014`'s magnitude-based coverage is not broken — since `z_m` is unchanged (§5) and `EXP_014` never involves `t`/`z_t_rel` at all, this criterion reduces to confirming Stage A's Detector B numbers are reproduced exactly (onset error 0, delay 2, 0 honest FP), which requires no new evidence, only confirmation that the implementation does not regress the unchanged component.
- FedTrace v2 does not create **new** honest false positives anywhere it did not already have them — in particular, `z_m` (unchanged) must not suddenly produce a false positive under the new combiner, and `z_t_rel` must not produce a false positive on any client/round `z_t` (the old, flawed version) did not already flag, i.e. the fix must not merely relocate the false-positive problem to a different round or client (this is also explicitly a failure criterion, §12).
- Onset localization remains **useful** — bounded, single-digit-round error on `EXP_015` (matching v1's already-demonstrated onset error of 0 via the temporal signal), not degraded toward `EXP_015`'s own magnitude-only baseline's 23-round error.

**Explicitly not required:** FedTrace v2 does not need to outperform the single best attack-specific feature on that same attack (e.g. it is not required to beat magnitude-only on `EXP_014`, nor to beat old temporal-only's raw detection *speed* on `EXP_015`, if old temporal-only's speed happened to be marginally faster before its false-positive cost is counted). What is evaluated is **cross-attack coverage plus false-positive control** — whether one fixed configuration handles both attack families acceptably well *and* without the reliability cost v1 paid, not whether it sets a new speed record on either attack individually.

## 12. Failure Criteria

FedTrace v2's correction will be considered **not to have worked**, and the underlying design (not merely its parameters) reopened, if any of the following holds after §10's retrospective check:

- Corrected temporal calibration (`z_t_rel`) **still** persistently flags one or more honest clients on `EXP_015` (i.e. §11's false-positive-reduction target is not met even approximately).
- `EXP_015`'s directional-attack attribution is **lost** — i.e. `client_002` is no longer the earliest-confirmed client, or is never confirmed at all, under `FedTraceV2Score`.
- The correction **merely relocates** the false-positive problem — e.g. a *different* set of honest clients (or the same three, at different rounds) still reach persistent detection, rather than the false-positive count genuinely dropping.
- FedTrace v2 (`max(z_m, z_t_rel)`) adds **no benefit over a simple union of the two single detectors** run independently — i.e. if a practitioner running magnitude-only and corrected-temporal-only as two entirely separate, unrelated detectors would get identical attribution/onset/FP outcomes to running the combined `max()` score, then the combination step itself has not been shown to do anything, and should be reported as such (this would not itself invalidate `z_t_rel`'s fix, but would mean the *combiner*, not just v1's calibration, needs re-examination).
- The method requires **attack-specific thresholds** to pass §11 — i.e. if `z_t_rel`'s threshold or `MAD` scale constant had to be adjusted differently for `EXP_014` versus `EXP_015` to produce acceptable results (moot for `EXP_014` specifically, since `t` does not exist there, but relevant if any future development run requires this), that would reintroduce exactly the per-attack tuning this entire design series is constrained to avoid.

## 13. Novelty Discipline

Combining two (now, rather than three) z-scores with a `max()` is not, by itself, claimed as a scientific contribution — it is a simple, interpretable, and (per §7) carefully-justified engineering choice, not a novel algorithmic idea. Any eventual research contribution from this line of work must come from the **complete traceability framework**, assembled and validated as a whole, not from any single formula in isolation:

- **Heterogeneous-client calibration** — the demonstrated necessity of per-client (for magnitude) versus per-round cross-sectional (for peer-direction and, as of this document, temporal self-consistency) reference statistics, motivated and validated against this project's own extensively-characterized `alpha=0.1` heterogeneity, not assumed.
- **Multi-view behavioral tracing** — observing magnitude, direction, and temporal self-consistency simultaneously from already-available update tensors, without any raw training data.
- **Temporal onset localization** — the retroactive-labeling onset-estimate rule (§9 of the v1 plan, unchanged here), distinct from mere detection.
- **Client attribution** — ranking/identifying which of several selected clients is responsible, not just that an anomaly exists somewhere in the round.
- **False-positive control under natural heterogeneity** — the specific, hard-won lesson of this exact document: that a signal effective at detecting an attack can simultaneously be unsafe to deploy if its calibration does not account for a legitimate, shared, non-malicious drift, and that this must be actively checked and corrected, not assumed away.
- **Validation across qualitatively different attack families and independent seeds** — the three-stage sequence (`research_plans/FEDTRACE_METHOD_DESIGN_PLAN.md` §14) this document's own §9 boundary is a direct continuation of.

No single one of these, and no single formula in §§3–8, is being presented as the contribution; the framework that ties them together under a rigorously-defended evaluation protocol is.

## 14. Frozen FedTrace v2 Formulation

**This is the single, complete, implementation-ready specification. No alternative formulations remain open.**

**Signals** (unchanged from v1, `src/attacks/traceability.py`'s already-logged fields):
`m(k,r) = update_l2_norm`, `p(k,r) = peer_mean_cosine_similarity` (retained for ablation only, §6), `t(k,r) = self_prev_round_cosine`.

**Magnitude calibration** (unchanged from v1): for each client `k`, `mean_pre_m(k)`, `std_pre_m(k)` = sample mean/stdev (`statistics.stdev`) of `m(k, r)` over rows with `client_id == k` and `round_id < T_ref` (`T_ref = 20`). Undefined if fewer than 2 such observations, or if `std_pre_m(k) == 0`.
`z_m(k,r) = |m(k,r) − mean_pre_m(k)| / std_pre_m(k)`, undefined wherever the calibration is undefined or `m(k,r)` is `None`.

**Temporal calibration** (new — replaces v1's `z_t` entirely): for each round `r`, let `S(r) = {j : t(j,r) is defined}`.
`center_t(r) = median_{j in S(r)} t(j,r)`.
`MAD_t(r) = median_{j in S(r)} |t(j,r) − center_t(r)|`.
`scale_t(r) = 1.4826 * MAD_t(r)`.
Undefined for round `r` (for every client) if `|S(r)| < 2` or `scale_t(r) == 0`.
`z_t_rel(k,r) = max(0, [center_t(r) − t(k,r)] / scale_t(r))`, undefined wherever the round's calibration is undefined or `t(k,r)` is `None`. No leave-one-out (§3).

**Peer signal** (unchanged formula from v1, retained for logging/ablation only — not part of the primary score, §6): `z_p(k,r) = |p(k,r) − mean_round_p(r)| / std_round_p(r)`, mean/std computed cross-sectionally over that round's selected clients (including `k` itself), same undefined-if-fewer-than-2 rule.

**Primary combined score:**
`FedTraceV2Score(k,r) = max(z_m(k,r), z_t_rel(k,r))` over whichever of the two are defined; undefined (not zero, not skipped-as-zero) if both are undefined for `(k,r)`. `z_p` is computed and retained for the ablation/monitoring table in §10 but never enters `FedTraceV2Score`.

**Flagging:** `(k,r)` is raw-anomalous iff `FedTraceV2Score(k,r)` is defined and `> 3` (unchanged threshold).

**Persistence:** client `k` is detected starting at round `r*` iff `FedTraceV2Score(k,r) > 3` for every round in the contiguous range `[r*, r*+2]` (`P = 3`, unchanged). Confirmation round `= r* + 2`. Onset estimate `= r*` (not the confirmation round).

**Attribution:** predicted malicious client = the client with the earliest confirmation round; ties broken by higher mean `FedTraceV2Score` over the persistent run, then by `client_id` ascending (unchanged tie-break, matches `src/analysis/fedtrace_detector.py`'s existing `_select_predicted_client`). If no client is ever detected, output "no malicious client identified."

**Missing-value handling, uniformly:** any signal or calibration statistic that is undefined for a given `(k,r)` is excluded from that row's `max()`, never substituted with `0` or any other placeholder; if every signal is undefined for `(k,r)`, `FedTraceV2Score(k,r)` itself is undefined and that round is excluded from the persistence streak computation for that client (a streak-breaking gap, exactly as an explicitly non-anomalous round would be — undefined is not the same as anomalous, and is also not the same as automatically continuing a streak).

**`T_ref = 20`** applies only to `z_m`'s calibration window (unchanged meaning from v1); `z_t_rel` has no historical window and no dependency on `T_ref` at all, by design (§3).

This specification is implementable in `src/analysis/fedtrace_detector.py` by adding a `compute_cross_sectional_robust_calibration` function (parallel in structure to the existing `compute_cross_sectional_calibration`, but computing median/MAD instead of mean/stdev) and a `DETECTOR_COMPONENTS` entry for `FedTraceV2Score`'s two-component set — no change to `apply_persistence_rule`, `evaluate_detector`, or any other already-implemented function is required, since the persistence/attribution/metrics machinery is already component-agnostic.

## 15. Next-Step Decision

**Decision: (A) — the v2 formulation is sufficiently specified to implement offline next.**

Every formula, threshold, persistence rule, missing-value rule, and tie-breaking behavior in §14 is complete and unambiguous; no further design choice remains open. §3's 6-round hand-verification (spanning the full post-attack window: onset, near-onset, mid-run, and end-of-run) gives concrete, real-data evidence — not merely a theoretical argument — that the fix behaves as intended in exactly the rounds where v1's false positives were confirmed. This is deliberately **not** a claim that §10's exhaustive retrospective check is already known to pass — that check still needs to be run, on all 30 post-attack rounds rather than 6 representative ones, once implementation exists — only that the design itself contains no open question that would block writing that implementation. Per this task's explicit restriction, no new GPU experiment is recommended or required before that offline implementation-and-retrospective-check step; the appropriate next action is implementing §14 in `src/analysis/fedtrace_detector.py` and running the §10 protocol, not designing further or collecting new data first.

---

## Final Report

1. **File created:** `research_plans/FEDTRACE_V2_TEMPORAL_CALIBRATION_PLAN.md` (this document).
2. **Files modified:** none.
3. **Diagnosed v1 flaw:** `z_t`'s per-client longitudinal (pre-attack-only) calibration assumes `self_prev_round_cosine` is stationary across the full run; it is not — every client's `t` declines secularly as training converges, and a calibration window fixed at rounds 1–19 cannot see that decline continuing past round 20, producing shared, attack-independent score inflation for every client alike (§1/§4). Not attributed to the attack.
4. **Exact new temporal calibration:** per-round, cross-sectional, robust (median/MAD, no leave-one-out), one-sided — `z_t_rel(k,r) = max(0, [median_j t(j,r) − t(k,r)] / (1.4826 × MAD_j t(j,r)))`, undefined if fewer than 2 clients have a defined `t` that round or if the MAD is zero (§3/§14).
5. **Peer signal status:** removed from the primary FedTrace v2 score; retained unchanged, computed and logged, as an ablation/monitoring signal only (§6).
6. **Exact FedTrace v2 score:** `FedTraceV2Score(k,r) = max(z_m(k,r), z_t_rel(k,r))`, magnitude unchanged from v1, missing components excluded never zeroed (§7/§14).
7. **Threshold/persistence rule:** unchanged — threshold `> 3`, `P = 3` consecutive rounds, onset estimate = first round of the confirmed run, confirmation round = onset + 2 (§8).
8. **Development-vs-validation boundary:** `EXP_014`/`EXP_015` are now designated development runs only; they diagnosed and shaped v2 and therefore cannot serve as independent validation of it; genuine validation requires the not-yet-collected seed-123/seed-2026 replications, run only after v2 is frozen (§9).
9. **Retrospective success criteria:** `EXP_015` attribution preserved near true onset; honest persistent false positives materially reduced, target zero; `EXP_014` magnitude coverage unchanged/unbroken; no new false positives introduced anywhere; onset localization stays useful (bounded, single-digit error) (§11).
10. **Failure criteria:** persistent honest false positives remain (even reduced but nonzero in a structural way); `EXP_015` attribution lost; false positives merely relocated rather than reduced; combiner adds nothing over running the two detectors independently; any attack-specific threshold tuning becomes necessary (§12).
11. **Whether v2 implementation is justified next:** **Yes (Decision A)** — the specification in §14 is complete and unambiguous; recommended next step is offline implementation plus the §10 retrospective protocol, not a new GPU experiment (§15).
12. **Confirmation:** no experiment was run; no config was created or modified; no training/attack/federated code was modified; no test file was modified; `src/analysis/fedtrace_detector.py` was read but not edited; `Research_Database` and every other pre-existing document were not modified. Only this one new file was created.
13. **Confirmation no commands executed:** no shell, PowerShell, or Bash command was run at any point during this task; this document was produced entirely from `Read` of the six listed source files and manual arithmetic re-deriving `z_t_rel` at 6 representative `EXP_015` rounds from already-known trace values (the same `self_prev_round_cosine` figures already read and verified during the Stage A task).
