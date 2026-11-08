# FedTrace v3: Historical Residual Calibration for the Temporal Signal — Design Plan

**Status:** Design document only. No code, config, test, README, or Research Database file was created or modified in producing it. No experiment was run. This document specifies exactly one frozen FedTrace v3 candidate; it does not implement it.

**Evidence base:** `research_plans/FEDTRACE_V2_TEMPORAL_CALIBRATION_PLAN.md`, `analysis/FEDTRACE_V2_RETROSPECTIVE_ANALYSIS.md` (the v2 evaluation that found `client_000` confirmed persistent at round 11, before `EXP_015`'s attack even begins), `research_plans/FEDTRACE_METHOD_DESIGN_PLAN.md`, `src/analysis/fedtrace_detector.py`, `tests/test_analysis_fedtrace_detector.py`, and hand-verified re-computation (below) of the proposed v3 score at the exact rounds that broke v2 (`client_000`, rounds 9–11, 15, 19, 45) plus the rounds that confirmed v1/v2's true positive (`client_002`, rounds 20–22), using already-known `self_prev_round_cosine` and round-median values from prior work in this series.

---

## 1. Diagnose v2 Precisely

Restating `analysis/FEDTRACE_V2_RETROSPECTIVE_ANALYSIS.md`'s finding exactly, and separating it clearly from the distinct v1 failure it is not:

- `EXP_015`'s malicious `client_002` produced a strong, correctly-signed relative temporal anomaly under v2: `z_t_rel` reached 14.97 at round 20 and stayed far above 3 through round 49. v2 confirmed `client_002` persistent at round 22, onset estimate 20 — this part of v2 worked exactly as designed.
- But honest `client_000` also received a confirmed persistent detection, at round 11 (rounds 9, 10, 11), **before the attack starts at round 20 and before `client_002` is ever confirmed**. Because attribution picks the *earliest*-confirmed client, `client_000` — not `client_002` — was v2's predicted malicious client. Final attribution was wrong.
- The mechanism, verified by hand in the v2 retrospective analysis: v2's scale statistic (`scale_t(r) = 1.4826 × MAD` of that round's 5 clients' `t` values) is estimated from only 5 observations, freshly every round. On rounds where the 4 non-`client_000` clients happened to cluster unusually tightly by chance (round 10's scale was only `0.00193`, round 9's `0.00706`), an ordinary, non-malicious gap for `client_000` (`0.024`–`0.028`, unremarkable in absolute terms — smaller than several other honest-to-honest gaps elsewhere in the run that did *not* trigger) was divided by a tiny denominator and inflated past threshold.

**This is a categorically different failure from v1's.** v1 failed because `z_t`'s calibration reference (a fixed pre-attack-only per-client mean) could not track a shared secular decline that continued past the calibration window — a **non-stationarity** problem, and its false positives (`client_001`, `client_003`, `client_004`) all appeared *late* in training, long after `client_002`'s own correct confirmation, so attribution was never actually affected by v1's flaw. v2 correctly diagnosed and fixed that non-stationarity problem (§4 of the v2 plan derived, and the retrospective analysis reconfirmed, that same-round cross-sectional referencing cancels a shared trend by construction). v2's failure is a **small-sample scale-estimation instability** problem — nothing to do with drift or stationarity, entirely to do with `n=5` being too few observations to estimate a robust spread reliably on any single round. **v3 must keep v2's fix for the first problem while replacing only the part of v2 that caused the second one.**

## 2. Define Temporal Residual

**Same-round common reference (kept from v2, re-justified below):**

```
c(r) = median_{j in S(r)} t(j, r)
```

where `S(r)` is the set of clients with a defined `t(j,r)` at round `r` (unchanged from v2's definition).

**Client residual:**

```
d(k, r) = c(r) − t(k, r)
```

Positive `d(k,r)` means client `k`'s temporal self-consistency is *lower* than the round's common (shared-drift-adjusted) behavior that round — the same sign convention as v2's numerator, kept for continuity and because v3 only replaces how this quantity's *scale* is estimated, not what it means.

**Leave-one-out decision: NOT used — `c(r)` includes client `k`'s own value, same as v2.** Re-examined here specifically (not merely carried over) because v3's structure changes what `c(r)` feeds into: in v2, `c(r)` fed directly into an absolute threshold test; in v3, `c(r)` only ever appears inside a *difference*, `d(k,r)`, which is itself later compared only to client `k`'s own historical typical difference. Two independent arguments support skipping LOO:

1. **Median robustness, re-confirmed.** As already established empirically in the v2 plan (§3) and reconfirmed in the v2 retrospective analysis, at `n=5` the median has a high enough breakdown point that a single client's own extreme value essentially never moves it — `c(r)` was drawn from an actual honest client's value at every round checked, never from `client_002`'s own near-zero post-attack outlier.
2. **Self-referential contamination is now further diminished, not amplified, by v3's structure.** If including `k`'s own value in `c(r)` pulls the median slightly toward `k`, this shifts `d(k,r)` slightly *toward* zero — a small, conservative bias that makes `k` look marginally more consistent with the round's center, not less. For an honest client this is harmless; for a malicious client (`client_002`, whose post-attack `z_t_res` values are computed below to be in the range of 30, §3) this tiny conservative bias is immaterial to detection. LOO would add an asymmetric computation (the malicious client's own reference computed from 4 honest peers, honest clients' references computed from 3 honest + 1 malicious peer) for a benefit too small to measure against the evidence available.

Note that LOO does **not** apply to §3's historical scale at all — that quantity is single-client by construction (each client's own past residuals), so there is no "other observations" to leave out or include.

## 3. Historical Residual Calibration

**Instead of estimating scale from the 5 clients present in the current round (v2's flaw), calibrate each client's residual using that same client's own historical residuals** — the same class of fix that already worked for `z_m` (per-client, longitudinal, `~19` pre-attack observations, never unstable in either development run) and that the old `z_t` used unsuccessfully for the *center* (not the scale) of a non-stationary quantity. Applying a longitudinal, per-client reference to a *residual* (which, per §4's derivation, has already had the shared drift removed and should be much closer to stationary) is the mechanism this plan is built around.

```
D_k = { d(k, r) : r in [1, T_ref) }        (T_ref = 20, unchanged deployment constant)
```

(`r` starts at 1, not 0, because `t(k,0)` — and therefore `d(k,0)` — is undefined for every client by construction, mirroring `self_prev_round_cosine`'s existing semantics.)

**Center/scale formulation: standard sample mean/standard deviation, not median/MAD.** This is a deliberate choice, not a default: `D_k` has up to 19 observations, an order of magnitude more than the `n=5` sample that caused v2's instability, and standard mean/stdev is already this project's established convention for longitudinal (many-observation) calibration — used by `z_m` since v1, and by the old `z_t` before it (whose flaw was in *what window* it referenced, not in mean/stdev as an estimator). Robust statistics (median/MAD) were introduced in v2 specifically because `n=5` made an ordinary sample standard deviation too unstable there; at `n≈19`, that specific justification for robustness does not apply, and using a second, different scale-estimation convention without evidence it is needed would be unjustified complexity. As a secondary consideration: a standard-deviation estimate over a window that happens to include a somewhat more volatile early-training sub-period (as this project's own data shows, e.g. round 1's residuals) would, if anything, tend to *inflate* the estimated scale relative to a robust alternative — the opposite failure direction from v2's problem (an accidentally *tiny* scale), and therefore not a risk of reproducing the same failure mode by a different route.

```
historical_center_k = mean(D_k)
historical_scale_k  = stdev(D_k)          (sample standard deviation, statistics.stdev convention)
```

**Score:**

```
z_t_res(k, r) = max(0, [d(k, r) − historical_center_k] / historical_scale_k)
```

One-sided, same convention as v2 and the same justification (an unusually *high* consistency relative to `k`'s own typical residual is not evidence of attack under this project's threat model).

**Missing/degenerate rules (never fabricate a value):** `historical_center_k`/`historical_scale_k` are undefined if fewer than `MIN_CALIBRATION_OBSERVATIONS = 2` values exist in `D_k` (same threshold used everywhere else in this series), or if `historical_scale_k == 0` (degenerate zero-variance case). `z_t_res(k,r)` is undefined wherever the calibration is undefined, or wherever `d(k,r)` itself is undefined (i.e. `c(r)` is undefined for round `r` — fewer than 2 clients present that round — or `t(k,r)` is undefined). No epsilon floor or substitute value is used at any step.

**Hand-verified check at the exact rounds that broke v2** (computed from already-known `self_prev_round_cosine` and round-median values established in the v2 implementation/retrospective work — not re-executed, not fabricated):

`client_000`'s historical residual calibration from `D_{client\_000}` (19 pre-attack values, computed from the same `c(r)` medians already used in v2): `historical_center = 0.01193`, `historical_scale = 0.04288`.

| Round | `d(client_000, r)` | `z_t_res(client_000, r)` | v2's `z_t_rel(client_000, r)` (for comparison) | Anomalous under v3? |
|---|---|---|---|---|
| 9 | 0.02761 | **0.366** | 3.91 | no |
| 10 | 0.02388 | **0.279** | 12.35 | no |
| 11 | 0.03436 | **0.523** | 4.45 | no |
| 15 | 0.04362 | **0.739** | 10.86 | no |
| 19 | 0.04602 | **0.795** | 3.84 | no |
| 45 | 0.05276 | **0.952** | 1.30 (v2, already below threshold there) | no |

At every one of these six rounds — including the exact three (9, 10, 11) whose false persistent confirmation broke v2's attribution — `z_t_res(client_000, ·)` stays below 1.0, nowhere close to the threshold of 3. The mechanism is direct: `client_000`'s historical calibration already "knows," from 19 observations, that this client typically runs about `0.012` below the round median with a spread of about `0.043` — so an ordinary round-9 gap of `0.028` (previously inflated to `z=3.91` by a same-round scale that happened to be only `0.007`) is now compared against a stable, many-observation scale and registers as unremarkable (`z=0.37`).

`client_002`'s (the true malicious client's) historical residual calibration: `historical_center = 0.00148`, `historical_scale = 0.00805` (tight — this client's own pre-attack residuals are consistently small, since it is frequently close to or at the round median pre-attack).

| Round | `d(client_002, r)` | `z_t_res(client_002, r)` |
|---|---|---|
| 20 | 0.26227 | **32.4** |
| 21 | 0.25278 | **31.2** |
| 22 | 0.24844 | **30.7** |

`client_002` remains overwhelmingly, immediately detected — if anything more sharply separated than under v2 (`z` in the 30s rather than the teens), since its own historical scale is tight and its post-attack residual is large in absolute terms regardless of any single round's cross-sectional clustering. Persistence confirms at round 22 exactly as under v1/v2 (rounds 20, 21, 22 all far above 3), onset estimate 20.

**This is a targeted check, not the full retrospective protocol.** It was deliberately aimed at the exact rounds already known to have broken v2 (rather than an arbitrary sample, which is what caused v2's own design-stage check to miss this class of problem in the first place) plus the rounds needed to confirm the true positive still works. A full round-by-round check of every client at every round of both development runs is §10's job, performed only after implementation, not here.

## 4. Avoid Circularity / Contamination

- **Reference window establishment:** `D_k` is computed once, from rows with `client_id == k` and `round_id` in `[1, T_ref)` — identical windowing convention to `z_m`'s existing `compute_longitudinal_calibration`. No round `>= T_ref` is ever included in `D_k`.
- **Minimum observations:** `>= 2`, same convention as every other calibration statistic in this series (`z_m`, v1's `z_t`, v2's `z_t_rel`).
- **Does scoring begin only after calibration is frozen?** For this offline/retrospective analysis, no special carve-out is introduced: `z_t_res(k,r)` is computed for *every* round `r >= 1` (including rounds inside `[1, T_ref)` itself) using the one fixed calibration derived from the whole `[1, T_ref)` window — this exactly mirrors `z_m`'s existing, already-accepted convention (`z_m` is likewise scored for pre-`T_ref` rounds using a calibration fit from that same window, including the row being scored). This is retrospectively well-defined and introduces no future-round leakage (the calibration window never extends past `T_ref` regardless of which round is being scored). A genuine *online* deployment would sensibly not act on any round's score before `T_ref` regardless — the persistence rule itself requires observing `P=3` consecutive anomalous rounds, so no detection decision could fire before round `T_ref + 2` at the very earliest even if scoring were technically defined earlier — but this document does not propose a different offline scoring rule than the one already used for `z_m`.
- **Round-0/first-observation artifacts:** `t(k,0)` is undefined for every client (no prior submitted update exists yet), so `d(k,0)` is undefined and round 0 is excluded from `D_k` and from scoring, by construction — no special-case logic is needed beyond the existing "undefined propagates, never fabricated" rule already used throughout this series.
- **An attacker beginning during the calibration period.** This is an explicit, not-hidden assumption, inherited unchanged from the existing threat model (`research_plans/FEDTRACE_METHOD_DESIGN_PLAN.md` §3: "a malicious client may begin attacking at any round, potentially after an honest warm-up period," matched by every experiment run so far, both of which use a delayed onset at exactly `T_ref`). **v3 assumes every client, including the eventual malicious one, behaves honestly throughout `[0, T_ref)`.** If this assumption is violated — an attacker active before `T_ref` — `D_k` for that client would itself be contaminated by malicious-round residuals, folding attack behavior into that client's own "normal" baseline. This could mask detection of an attack that began before `T_ref` and continues past it (the client's own inflated `historical_scale_k` would make its later, still-malicious residuals look unremarkable relative to its own already-corrupted baseline) — a known, explicitly out-of-scope limitation of the whole calibration approach (already true of `z_m` in exactly the same way), not a new one introduced by v3, but worth restating here because v3 now has **two** calibrated per-client statistics (`z_m`'s and `z_t_res`'s) sharing this identical vulnerability, where v2 had only one (`z_m`'s — `z_t_rel`'s same-round reference had no historical window to poison, which was arguably its only structural advantage over this respect, traded away here for stability elsewhere).

## 5. Magnitude Component

**Unchanged from v1/v2.** No evidence from either development run's retrospective analysis shows a structural problem in `z_m`'s per-client longitudinal calibration — zero honest false positives on either run, under either v1 or v2, across every check performed so far. Per this task's explicit instruction, it is not altered here either.

## 6. Peer Signal

**Kept outside the primary v3 score, unchanged from v2's decision.** No new evidence from the v2 retrospective analysis changes §6 of the v2 plan's finding that `z_p` never contributed to a single detection or false alarm in either development run. It remains computed via v1's own unchanged functions for ablation/monitoring purposes (§10's comparison table), exactly as in v2. It is not tuned to force contribution.

## 7. FedTrace v3 Primary Score

**Candidate, evaluated and retained:**

```
FedTraceV3Score(k, r) = max(z_m(k, r), z_t_res(k, r))
```

over whichever of the two are defined; undefined (never zero) if both are undefined.

**Explicitly re-evaluated, not defaulted to.** The alternative most worth considering was requiring *both* signals to agree (an AND-style rule) — rejected for the same reason it was rejected in the v2 plan (§7 there): `z_t_res` is entirely undefined on `EXP_014` (`self_prev_round_cosine` does not exist in that trace at all), so any AND-style rule would make `EXP_014` permanently undetectable regardless of `z_m`'s strength, reintroducing exactly the coverage failure `max()` exists to prevent. A weighted sum was again rejected: with only two components, any attack-agnostic weighting reduces to an arbitrary rescaling of `max` with no evidenced benefit, and an attack-specific weighting is the exact shortcut this entire design series exists to avoid.

**Framing, honestly re-stated given the new evidence.** §9 of the v2 retrospective analysis already established that `max()`'s "coverage, not fusion" property does not protect a component from its own flaws — it did not shield v2 from `z_t_rel`'s instability, and there is no reason to expect it would behave differently for v3 if `z_t_res` turns out to have its own undiscovered flaw. What has changed is that `z_t_res`'s specific, identified failure mode (small-`n` same-round scale instability) does not apply to it by construction (its scale is now single-client historical, not 5-client cross-sectional) — but this is a claim about *that one mechanism* being closed, not a claim that `z_t_res` is now provably free of every possible weakness. `FedTraceV3Score`'s value, exactly as with v1 and v2, is best understood as cross-attack coverage: a single fixed detector that does not need to be told in advance whether a future attack will manipulate magnitude or temporal self-consistency, not as synergistic fusion between the two.

No learned model and no attack-specific weights are introduced anywhere in this design.

## 8. Threshold / Persistence

**Unchanged: threshold `> 3`, `P = 3` consecutive anomalous rounds.** Neither v2's flaw nor its fix has anything to do with the threshold or persistence values — the problem was entirely in how the *scale* denominator was estimated, and §3's hand-verified table shows the fix working with these exact same, unaltered values (`client_002`'s scores land in the 30s, `client_000`'s stay under 1.0 — nothing here comes close to suggesting the threshold itself needs adjustment). Per this task's explicit instruction, threshold and persistence are not tuned to fix v2; they are carried over unchanged.

**Definitions, unchanged from v1/v2, now applied to `FedTraceV3Score`:**

- **First anomalous round:** first single round, per client, where `FedTraceV3Score(k,r) > 3`.
- **Confirmation round:** the round at which the 3rd consecutive anomalous round is observed.
- **Estimated onset round:** the first round of that confirmed 3-round run (`confirmation_round − 2`).
- **Predicted malicious client:** the client with the earliest confirmation round; ties broken by higher mean `FedTraceV3Score` during the persistent run, then by `client_id` (unchanged tie-break, matches `src/analysis/fedtrace_detector.py`'s existing `_select_predicted_client`).
- **Honest persistent false positive:** any client other than the true malicious client that reaches confirmed persistent detection at any point in the run.

## 9. Causality

Every `FedTraceV3Score(k,r)` value is computable using only information available by round `r`: `z_m(k,r)` depends only on `k`'s own `update_l2_norm` history up to `T_ref` (fixed before `r` for any `r >= T_ref`, and, per §4, computed identically to `z_m`'s existing precedent for `r < T_ref`); `z_t_res(k,r)` depends only on `d(k,r)` (itself a function of round `r`'s own cross-section, never a future round) and `k`'s own `D_k` (fixed from `[1, T_ref)`, never updated with data from round `r` or later). **The historical residual calibration `D_k` becomes frozen once `[1, T_ref)` is complete — round `T_ref − 1` (round 19) is the last round contributing to `D_k`, and it is never revisited or extended afterward, for any client, regardless of how the run subsequently unfolds.** This mirrors `z_m`'s already-established calibration-freezing convention exactly; no new freezing rule is introduced. For an eventual online deployment (not proposed or authorized here — see §14), the natural operational interpretation is that no detection decision can be acted upon before round `T_ref + 2` in any case (the earliest a `P=3` persistent run can complete once scoring begins at round `T_ref`), independent of whether individual round scores are technically defined earlier.

## 10. Retrospective Development Test Protocol

**To be executed after implementation, not as part of this design document.** Extends `src/analysis/fedtrace_detector.py`'s existing v1/v2 machinery (adding a `compute_historical_residual_calibration` function parallel in structure to `compute_longitudinal_calibration`, a `d(k,r)` computation step, and a `z_t_res` component) and applies it to both already-collected development traces. Eight detectors, computed identically to the exact formulas frozen in §§2–8 above, with no threshold or parameter adjustment based on the outcome:

1. **Magnitude-only** (`z_m`, unchanged from v1) — unchanged reference point.
2. **Peer-only** (`z_p`, unchanged from v1).
3. **v1 temporal** (`z_t`, absolute/longitudinal calibration) — the original "before."
4. **v2 temporal** (`z_t_rel`, same-round cross-sectional/robust) — the intermediate "before," carrying its own known flaw.
5. **v3 temporal residual** (`z_t_res`, this document) — the new candidate.
6. **FedTrace v1** (`max(z_m, z_p, z_t)`).
7. **FedTrace v2** (`max(z_m, z_t_rel)`).
8. **FedTrace v3** (`max(z_m, z_t_res)`).

For every detector/run combination (skipping combinations NOT EVALUABLE on `EXP_014`, i.e. anything requiring a `t`-derived component there), report the same full metrics suite used throughout this series: malicious-client attribution correctness, first anomalous round, confirmation round, onset localization error, detection delay, honest-client persistent false-positive count (per-client, explicitly naming which clients), and precision/recall/F1. **No threshold, persistence length, `T_ref`, or calibration-window adjustment of any kind is permitted after seeing these retrospective outcomes** — any remaining weakness is reported as a finding, exactly as v2's own weakness was, not silently corrected by further tuning.

## 11. Success Criteria

Pre-registered now, to be checked without modification once §10 is executed:

**`EXP_015`:**
- `client_002` is correctly attributed (the predicted malicious client, not merely a detected one).
- Onset estimate at or very near round 20 (consistent with §3's hand-verified `z_t_res` values, which give no reason to expect a materially different confirmation round than the already-excellent round 22 both v1 and v2 achieved via the temporal signal).
- **No honest client is confirmed before `client_002`** — this is the specific, binary condition v2 violated and v3 exists to fix; it is checked directly, not inferred from an aggregate false-positive count.
- Honest persistent false positives are zero, or materially reduced relative to both v1 (3 clients) and v2 (1 client, but attribution-breaking) — with explicit attention to *whether any remaining false positive, if any, occurs early enough to again precede `client_002`'s confirmation* (§12 makes this a failure condition in its own right, not merely a count to minimize).

**`EXP_014`:**
- Magnitude-based coverage remains intact (`z_m` unchanged, so this reduces to confirming no regression was introduced elsewhere in the implementation).
- No new false positives are introduced (trivially expected, since `EXP_014` has no `t`-derived signal to introduce one via, but confirmed rather than assumed).

**Across both development runs:**
- v3 must fix v2's specific `client_000` false-positive/attribution failure — not merely reduce some aggregate metric while leaving the underlying mechanism (small-sample same-round scale instability) present and capable of recurring at a different round or client.
- v3 must not merely relocate the false-positive problem to a different client or round (the same failure criterion v2 itself was found to have triggered, per §12 of the v2 plan, must not recur here).
- No attack-specific thresholds anywhere in the realized implementation.

## 12. Failure Criteria

FedTrace v3 will be considered **not to have worked**, and reopened for further design (not merely re-tuned), if any of the following holds after §10's retrospective check:

- Honest persistent false positives remain, at any round, for any client, on either development run.
- The wrong client is attributed on `EXP_015` (i.e. any client other than `client_002` is the earliest-confirmed).
- Onset localization for `client_002` becomes materially delayed relative to v1/v2's already-established round-22 confirmation (a regression in detection speed traded for stability would itself be a partial failure, worth reporting even if attribution is otherwise correct).
- The correction is found to work only because of a post-hoc, attack-specific choice (e.g. if `historical_scale_k`'s mean/stdev formulation, rather than a robust alternative, was quietly selected *because* it happened to produce a favorable result on these two runs rather than for the independently-stated precedent-based reason given in §3 — this document commits to the mean/stdev choice *before* running §10, specifically to foreclose this failure mode).
- The historical residual calibration `D_k` is found to be degenerate or unstable for any client on either run (e.g. fewer than 2 valid pre-`T_ref` observations, or a zero/near-zero `historical_scale_k` — not observed in §3's spot check, but not yet exhaustively ruled out either; §10 must check this explicitly for all 5 clients).
- v3 adds no operational value beyond simply running magnitude-only and v3-temporal-only as two entirely separate, uncombined detectors — i.e. if `max(z_m, z_t_res)` produces identical attribution/onset/FP outcomes to the two single detectors run independently, the combination step itself has not been shown to do anything (this would not invalidate `z_t_res` itself, but would mean the *combiner*, not the calibration fix, needs re-examination).

## 13. Development vs. Validation Boundary

**`EXP_014` and `EXP_015` remain development-only runs, exactly as designated in `research_plans/FEDTRACE_V2_TEMPORAL_CALIBRATION_PLAN.md` §9 and re-confirmed in `analysis/FEDTRACE_V2_RETROSPECTIVE_ANALYSIS.md` §12.** This document's own §3 hand-verification was performed by directly inspecting the exact rounds (`client_000`, rounds 9–11) that broke v2 on `EXP_015` — this is legitimate, targeted method-development practice, but it means `EXP_014`/`EXP_015` **cannot** subsequently be cited as independent evidence that FedTrace v3 works, for exactly the same reason v2 could not be validated against the same data used to diagnose it. §10's retrospective protocol, once executed, remains a **development-set sanity check**, not a validation result. **Once v3 is frozen (§16, this document), independent validation must use new, unseen runs/seeds — the already-anticipated seed-123/seed-2026 replications — with no further tuning of any kind, including no further revision of v3's formulas based on how those new runs turn out.** Any future write-up must state prominently which supporting evidence is development-only versus independently validated, and the two must never be merged into a single evidence count.

## 14. Independent Validation Gate

Before any Stage B GPU run (seed 123/2026 replication) is authorized, **all** of the following must hold:

1. **Detector implementation tests passing.** `tests/test_analysis_fedtrace_detector.py` (extended with v3-specific tests once implemented) passes in full, including regression tests confirming v1 and v2's own behavior remain unaffected by the v3 addition.
2. **Retrospective v3 development criteria passing.** §10's full comparison protocol, executed on both `EXP_014` and `EXP_015`, meets every criterion in §11 with no criterion in §12 triggered.
3. **Formulas/thresholds frozen.** §16's specification is the exact implementation, with no post-hoc adjustment made after seeing §10's results (verified by the same "commit before running" discipline already demonstrated in §3's spot check and required again, more fully, at implementation time).
4. **No remaining known methodological blocker.** Specifically: no client's `D_k` found degenerate on either development run (§12), no evidence that `max(z_m, z_t_res)` merely duplicates running the two components separately (§12), and no unresolved contamination scenario from §4 applicable to the two development runs as actually configured (both use a delayed onset at exactly `T_ref`, so this specific concern is inert for `EXP_014`/`EXP_015` themselves, but must be explicitly re-confirmed, not assumed, once §10 is run).

If any of these four is not met, Stage B remains not authorized, regardless of how favorable any individual metric looks in isolation — this mirrors exactly the discipline that caught v2's flaw before it reached Stage B.

## 15. Novelty Discipline

`z_t_res`'s own formula — a same-round drift-removal step feeding a per-client historical z-score — is not, by itself, claimed as a scientific contribution; it is a targeted, evidence-driven engineering correction to a specific, diagnosed instability, in the same spirit as v2's own (partially successful) correction to v1. Any eventual research contribution from this line of work continues to rest on the **complete traceability framework**, not on any single formula:

- **Heterogeneous-client calibration**, now spanning three distinct calibration strategies applied to three different stationarity/sample-size regimes (per-client longitudinal for `m`, once at `n≈19`; cross-sectional per-round for the shared-drift-removal step in `d(k,r)`, at `n=5`; per-client longitudinal again for the resulting residual's own scale, at `n≈19`) — a demonstrated, evidence-driven pattern of matching the calibration strategy to the statistical regime actually present, not a single one-size-fits-all rule.
- **Shared-drift removal**, the specific mechanism (§2's `c(r)`) that both v2 and v3 rely on and that directly targets this project's own repeatedly-observed secular-decline confound.
- **Client-specific temporal residual modeling**, the new element this document adds: recognizing that a *residual* (post-drift-removal) quantity can be longitudinally stationary even when the raw signal it was derived from is not, and calibrating accordingly.
- **Cross-view coverage** (magnitude + temporal, `z_p` retained for ablation), unchanged in spirit from v1/v2.
- **Onset localization**, **attacker attribution**, **false-positive control**, and **independent validation across attack families and seeds** — the same five items already listed in the v2 plan's own §13, carried forward unchanged, since v3 does not add or remove any of them from the framework's eventual scope.

No single one of these, and no formula in §§2–9, is presented as sufficient novelty on its own.

## 16. Frozen FedTrace v3 Formulation

**This is the single, complete, implementation-ready specification. No alternative formulations remain open.**

**Signals** (unchanged fields from v1/v2): `m(k,r) = update_l2_norm`, `p(k,r) = peer_mean_cosine_similarity` (ablation only, §6), `t(k,r) = self_prev_round_cosine`.

**Magnitude calibration** (unchanged from v1/v2): `mean_pre_m(k)`, `std_pre_m(k)` = sample mean/stdev of `m(k,r)` over `round_id < T_ref` (`T_ref = 20`), undefined if `< 2` observations or zero variance. `z_m(k,r) = |m(k,r) − mean_pre_m(k)| / std_pre_m(k)`.

**Same-round reference** (unchanged from v2): for round `r`, `S(r) = {j : t(j,r) is defined}`. `c(r) = median_{j in S(r)} t(j,r)`, undefined if `|S(r)| < 2`. No leave-one-out (§2).

**Residual:** `d(k,r) = c(r) − t(k,r)`, undefined wherever `c(r)` or `t(k,r)` is undefined.

**Historical residual calibration** (new): `D_k = {d(k,r) : round_id(r) in [1, T_ref)}`. `historical_center_k = mean(D_k)`, `historical_scale_k = stdev(D_k)` (sample standard deviation). Undefined if `|D_k| < 2` or `historical_scale_k == 0`.

**Temporal residual score:** `z_t_res(k,r) = max(0, [d(k,r) − historical_center_k] / historical_scale_k)`, undefined wherever the calibration or `d(k,r)` is undefined.

**Primary combined score:** `FedTraceV3Score(k,r) = max(z_m(k,r), z_t_res(k,r))` over whichever is defined; undefined if both are undefined. `z_p` never enters this score (§6).

**Flagging:** `(k,r)` is raw-anomalous iff `FedTraceV3Score(k,r)` is defined and `> 3` (unchanged).

**Persistence:** client `k` is detected starting at round `r*` iff `FedTraceV3Score(k,r) > 3` for every round in the contiguous range `[r*, r*+2]` (`P=3`, unchanged). Confirmation round `= r*+2`. Onset estimate `= r*`.

**Attribution:** predicted malicious client = earliest-confirmed client; ties broken by higher mean `FedTraceV3Score` over the persistent run, then `client_id` ascending (unchanged, matches `src/analysis/fedtrace_detector.py`'s existing `_select_predicted_client`). If no client is ever detected: "no malicious client identified."

**Missing-value handling, uniformly:** any undefined signal/statistic is excluded from `max()`, never substituted with 0 or any placeholder; if `FedTraceV3Score(k,r)` itself is undefined, that round is a streak-breaking gap for client `k`'s persistence computation (not automatically anomalous, not automatically continuing a prior streak).

**`T_ref = 20`** governs both `z_m`'s calibration window (unchanged meaning) and `D_k`'s calibration window (new, same numeric value, same "fixed once, never updated" freezing rule, §9). `c(r)`/`d(k,r)` themselves have no historical window (computed fresh every round, §2, unchanged from v2).

This specification is implementable in `src/analysis/fedtrace_detector.py` by adding: a `compute_common_round_reference` function (median-only version of v2's `compute_cross_sectional_robust_calibration`, or a direct reuse of its `center` output alone), a `compute_historical_residual_calibration` function (parallel in structure to `compute_longitudinal_calibration`, applied to precomputed `d(k,r)` values rather than a raw trace field), a `ScoredRowV3` dataclass, and `compute_all_scores_v3`/`apply_persistence_rule_v3`/`evaluate_detector_v3` following the exact same duplication-not-refactoring pattern already used for v2 (so that v1 and v2 remain provably untouched by this addition, exactly as v2 left v1 untouched).

## 17. Next-Step Decision

**Decision: (A) — the v3 formulation is sufficiently specified to implement offline next.**

Every formula, threshold, persistence rule, missing-value rule, tie-breaking behavior, and calibration-freezing rule in §16 is complete and unambiguous. §3's hand-verification — targeted specifically at the rounds already known to have broken v2, not an arbitrary sample — gives concrete, real-data evidence that the mechanism this document identifies (small-sample same-round scale instability) is genuinely absent from `z_t_res`'s construction: `client_000`'s score at the three previously-failing rounds drops from `3.91/12.35/4.45` (v2) to `0.37/0.28/0.52` (v3), while `client_002`'s true-positive detection remains immediate and, if anything, more sharply separated (`z` in the 30s). This is deliberately **not** a claim that §10's full retrospective protocol is already known to pass in every particular — §10 must still check all 5 clients across all rounds of both runs, and §14's independent-validation gate lists exactly what remains to be confirmed before any further step — only that no open design question remains that would block writing that implementation. Per this task's explicit restriction, no GPU experiment is recommended or required before that offline implementation-and-retrospective-check step.
