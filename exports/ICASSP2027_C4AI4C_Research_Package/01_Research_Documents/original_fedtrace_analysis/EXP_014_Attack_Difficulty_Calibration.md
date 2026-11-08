# EXP_014 — Attack Difficulty Calibration (Analytical, Offline)

**Analysis date:** 2026-08-08
**This is a purely analytical study using already-collected honest/attack
update-norm statistics from `client_round_trace.csv`
(`results/federated/cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed42/`)
and the paired honest baseline `EXP_007`. No experiment was rerun, no
config was created, no attack strength or detector code was modified, and
no FedTrace algorithm is designed here.**

**Goal:** determine, analytically, whether a lower-`rho` Gaussian attack
could produce a non-trivial attribution benchmark, or whether a
norm-preserving attack family is scientifically necessary to meaningfully
stress attribution beyond what `analysis/EXP_014_Norm_Aware_Attribution_Study.md`
already showed `rho=1.0` fails to do (it is solved almost perfectly by a
three-parameter z-score rule).

---

## 1. Geometry of the Current Gaussian Attack

For honest delta `delta` and independently-sampled noise `n` constructed
(per `src/attacks/traceability.py::generate_gaussian_attack_noise`) so
that `‖n‖₂ = rho · ‖delta‖₂`:

```
‖delta + n‖₂² = ‖delta‖₂² + 2⟨delta, n⟩ + ‖n‖₂²
```

In an 11.2M-parameter space, an independently-drawn random vector `n` and
a fixed vector `delta` are, with overwhelming probability, close to
orthogonal — the inner product `⟨delta, n⟩` concentrates near zero
relative to `‖delta‖₂·‖n‖₂` as dimensionality grows (a standard
concentration-of-measure fact for high-dimensional random vectors, not
specific to this project). Dropping that term:

```
‖delta + n‖₂² ≈ ‖delta‖₂² + ‖n‖₂² = ‖delta‖₂²·(1 + rho²)

‖delta + n‖₂ / ‖delta‖₂ ≈ √(1 + rho²)
```

**This is explicitly an approximation, not an exact identity for every
individual update.** It holds in expectation / with high probability
under the orthogonality assumption; for any single, specific `(delta, n)`
pair the inner-product term is a genuine nonzero random quantity, and
`delta` itself is not a fixed reference vector but the client's own,
round-varying honest update — both are sources of deviation from the
idealized formula, quantified next.

---

## 2. Comparing the Prediction Against Observed `EXP_014` Data at `rho=1.0`

**Theoretical prediction:** `√(1 + 1.0²) = √2 ≈ 1.41421`, i.e. **+41.42%**
norm inflation.

**Observed (`client_round_trace.csv`, `client_002`):**

| Comparison | Ratio | Inflation |
|---|---:|---:|
| Immediate boundary: round 19 (5.9146) → round 20 (8.4816) | 1.4339 | **+43.39%** |
| Full-window mean: pre-attack mean (6.0034) → post-attack mean (8.8253) | 1.4701 | **+47.01%** |

**Assessment: broadly consistent with the theoretical prediction, running
modestly higher** (by roughly 2–6 percentage points of inflation,
depending on which observed comparison is used). Two known,
non-alarming sources fully account for this gap without needing to
question the attack's correctness (already independently verified
correct by `analysis/PHASE5_Implementation_Verification.md`):

1. **The reference norm used is a proxy, not the true counterfactual.**
   Neither comparison can access `client_002`'s actual *un-attacked*
   honest delta norm for rounds 20–49 (the attack overwrites it before
   logging) — round 19's value, or the rounds 0–19 mean, are the closest
   available proxies, but honest update norms drift slightly round to
   round even without any attack (the four honest clients' own
   pre-vs-post-window means move by −2.8% to +1.3%, per
   `analysis/EXP_014_Norm_Aware_Attribution_Study.md` §2) — a real
   *EXP_007*-observed source of a few percentage points of discrepancy on
   its own.
2. **Finite-dimensional, single-draw noise is not exactly orthogonal to
   delta.** The `√(1+ρ²)` formula is an asymptotic/high-probability
   statement; any one realized noise draw carries a nonzero (if small)
   inner-product term.

Neither source is large enough to be concerning; the observed inflation
is the same order of magnitude as, and within roughly 10% (relative) of,
the theoretical prediction.

---

## 3. Pre-Attack (Rounds 0–19) Honest Update-Norm Reference

Using `update_l2_norm` for rounds 0–19 only (identical source data and
values as `analysis/EXP_014_Norm_Aware_Attribution_Study.md` §2):

**`client_002` (the malicious client's own honest reference, before it is
ever attacked):**

| Statistic | Value |
|---|---:|
| Mean | 6.0034 |
| Sample standard deviation | 0.2690 |
| 3-sigma upper threshold (`mean + 3·std`) | **6.8104** |
| Coefficient of variation (`std / mean`) | **4.48%** |

**All five clients, for comparison:**

| Client | Pre-attack mean | Pre-attack std | 3σ threshold | CV |
|---|---:|---:|---:|---:|
| `client_000` | 6.5156 | 0.3150 | 7.4606 | 4.84% |
| `client_001` | 5.8221 | 0.4836 | 7.2729 | 8.31% |
| `client_002` | 6.0034 | 0.2690 | 6.8104 | 4.48% |
| `client_003` | 6.0395 | 0.3421 | 7.0658 | 5.66% |
| `client_004` | 5.9329 | 0.2770 | 6.7639 | 4.67% |

Natural, purely-honest round-to-round variability (as measured by CV) is
small — 4.5% to 8.3% of each client's own mean — but not negligible; this
range is the yardstick Sections 5–6 compare candidate attack strengths
against.

---

## 4. Approximate `rho` Boundary for `client_002`'s 3-Sigma Threshold

Solving `√(1 + rho²) · honest_reference_norm = norm_detection_threshold`
for `rho`, using `client_002`'s own pre-attack mean as
`honest_reference_norm` (6.0034) and its own 3-sigma threshold as
`norm_detection_threshold` (6.8104) — both choices made explicit, per
this task's instruction, and both drawn only from Section 3's already-
computed pre-attack statistics:

```
√(1 + rho²) = 6.8104 / 6.0034 = 1.13444
1 + rho²    = 1.28696
rho²        = 0.28696
rho         ≈ 0.5357
```

**Approximate boundary: `rho ≈ 0.54`.** Below this value, the *expected*
(mean) attacked update norm is predicted to fall **below** `client_002`'s
own 3-sigma detection threshold; above it, the expected attacked norm is
predicted to exceed that threshold. This is a statement about the
*expected/mean* effect at a given `rho`, not a guarantee for any
individual round — Section 6 addresses the natural per-round scatter
around this boundary.

---

## 5. Analytical Grid: Predicted Norm Inflation by `rho`

**This is analysis only — no experiments were run or proposed at these
values.**

| `rho` | `√(1 + rho²)` | Predicted norm inflation |
|---:|---:|---:|
| 1.00 | 1.41421 | **+41.42%** |
| 0.50 | 1.11803 | **+11.80%** |
| 0.25 | 1.03078 | **+3.08%** |
| 0.10 | 1.00499 | **+0.50%** |

Cross-referencing against Section 4's boundary (`rho≈0.54`, requiring
+13.44% inflation to reach `client_002`'s own 3-sigma threshold from its
mean):

- `rho=1.0` (**+41.42%**) is far above the boundary — consistent with
  `EXP_014`'s observed, essentially trivial detection (§2 above;
  `analysis/EXP_014_Norm_Aware_Attribution_Study.md` §§4–6, 10).
- `rho=0.5` (**+11.80%**) sits **just below** the +13.44% boundary — an
  *expected-value* near-miss, not a confident prediction either way for
  any single round (Section 6).
- `rho=0.25` (**+3.08%**) is smaller than every honest client's own CV
  (4.48%–8.31%, Section 3) — analytically indistinguishable, on average,
  from ordinary honest round-to-round variability.
- `rho=0.10` (**+0.50%**) is far below any honest client's natural
  variability — analytically invisible to a norm-based signal.

---

## 6. Is There a Plausible Lower-`rho` Region for a Non-Trivial Benchmark?

**Analytically, yes — a plausible candidate region exists, roughly
`rho ∈ [0.25, 0.5]`, but this is not experimentally validated by this
document.**

Reasoning: at `rho=0.5`, the predicted mean norm inflation (+11.80%) is
larger than every honest client's own CV (max 8.31%, `client_001`) — so
the effect is not certainly invisible — but it sits below the
`+13.44%` needed to clear `client_002`'s own 3-sigma threshold in
expectation. Combined with the natural per-round scatter already measured
(pre-attack std of 0.2690 around a mean of 6.0034 for `client_002` alone,
before any attack), this is exactly the kind of regime where a
single-round `z_norm > 3` rule would be expected to behave inconsistently
round to round — sometimes crossing the threshold, sometimes not —
rather than firing with the overwhelming, every-single-round margin
`rho=1.0` produced (§2 above; every one of 30 post-attack rounds at
`z_norm > 9`). At `rho=0.25`, the predicted effect (+3.08%) is smaller
than the honest CV range entirely, suggesting a genuinely hard-to-detect
regime by this signal.

**Explicit caveat, as required:** this is a geometric prediction from
already-measured honest statistics, not a demonstrated experimental
result. Actual behavior at these `rho` values would depend on the
realized noise draws, the realized honest deltas at each specific round
(which this analysis cannot observe for a hypothetical unrun
configuration), and possibly on effects the orthogonality approximation
does not capture. No claim is made that `rho=0.5` or `rho=0.25` would, if
actually run, produce any particular detection outcome — only that the
*expected-value* geometry makes this range a scientifically motivated
candidate to investigate, not a settled conclusion.

---

## 7. Conceptual Examination: A Norm-Preserving Directional Attack

Consider (conceptually only — **not implemented, not designed in
detail**) a malicious update `delta_attack` constructed to satisfy
approximately `‖delta_attack‖₂ ≈ ‖delta_honest‖₂` while altering the
update's *direction* (e.g., a rotation, a projection toward some other
target direction, or a redirected component orthogonal to the honest
gradient combined with a compensating reduction elsewhere to hold the
norm fixed).

**Why this would specifically defeat a norm-reliant detector:** both
diagnostics evaluated so far in this study series key off magnitude —
the raw `update_l2_norm` (`analysis/EXP_014_Gaussian_Attack_Traceability_Analysis.md`)
and the per-client `z_norm` (`analysis/EXP_014_Norm_Aware_Attribution_Study.md`)
are, by construction, blind to direction entirely. A norm-preserving
attack, by definition, produces `z_norm ≈ 0` on every round, for any
`rho`-style strength parameter — there is no norm signature to detect,
regardless of how "strong" the directional change is. This is not a
matter of finding the right threshold (as Sections 4–6 explore for the
Gaussian family); it removes the entire signal class both baselines rely
on.

**Why this is a stronger test of *temporal* traceability specifically:**
with magnitude no longer informative, any viable detector must instead
recognize a *change in behavior over time* for one specific client — e.g.
a shift in the *direction* of that client's own successive updates,
or a change in how that client's update relates to its peers, that
begins at a specific round and persists. This directly targets the
flagship requirement to identify **both** the attacker **and** the
specific onset round through a genuinely temporal/behavioral signature,
rather than a single-round magnitude outlier check — a materially
different (and harder) attribution problem than anything the Gaussian
family, at any `rho`, has been shown to pose.

---

## 8. Comparing the Two Candidate Next Challenges Against the Flagship Objective

The flagship objective (`FEDTRACE_FM_MASTER.md` Research Goal;
`research_plans/PHASE5_Malicious_Client_Traceability_Plan.md` §2) is to
distinguish malicious client behavior from honest strong Non-IID
heterogeneity **while identifying both the attacker and the attack-onset
round**.

**A. Lower-`rho` Gaussian attack (Sections 4–6).** Remains, structurally,
a *magnitude*-based attack at every `rho` — only its detectability
threshold moves. This project's own accumulated evidence
(`analysis/EXP_014_Gaussian_Attack_Traceability_Analysis.md` §4) already
shows that *honest* Non-IID heterogeneity under this project's
Dirichlet(α=0.1) setup manifests primarily as a **directional**
phenomenon (large, shared, round-dependent drift in peer-cosine
similarity, near-identical across all five clients) far more than a
**magnitude** phenomenon (pre-attack norm CVs of only 4.5%–8.3% across
honest clients, Section 3). A lower-`rho` Gaussian attack calibrates
*how hard* the same, already-solved magnitude problem is — it does not
introduce the directional confound that this project's own honest
baseline is actually dominated by, and so does not stress-test
attribution against the kind of variation (heterogeneity-driven
directional drift) that is the flagship objective's central named
concern.

**B. Norm-preserving directional attack (Section 7).** Directly targets
that gap: it forces any detector to separate malicious *directional*
deviation from the already-quantified, large, honest *directional* drift
every client naturally exhibits — precisely the distinction the flagship
objective, and `PHASE4_RESEARCH_STRATEGY.md` §5's research question,
were framed around from the start ("does a malicious client's update
produce a statistical signature distinguishable from the background
divergence that ordinary, honest Non-IID heterogeneity already
produces"). It is the harder, and more relevant, next challenge — not
merely a re-tuned version of the one just solved.

---

## 9. The Round-0 Artifact and a Clean Evaluation Convention

`analysis/EXP_014_Norm_Aware_Attribution_Study.md` §5 documented that
every client's round 0 spuriously exceeds `z_norm > 3` when scored
against reference statistics (`mean_pre`, `std_pre`) computed **from a
window that includes round 0 itself** — a boundary artifact of
self-referential scoring, not a real detection failure (no post-attack
round was ever affected by it).

**Clarification for future operational evaluation:** a reference window
used to *fit* a detector's statistics should never also be the window the
same detector is *scored against* — doing so guarantees the single most
atypical point in that window (here, the shared round-0 initialization
transient) will appear anomalous relative to its own contribution to the
very statistics it is compared to. **Recommended convention (documented
here only — no code is changed by this document):** designate rounds
0–19 purely as a *calibration/burn-in* window used only to compute
`mean_pre`/`std_pre`, and begin *scoring* (i.e., the first round eligible
to be flagged at all, honest or otherwise) at round 20 onward — the first
round genuinely independent of the calibration data. Under this
convention, the round-0 artifacts recorded in the prior study would not
occur, and precision, in Section 10 of that study, would read 100%
without the "raw vs. excluding round-0" caveat.

---

## 10. Decision: Exactly ONE Next Benchmark Direction

**Chosen: B — design a norm-preserving directional attack as the next
benchmark challenge.**

Justification, on scientific difficulty and relevance grounds (not
convenience — B is clearly the more involved design effort of the two):

- Section 6 shows a lower-`rho` region is only a **plausible, unvalidated**
  candidate for non-trivial difficulty, and even if validated, it would
  still be solvable by a norm-based signal in principle (Section 8, A) —
  it recalibrates difficulty without changing what kind of problem is
  being solved.
- Section 8's comparison shows only the norm-preserving direction (B)
  actually confronts the flagship objective's named central concern —
  separating malicious signal from honest, heterogeneity-driven
  *directional* drift, which this project has already shown (via the
  failed cosine detector) to be large and non-trivial on its own.
- Choosing the analytically "easier to reach" option (A, a lower `rho`
  value within the existing, already-implemented attack family) would be
  the more convenient next step, but Section 8's analysis indicates it
  would not, on its own, produce a benchmark that meaningfully tests
  attribution against the specific confound (honest directional
  heterogeneity) the flagship objective is centrally concerned with —
  which is why it is not the chosen direction here.

**No broad parameter sweep is recommended.** This document evaluates
exactly four `rho` values analytically (Section 5) purely to establish
the boundary geometry, not as a proposed experimental grid. **No attack
is implemented and no experiment config is created by this document.**

---

## Files Created

- `analysis/EXP_014_Attack_Difficulty_Calibration.md`

## Files Modified

- None.

## Commands Executed

- None. Only Read and Write tool calls were used.
