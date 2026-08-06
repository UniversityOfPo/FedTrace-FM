# EXP_007 — Non-IID Dirichlet(α=0.1) Analysis (EXP_004 vs. EXP_006 vs. EXP_007)

**Analysis date:** 2026-08-05
**Subject experiments:**
- `EXP_004` — FedAvg IID, 50 rounds (frozen reference protocol)
- `EXP_006` — FedAvg Non-IID, Dirichlet α=0.5, 50 rounds
- `EXP_007` — FedAvg Non-IID, Dirichlet α=0.1, 50 rounds

**Data sources (verified, no invented values):**
- `results/federated/cifar100_fedavg_iid_50rounds/round_history.csv`,
  `summary.json`, `final_test_metrics.json`
- `results/federated/cifar100_fedavg_dirichlet_a05_50rounds/round_history.csv`,
  `summary.json`, `final_test_metrics.json`, `client_partition_summary.csv`
- `results/federated/cifar100_fedavg_dirichlet_a01_50rounds/round_history.csv`,
  `summary.json`, `final_test_metrics.json`, `client_partition_summary.csv`
- `experiments/EXP_006_FedAvg_Dirichlet_A05.md`,
  `analysis/EXP_006_Dirichlet_A05_Analysis.md` (prior comparison
  methodology reused here for consistency)

Round-block marginal-gain figures are simple arithmetic (subtraction and
division) applied directly to the verified `validation_top1`/
`validation_loss` columns of each run's `round_history.csv`, using the
same previous-block-endpoint-baseline convention established in
`analysis/EXP_004_Convergence_Analysis.md` and reused in
`analysis/EXP_005_Cosine_Scheduler_Analysis.md` and
`analysis/EXP_006_Dirichlet_A05_Analysis.md`.

This document performs analysis only. No code, config, test, README, or
Research Database file was created or modified in the process of
producing it.

---

## 1. Controlled-Variable Confirmation

Per `configs/cifar100_fedavg_dirichlet_a01_50rounds.yaml`, `EXP_007`
changes only `dirichlet_alpha` (`0.5 → 0.1`) relative to `EXP_006`; every
other field — including `partition: dirichlet` itself — is identical.
Relative to `EXP_004`, both `EXP_006` and `EXP_007` change only
`partition` (`iid → dirichlet`) plus that variable's required companion
`dirichlet_alpha`. All three runs' own `summary.json` confirm every other
tracked setting matches: `num_clients=5`, `clients_per_round=5`,
`minimum_updates=5`, `communication_rounds=50` (all completed),
`local_epochs=1`, `aggregation_strategy=fedavg`, `optimizer=sgd`,
`learning_rate=0.01`, `batch_size=32`, `seed=42`, `total_parameters=11,220,132`,
same GPU. As with `EXP_006`, round 0 is **not** numerically identical
across the three runs — each partitioning strategy/`alpha` produces a
genuinely different assignment of samples to clients even at a matched
seed (`EXP_004` round 0 val top-1: 1.86%; `EXP_006`: 1.36%; `EXP_007`:
1.06% — a real, expected difference in initial client composition, not a
data error).

---

## 2. Three-Way Comparison: IID vs. α=0.5 vs. α=0.1 — Final Test Metrics

| Metric | EXP_004 (IID) | EXP_006 (α=0.5) | EXP_007 (α=0.1) |
|---|---:|---:|---:|
| Final test loss | 1.1049091817855834 | 1.1717977018356323 | 1.3758617393493653 |
| Final test top-1 | 68.52% | 66.22% | **59.68%** |
| Final test top-5 | 91.27% | 90.63% | 88.65% |
| Macro precision | 0.6901 | 0.6785 | 0.6509 |
| Macro recall | 0.6850 | 0.6625 | 0.5965 |
| Macro F1 | 0.6814 | 0.6589 | 0.5876 |
| Balanced accuracy | 0.6850 | 0.6625 | 0.5965 |
| Best validation round | 49 (final) | 46 (not final) | **49 (final)** |
| Total experiment time | 4,055.95 s | 4,241.69 s | 4,136.48 s |

**`EXP_007` is the first federated result since `EXP_003` to fall *below*
the centralized `EXP_001` reference** (62.71% test top-1): 59.68% is
**−3.03 percentage points** under `EXP_001`, reversing the pattern set by
`EXP_004` (+5.81pp), `EXP_005` (+2.41pp), and `EXP_006` (+3.51pp), all of
which exceeded it. Sufficiently strong client heterogeneity, at a fixed
50-round budget, erases the round-budget advantage federated training had
built up over the earlier experiments in this series.

---

## 3. Quantifying Degradation Across All Metrics

| Metric | EXP_004 → EXP_006 (Δ, α ∞→0.5) | EXP_006 → EXP_007 (Δ, α 0.5→0.1) | EXP_004 → EXP_007 (Δ, total) |
|---|---:|---:|---:|
| Test loss | +0.0669 | **+0.2041** | +0.2710 |
| Test top-1 | −2.30pp | **−6.54pp** | −8.84pp |
| Test top-5 | −0.64pp | **−1.98pp** | −2.62pp |
| Macro precision | −0.0116 | −0.0276 | −0.0392 |
| Macro recall | −0.0225 | −0.0660 | −0.0885 |
| Macro F1 | −0.0225 | −0.0713 | −0.0938 |
| Balanced accuracy | −0.0225 | −0.0660 | −0.0885 |

**The relationship between `alpha` and accuracy is not linear — it
accelerates.** The `EXP_006 → EXP_007` step (α: 0.5→0.1) costs roughly
2.8x more test-top-1 accuracy (−6.54pp) than the `EXP_004 → EXP_006` step
(α: ∞→0.5, −2.30pp) cost, despite both steps moving `alpha` in the same
direction. Every metric shows the same pattern: the second heterogeneity
increase is substantially more damaging than the first. This directly
revises the severity classification in
`analysis/EXP_006_Dirichlet_A05_Analysis.md` § 8 (which called `alpha=0.5`
"mild") — `alpha=0.1`'s degradation is **moderate to severe**: an 8.84pp
(12.9% relative) test top-1 reduction from the IID reference, and — unlike
`EXP_006` — a reversal of `EXP_004`'s advantage over the centralized
baseline. It is not, however, a collapse: `EXP_007` still reaches 59.68%
test top-1, well above chance (1%) and above `EXP_003`'s 54.69%
(IID, 20-round budget), and all 50 rounds completed with 5/5 successful
clients and 0 rejected updates — the pipeline remained stable and
functional throughout.

---

## 4. Convergence Behavior

**Validation top-1 at decade-block boundaries** (`round_history.csv`):

| Round | 0 | 9 | 19 | 29 | 39 | 49 |
|---|---:|---:|---:|---:|---:|---:|
| EXP_004 (IID) | 1.86% | 37.54% | 54.22% | 61.10% | 65.50% | 68.68% |
| EXP_006 (α=0.5) | 1.36% | 36.00% | 51.60% | 58.86% | 63.44% | 65.36% |
| EXP_007 (α=0.1) | 1.06% | 30.24% | 46.00% | 52.44% | 55.54% | 59.22% |

**Average validation-top-1 gain per round, by decade block**
(previous-block endpoint as baseline):

| Block | EXP_004 (pp/round) | EXP_006 (pp/round) | EXP_007 (pp/round) |
|---|---:|---:|---:|
| Rounds 0–9 | ≈3.964 | ≈3.849 | ≈3.242 |
| Rounds 10–19 | ≈1.668 | ≈1.560 | ≈1.576 |
| Rounds 20–29 | ≈0.688 | ≈0.726 | ≈0.644 |
| Rounds 30–39 | ≈0.440 | ≈0.458 | ≈0.310 |
| Rounds 40–49 | ≈0.318 | ≈0.192 | **≈0.368** |

**Average validation-loss reduction per round, by decade block:**

| Block | EXP_004 (loss/round) | EXP_006 (loss/round) | EXP_007 (loss/round) |
|---|---:|---:|---:|
| Rounds 0–9 | ≈0.2649 | ≈0.2680 | ≈0.2359 |
| Rounds 10–19 | ≈0.0707 | ≈0.0669 | ≈0.0746 |
| Rounds 20–29 | ≈0.0288 | ≈0.0257 | ≈0.0242 |
| Rounds 30–39 | ≈0.0167 | ≈0.0182 | ≈0.0149 |
| Rounds 40–49 | ≈0.0102 | ≈0.0079 | ≈0.0092 |

**`EXP_007`'s block-to-block pattern is not uniformly worse than
`EXP_004`'s, and it is not monotonically shrinking the way `EXP_004`'s and
`EXP_006`'s are.** `EXP_007` trails in blocks 0–9, 20–29, and 30–39, but is
briefly *ahead* of `EXP_004` in blocks 10–19 (1.576 vs. 1.668 — nearly
matched) and 40–49 (0.368 vs. 0.318). Both `EXP_004` and `EXP_006` show a
smooth, block-over-block *decrease* in marginal top-1 gain with no
reversal; `EXP_007`'s marginal gain **increases** from block 30–39
(≈0.310pp/round) to block 40–49 (≈0.368pp/round) — the only reversal of
this kind observed in any of the three runs. Validation *loss*, by
contrast, keeps shrinking smoothly for `EXP_007` in every block (0.2359 →
0.0746 → 0.0242 → 0.0149 → 0.0092) with no equivalent reversal — the
top-1 uptick in the final block is a discretized-metric effect layered on
top of an otherwise-ordinary diminishing-returns loss curve, not evidence
of a qualitatively different convergence regime.

---

## 5. Late-Round Efficiency (Rounds 40–49)

Round-by-round validation top-1/loss for `EXP_007`:

| Round | Val Loss | Δ Loss | Val Top-1 | Δ Top-1 |
|---:|---:|---:|---:|---:|
| 39 | 1.5158 | — | 55.54% | — |
| 40 | 1.5151 | −0.0007 | 56.26% | +0.72pp |
| 41 | 1.5010 | −0.0141 | 57.08% | +0.82pp |
| 42 | 1.4919 | −0.0091 | 56.84% | −0.24pp |
| 43 | 1.4319 | −0.0599 | 58.24% | +1.40pp |
| 44 | 1.4529 | +0.0210 | 57.62% | −0.62pp |
| 45 | 1.4606 | +0.0077 | 57.26% | −0.36pp |
| 46 | 1.4431 | −0.0174 | 58.38% | +1.12pp |
| 47 | 1.4437 | +0.0005 | 57.78% | −0.60pp |
| 48 | 1.4133 | −0.0304 | 58.48% | +0.70pp |
| 49 | 1.4239 | +0.0107 | 59.22% | +0.74pp |

Net movement over the window is positive on both metrics (loss −0.0919,
top-1 +3.68pp), and — unlike `EXP_005` (best round 45 of 50) and `EXP_006`
(best round 46 of 50) — **`EXP_007`'s final round (49) is also its best
round**, matching `EXP_004`'s pattern. This is **not** evidence that
strong heterogeneity produces a "healthier" late-training trajectory: it
is more directly explained by `EXP_007` being further from any accuracy
ceiling at round 49 (59.22% vs. `EXP_004`'s 68.68%) and therefore having
more headroom left to improve, not by the training dynamics themselves
being more stable. Indeed, the window shows **four negative single-round
top-1 transitions** (42→ , 44→, 45→, 47→) out of nine, versus two in
`EXP_004`'s equivalent window
(`analysis/EXP_004_Convergence_Analysis.md` § 5) — `EXP_007`'s round-to-round
trajectory is *more* volatile, not less, consistent with stronger
heterogeneity producing noisier per-round aggregation outcomes even while
the net trend remains upward.

---

## 6. Client Sample-Count Imbalance

| Client | EXP_004 (IID) | EXP_006 (α=0.5) | EXP_007 (α=0.1) |
|---|---:|---:|---:|
| client_000 | 9,000 | 9,696 | 10,883 |
| client_001 | 9,000 | 9,079 | 8,003 |
| client_002 | 9,000 | 8,487 | 8,868 |
| client_003 | 9,000 | 9,255 | 8,574 |
| client_004 | 9,000 | 8,483 | 8,672 |
| **Total** | **45,000** | **45,000** | **45,000** |
| Coefficient of variation | 0% | ≈5.2% | **≈10.9%** |

`EXP_007`'s total-sample-count imbalance across clients (coefficient of
variation ≈10.9%) is roughly double `EXP_006`'s (≈5.2%), and its range
(8,003–10,883 = 2,880 samples) is about 2.4x `EXP_006`'s range (1,213
samples). As with `EXP_006`, this total-count imbalance is a downstream
side effect of per-class Dirichlet allocation, not a direct measurement of
per-client class-distribution skew (`client_partition_summary.csv` still
records only per-client totals — the same instrumentation gap flagged in
`analysis/EXP_006_Dirichlet_A05_Analysis.md` § 6 remains open for
`EXP_007`). That said, the total-count imbalance roughly doubling in
lockstep with `alpha` dropping from 0.5 to 0.1 is at least directionally
consistent with — though not proof of — a correspondingly larger
per-class skew.

---

## 7. What Do These Results Imply About Client Heterogeneity?

- **FedAvg's sensitivity to heterogeneity in this pipeline is real and
  non-linear.** A moderate skew (α=0.5) cost only 2.30pp — mild, as
  classified in `analysis/EXP_006_Dirichlet_A05_Analysis.md`. A stronger
  skew (α=0.1) cost an additional 6.54pp — large enough to erase the
  entire round-budget advantage that made `EXP_004` exceed the
  centralized baseline. This is consistent with the general FL-literature
  expectation that FedAvg degrades gracefully under mild heterogeneity but
  more sharply once heterogeneity crosses into a more severe regime — this
  project's own data now shows a concrete instance of that transition
  between α=0.5 and α=0.1.
- **The effect is on convergence *quality*, not pipeline stability.**
  All 50 rounds completed successfully at both α values, with 5/5 client
  updates accepted every round — the degradation shows up in accuracy and
  in noisier round-to-round dynamics (§ 5), not in outright failure.
- **Directly relevant to Phase 4/5 (`research_plans/PHASE4_RESEARCH_STRATEGY.md`
  § 5):** the "honest heterogeneity noise floor" that any future
  attribution mechanism must not confuse with malicious behavior is now
  known to be *alpha-dependent* and non-trivial in size at α=0.1 —
  round-to-round validation swings of over a percentage point in either
  direction (§ 5) are a normal feature of honest clients at this
  heterogeneity level, not necessarily a signal of anything adversarial.
- **This crosses the evidence bar `research_plans/PHASE4_RESEARCH_STRATEGY.md`
  § 7 set for FedProx**: that memo specified that a "substantially larger
  degradation than `EXP_006`'s mild 2.30pp gap" would justify implementing
  a heterogeneity-mitigation strategy. `EXP_007`'s 8.84pp gap (and its
  reversal of the centralized comparison) meets that bar — see § 9 for how
  this should be sequenced against replication.

---

## 8. What Can and Cannot Be Concluded From a Single Seed

**Can be concluded (directly verified, this analysis):** at seed 42,
reducing `dirichlet_alpha` from 0.5 to 0.1 (all else identical) produced a
substantially larger accuracy and convergence-quality degradation than the
0.5 step did, reversing `EXP_004`'s advantage over the centralized
baseline, with the specific numeric gaps reported in §§ 2–3 above.

**Cannot be concluded:**
- That 8.84pp is the "true," seed-independent cost of α=0.1 heterogeneity
  for this pipeline. No repeated-seed trial exists for `EXP_004`,
  `EXP_006`, or `EXP_007` — the exact magnitude of every reported gap in
  this document has not been checked against ordinary run-to-run noise
  (CUDA nondeterminism, or the specific random partition drawn at seed 42
  versus another seed at the same `alpha`).
- That the α=0.5→0.1 acceleration pattern (§ 3) is a smooth, predictable
  function of `alpha` in general — two points establish a trend, not a
  curve; the true shape of the accuracy-vs-heterogeneity relationship
  between, at, and beyond these two `alpha` values is unknown.
- That `EXP_007`'s specific qualitative features (final round = best
  round, block 40–49's marginal-gain uptick, higher round-to-round
  volatility) are stable properties of α=0.1 heterogeneity rather than
  artifacts of this one partition draw.
- That any of this generalizes beyond CIFAR-100 + ResNet-18 + FedAvg + 5
  clients + a 50-round, `local_epochs=1` budget — this project's only
  tested configuration to date.

The verified, narrow conclusion is: **at seed 42, Dirichlet(0.1)
partitioning produced a moderate-to-severe accuracy and convergence-quality
degradation relative to both `EXP_004`'s IID reference and `EXP_006`'s
Dirichlet(0.5) result** — not a general characterization of FedAvg's
heterogeneity sensitivity.

---

## 9. Next Scientifically Justified Experiment

Given § 8, the most scientifically disciplined next step is **not** to
immediately implement FedProx (even though § 7 shows its evidence bar is
now nominally met) — this project's own standing practice, restated in
every prior analysis document, is to not generalize from a single seed
before investing in a larger implementation effort. `EXP_007`'s result is
also the first *qualitative* reversal in this experiment series (falling
back below the centralized baseline), which raises the stakes on ruling
out a seed-specific artifact before acting on it.

**Recommended: `EXP_008` — Dirichlet(α=0.1) replication at a second
seed.** Single controlled-variable change relative to `EXP_007`: `seed`
only (a new value, e.g. a seed not yet used in this project's federated
experiments), with `dirichlet_alpha=0.1` and every other field identical
to `EXP_007`. This requires no new code or partitioning logic — only a
new config, exactly analogous to how `EXP_003` re-verified `EXP_002`'s
undertraining hypothesis with a single-variable change.

**Falsifiable framing:** if `EXP_008` reproduces a test top-1 accuracy
within a few points of `EXP_007`'s 59.68% and similarly falls below (or
close to) `EXP_001`'s 62.71% centralized reference, that corroborates
§ 7's conclusion and directly justifies proceeding to implement FedProx
per `research_plans/PHASE4_RESEARCH_STRATEGY.md` § 7. If `EXP_008`
instead lands close to `EXP_006`'s territory (mild degradation, clearly
above the centralized reference), that would indicate `EXP_007`'s result
is highly seed-sensitive — an important finding in its own right, and one
that would argue for characterizing variance (multiple seeds per `alpha`)
before drawing any further conclusions about the heterogeneity-accuracy
relationship, rather than proceeding directly to FedProx.
