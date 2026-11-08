# EXP_014 — Gaussian Malicious-Update Attack & Naive-Detector Forensic Analysis (Paired vs. EXP_007, Seed 42)

**Analysis date:** 2026-08-08
**Subject experiments (paired, seed 42 only):**
- `EXP_007` — Honest FedAvg, Dirichlet α=0.1, 50 rounds, seed 42 (no attack)
- `EXP_014` — Same protocol + `gaussian_update_noise` attack on `client_002`, `attack_start_round=20`, `attack_strength_rho=1.0`

**Data sources (verified, no invented values):**
- `results/federated/cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed42/`
  (`summary.json`, `final_test_metrics.json`, `round_history.csv`,
  `client_round_trace.csv`, `client_partition_summary.csv`)
- `results/federated/cifar100_fedavg_dirichlet_a01_50rounds/`
  (`summary.json`, `final_test_metrics.json`, `round_history.csv`,
  `client_partition_summary.csv`)
- `research_plans/PHASE5_Malicious_Client_Traceability_Plan.md`,
  `analysis/PHASE5_Implementation_Verification.md`

This is a forensic analysis of an already-completed run — no experiment
was executed to produce it, and no code, config, test, README, or
Research Database file was created or modified.

---

## 1. Pairing Verification

Both runs' `summary.json` and `client_partition_summary.csv` confirm an
exact pairing on every scientific setting: `dataset="CIFAR-100"`,
`partition="dirichlet"` (`dirichlet_alpha=0.1` per both configs),
`model="resnet18_cifar"`, `total_parameters=11,220,132`, `num_clients=5`,
`clients_per_round=5`, `minimum_updates=5`,
`communication_rounds_configured/_completed=50/50`, `local_epochs=1`,
`optimizer="sgd"`, `learning_rate=0.01`, `batch_size=32`,
`aggregation_strategy="fedavg"`, `seed=42`, same GPU (NVIDIA GeForce
RTX 5060 Laptop GPU).

**Partition realization is identical**, not merely configured
identically: `client_partition_summary.csv` reports the exact same
per-client sample counts in both runs — `client_000`: 10,883;
`client_001`: 8,003; `client_002`: 8,868; `client_003`: 8,574;
`client_004`: 8,672.

Both runs completed 50/50 rounds with 5/5 successful clients every round
(`round_history.csv`, `round_status="completed"` on all 50 rows in both
files) — the attack never caused a client or round failure.

**Confirmed: the malicious attack is the only intended scientific
difference.** `EXP_014`'s `summary.json` carries the additional `"attack"`
block (`attack_type="gaussian_update_noise"`,
`malicious_client_id="client_002"`, `attack_start_round=20`,
`attack_strength_rho=1.0`); no other field differs from `EXP_007`.
`local_training_method`/`fedprox_mu` are absent from both configs —
FedProx is not involved.

---

## 2. Attack Execution Verification

`client_round_trace.csv`'s `attack_active` column confirms, for every one
of the 250 client-round rows: `attack_active=False` for **all** clients at
rounds 0–19; `attack_active=True` **only** for `client_002` at rounds
20–49; `attack_active=False` for `client_000`/`client_001`/`client_003`/
`client_004` at **every** round, including 20–49. `attack_strength_rho=1.0`
and `malicious_client_id="client_002"` are constant on every row. This
exactly matches the pre-registered `EXP_014` configuration.

**Quantifying `client_002`'s update L2 norm before vs. after the attack**
(`update_l2_norm` column):

| Window | `client_002` mean norm |
|---|---:|
| Pre-attack (rounds 0–19) | 6.003 |
| Post-attack (rounds 20–49) | 8.825 |
| **Change** | **+47.0%** |

The shift is immediate, not gradual: round 19 → round 20 is
`5.9146 → 8.4816` (**+43.4%** in a single round transition), and stays
elevated for the rest of the run (round 49: 9.462).

**As a mathematical sanity check**, the attack's own design (plan §5)
predicts `‖noise‖₂ = ρ·‖delta‖₂` with `ρ=1.0`; if the honest delta and the
injected noise are (as expected in an 11.2M-parameter space)
near-orthogonal, the perturbed update's norm should be close to
`√(1+ρ²)·‖delta‖₂ = √2·‖delta‖₂ ≈ 1.414·‖delta‖₂`. The observed round
19→20 ratio (1.434, using round 19 as the nearest honest-norm proxy) is
close to this theoretical value — independent numerical corroboration
that the attack is applying the designed perturbation, not an
implementation artifact.

**Comparison against the four honest clients** (mean `update_l2_norm`,
same pre/post windows):

| Client | Pre-attack mean | Post-attack mean | Change |
|---|---:|---:|---:|
| `client_000` (honest) | 6.516 | 6.561 | +0.7% |
| `client_001` (honest) | 5.822 | 5.659 | **−2.8%** |
| `client_002` (**malicious**) | 6.003 | **8.825** | **+47.0%** |
| `client_003` (honest) | 6.040 | 6.107 | +1.1% |
| `client_004` (honest) | 5.933 | 6.011 | +1.3% |

`client_002` is the only client with anything resembling a systematic,
sustained shift across the round-20 boundary; the four honest clients'
mean norms move by −2.8% to +1.3% — ordinary round-to-round drift,
nowhere close to `client_002`'s +47.0%. **The attack is clearly and
unambiguously visible in the update-norm diagnostic.**

---

## 3. Naive Detector Analysis

The detector's feature is `peer_mean_cosine_similarity`, thresholded per
client at `mean_pre_attack − 3·std_pre_attack` using only rounds 0–19.

**`client_002`'s pre-attack peer-cosine distribution** (rounds 0–19):
values range from 0.0417 (round 19) up to 0.2355 (round 2), mean ≈
**0.1227**. Back-solving from the logged `naive_threshold` (see below)
gives an implied sample standard deviation of ≈ **0.0638**.

**`client_002`'s fixed threshold** (from `client_round_trace.csv`,
`naive_threshold`, constant across all its rows): **−0.06868**.

**`client_002`'s post-attack peer-cosine behavior** (rounds 20–49):
starts at 0.0288 (round 20), declines through the window, bottoms around
−0.0112 (round 41), ends at −0.0076 (round 49); mean ≈ **0.0023**. Every
single one of these 30 values is comfortably **above** the −0.06868
threshold.

**Was `client_002` ever flagged?** **No.** `naive_flag=False` on every
one of its 30 post-attack rows.

**Was ANY client ever flagged?** **No.** Scanning all 250 rows of
`client_round_trace.csv`, `naive_flag=False` on every single row, for
every client, at every round. `first_flagged_round` is empty (unset) for
every row in the file.

**Why did the detector fail — quantified:** the threshold
(−0.06868) was computed from `client_002`'s own pre-attack window, whose
cosine values (mean 0.1227, std 0.0638) come from the fast-changing,
high-cosine early-training phase (rounds 0–19, where directional
agreement across clients is naturally high because all models are moving
in the same coarse direction from a shared random initialization). By the
time the attack activates at round 20, the *entire system's* cosine
similarity has already fallen far below that early-window baseline for
**every** client (see Section 4) — `client_002`'s post-attack values
(mean 0.0023) sit only about 1.1 standard deviations below its own
pre-attack mean, nowhere near the 3-sigma boundary the threshold requires.
The attack's actual effect on this specific feature (if any) is
completely swamped by a much larger, attack-independent secular decline.

---

## 4. Diagnostic: Does Peer-Cosine Similarity Also Drop for Honest Clients?

**Yes — and by a comparable magnitude to `client_002`'s own drop.**

| Client | Pre-attack mean (rounds 0–19) | Post-attack mean (rounds 20–49) | Drop |
|---|---:|---:|---:|
| `client_000` (honest) | 0.1400 | 0.0058 | 0.1342 |
| `client_001` (honest) | 0.1257 | 0.0096 | 0.1161 |
| `client_002` (**malicious**) | 0.1227 | 0.0023 | **0.1204** |
| `client_003` (honest) | 0.1196 | 0.0060 | 0.1136 |
| `client_004` (honest) | 0.1274 | 0.0049 | 0.1225 |

`client_002`'s drop (0.1204) sits **inside the range spanned by the four
honest clients' own drops (0.1136–0.1342)** — it is not the largest, not
the smallest, and not an outlier by any visual or numeric measure. All
five clients — honest and malicious alike — show the same qualitative
pattern: peer-cosine similarity starts high (~0.12–0.14) in early rounds
and decays to near-zero/slightly-negative by round 49.

**Interpretation.** This decay is a property of the *shared global
training trajectory*, not of the attack: as the global model converges
under Dirichlet(α=0.1) heterogeneity, every client's local gradient
becomes smaller and increasingly dominated by client-specific
(honestly-heterogeneous) idiosyncrasies rather than a common
descent direction, so directional agreement with peers falls for
everyone, attacked or not. **This is direct, quantified evidence for the
concern raised in the Phase 5 plan (§12): absolute peer-cosine
thresholding is poorly discriminative for attribution here, because the
feature's dominant source of variation is round-dependent global
dynamics common to all clients, not client-specific malicious behavior.**
A detector using this feature cannot distinguish "this client is
attacked" from "it is late in training and every client's cosine
similarity is naturally low now" without first removing or controlling
for that shared trend.

---

## 5. Paired Model-Performance Comparison

| Metric | EXP_007 (honest) | EXP_014 (attacked) | Δ (014 − 007) |
|---|---:|---:|---:|
| Final test loss | 1.3758617393493653 | 1.340834288406372 | −0.0350 (lower) |
| Final test top-1 | 59.68% | 60.51% | **+0.83pp** |
| Final test top-5 | 88.65% | 89.16% | +0.51pp |
| Macro precision | 0.6509177318450269 | 0.6503461935171988 | −0.0006 |
| Macro recall | 59.65% | 60.52% | +0.87pp |
| Macro F1 | 0.5875828060131298 | 0.5957853875845828 | +0.0082 |
| Balanced accuracy | 59.65% | 60.52% | +0.87pp |
| Best validation top-1 | 59.22% (round 49) | 59.96% (round 48) | +0.74pp (different best round) |

**Every metric except macro precision moved in the "attacked run is
better" direction, by 0.5–0.9 percentage points.** Per this task's
explicit instruction, this is **not** interpreted as evidence the attack
is beneficial. Grounds for caution:

- This is a single paired observation (n=1) at one seed — this project's
  own established discipline (`analysis/EXP_009_Three_Seed_Baseline_Analysis.md`)
  has repeatedly shown single-observation deltas of this size can be
  ordinary run-to-run variation, not a real effect.
- The magnitude (≤0.87pp on any accuracy-style metric) is well below the
  2.85-percentage-point cross-seed noise floor already established for
  this exact configuration (`analysis/EXP_009_Three_Seed_Baseline_Analysis.md`
  §3) — though that floor was measured across *different* seeds/partition
  draws, not same-seed repeated runs, so it is an upper-bound reference,
  not a directly-applicable statistical test here.
- The injected perturbation is **zero-mean Gaussian noise** on one of five
  clients (≈19.7% sample-weight), active for only the latter 30 of 50
  rounds — there is no mechanistic reason to expect a zero-mean
  perturbation to systematically *improve* the aggregate FedAvg direction;
  an unbiased explanation (ordinary training-run variance between two
  separately-executed processes, e.g. GPU/cuDNN non-determinism already
  documented elsewhere in this project) is at least as plausible as any
  causal "attack helped" story, and no evidence here distinguishes
  between them.
- `EXP_014`'s own best-validation round (48) is *not* its final round —
  round 49 dips 0.70pp below round 48 — a late fluctuation of the same
  kind already documented in this project's honest-only runs (e.g.
  `EXP_008`), occurring 29 rounds after the attack began, with no
  apparent correlation to the attack's onset timing.

**Conclusion for this section: no reliable evidence of either benefit or
harm to global model accuracy from this attack, at this `rho`, from a
single paired run.**

---

## 6. Round Trajectory Around the Attack Boundary (Rounds 15–25)

| Round | EXP_007 val. loss | EXP_014 val. loss | EXP_007 val. top-1 | EXP_014 val. top-1 |
|---|---:|---:|---:|---:|
| 15 | 2.1936 | 2.1381 | 39.42% | 40.80% |
| 16 | 2.0963 | 2.0959 | 41.28% | 42.76% |
| 17 | 2.0325 | 2.1003 | 42.86% | 42.16% |
| 18 | 2.0120 | 1.9697 | 43.84% | 44.62% |
| 19 | 1.9075 | 1.9879 | 46.00% | 44.48% |
| **20** | **1.8824** | **1.9362** | **46.58%** | **45.28%** |
| 21 | 1.8948 | 1.8735 | 46.60% | 47.10% |
| 22 | 1.8398 | 1.8628 | 47.74% | 47.64% |
| 23 | 1.7968 | 1.8042 | 48.86% | 48.90% |
| 24 | 1.7910 | 1.8036 | 49.20% | 48.82% |
| 25 | 1.7533 | 1.7484 | 49.60% | 50.22% |

Both runs' validation loss and top-1 continue their ordinary, smooth,
improving trajectory straight through round 20 — there is no visible
spike, plateau, or kink in either curve exactly at the attack boundary.
The two trajectories interleave (sometimes `EXP_007` ahead, sometimes
`EXP_014` ahead) throughout rounds 15–25, consistent with Section 5's
finding of no detectable net performance impact, and inconsistent with a
delayed-but-detectable degradation appearing shortly after round 20.

---

## 7. Research Interpretation

Three distinct questions, kept explicitly separate:

**A. Attack pipeline success — YES, unambiguously.** The attack activates
exactly at round 20 for exactly `client_002` (Section 2), the perturbed
update's norm shows an immediate (+43.4% at the boundary), sustained
(growing to +47.0% mean, up to +50%+ by round 49), and mathematically
consistent (matches the `ρ=1` theoretical scaling) shift unmatched by any
of the four honest clients. All 50/50 rounds completed with 5/5
successful updates throughout. The mechanism works exactly as designed.

**B. Attack impact on global model performance — no reliable impact
detected.** Neither the final paired metrics (Section 5) nor the
round-by-round trajectory around the boundary (Section 6) show a
degradation attributable to the attack; the small final-metric
differences observed lean toward "attacked run slightly better," which is
explicitly *not* interpreted as a real beneficial effect (Section 5).

**C. Malicious-client attribution success — FAILS.** The naive detector
never flags `client_002` — or any client — at any round
(Section 3). This is a clean, reproducible failure with an identified
mechanism (Section 4): the detector's single feature is dominated by a
system-wide, attack-independent secular trend that affects all five
clients by a comparable magnitude, burying whatever attack-specific
signal that feature might otherwise carry.

**Stated plainly: the naive detector fails to identify `client_002` as
malicious under this exact protocol** — not because the attack is
invisible in the data generally (Section 2 shows a large, clean signal in
a *different*, already-logged feature, `update_l2_norm`), but because the
one feature the detector actually uses is the wrong one for this
attack/configuration combination.

---

## 8. Is the Phase 5 Gate Now Satisfied?

The gate (per `research_plans/PHASE5_Malicious_Client_Traceability_Plan.md`
§21 and `research_plans/PHASE4_RESEARCH_STRATEGY.md` §9) requires **both**:

1. **A working malicious-update simulation.** — **Satisfied.** Section 2's
   evidence (clean, large, immediate, sustained norm shift; correct
   activation timing; no pipeline failures) demonstrates this
   unambiguously.
2. **A naive attribution baseline with a documented weakness.** —
   **Satisfied.** The detector's failure (Section 3) is not merely
   "it didn't work" — Section 4 identifies *why*, with quantitative
   evidence (all five clients' cosine similarity drops by a comparable
   ~0.11–0.13 over the run; `client_002`'s post-attack values sit fully
   inside the honest clients' own post-attack range), and Section 2
   independently shows a *different* already-collected signal
   (`update_l2_norm`) does separate `client_002` cleanly from its peers.
   This is precisely the kind of specific, evidence-backed weakness (the
   detector confuses attack-induced signal with honest,
   trajectory-driven heterogeneity) the gate requires.

**Both conditions are met. The Phase 5 gate is satisfied by this single
pilot run**, on the evidence available.

---

## 9. Recommended Next Research Step

**Re-evaluate a norm-aware attribution signal against the already-collected
`EXP_014` trace data before running any new experiment.**

Specifically: `client_round_trace.csv` already contains `update_l2_norm`
for every client-round, and Section 2 of this analysis shows that signal
cleanly separates `client_002` from its four honest peers across the
entire post-attack window (+47.0% mean shift vs. −2.8% to +1.3% for
honest clients) — a far stronger candidate signal than the
`peer_mean_cosine_similarity` feature the current naive baseline actually
uses. The single highest-value next step is to define and evaluate a
second, norm-based (or norm-plus-cosine combined) naive-baseline variant
against this same, already-collected `EXP_014` data — no new attack run,
no new config, and no change to `rho` or `attack_start_round` is needed
to take this step, since the relevant diagnostic evidence is already on
disk. This directly targets the attribution weakness identified in
Sections 3–4, rather than treating attack strength as the blocking
factor — which the evidence here does not support, since the attack is
already clearly visible in an existing diagnostic at `rho=1.0`.

---

## Files Created

- `analysis/EXP_014_Gaussian_Attack_Traceability_Analysis.md`

## Files Modified

- None.

## Commands Executed

- None. Only Read and Write tool calls were used to produce this
  analysis.

## Headline Finding

EXP_014's attack pipeline worked exactly as designed — `client_002`'s
update norm jumps immediately and durably (+43–50%) the moment the
attack activates at round 20, cleanly separated from all four honest
clients' stable norms — but the naive peer-cosine detector never flags
`client_002` (or anyone) at any round, because peer-cosine similarity
declines by a comparable amount for every client as an artifact of the
shared training trajectory, not the attack; since this failure is
specific and evidence-backed (and a stronger candidate signal,
`update_l2_norm`, already sits unused in the same trace file), the Phase 5
gate — a working attack plus a naive baseline with a documented weakness
— is satisfied, and the highest-value next step is re-analyzing this
same data with a norm-aware detector before any new experiment is run.
