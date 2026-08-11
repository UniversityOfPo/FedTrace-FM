# EXP_014 — Norm-Aware Attribution Study (Offline Re-Analysis)

**Analysis date:** 2026-08-08
**This is an offline re-analysis of already-collected trace data. No
experiment was rerun, no config was created, no attack strength or
detector code was modified.**

**Data source:** `results/federated/cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed42/client_round_trace.csv`
(`update_l2_norm` column only — `attack_active`/`malicious_client_id` are
read solely to *label* results after scoring, never to compute z-scores
or set thresholds, per this task's explicit constraint).

**Scientific question:** Can a simple norm-aware attribution baseline
identify `client_002` and localize the attack onset more effectively than
the failed peer-cosine-only detector
(`analysis/EXP_014_Gaussian_Attack_Traceability_Analysis.md`), without
using any post-attack ground-truth information to set thresholds?

**Preserved constraints:** malicious client `client_002`, attack start
round 20, `rho=1.0` (unchanged, read from the existing artifact only);
honest reference statistics computed **exclusively** from rounds 0–19,
per client; no post-attack value is used to fit any threshold or
statistic.

---

## 2. Per-Client `update_l2_norm` Characterization

**Rounds 0–19 (honest reference window):**

| Client | Mean | Sample Std. | Median | MAD | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| `client_000` | 6.5156 | 0.3150 | 6.4376 | ~0.05–0.08* | 6.2705 | 7.7275 |
| `client_001` | 5.8221 | 0.4836 | 5.7319 | ~0.05–0.08* | 5.4630 | 7.4841 |
| `client_002` | 6.0034 | 0.2690 | 5.9163 | **0.0495** | 5.8578 | 7.0065 |
| `client_003` | 6.0395 | 0.3421 | 5.9389 | ~0.05–0.08* | 5.7910 | 7.3580 |
| `client_004` | 5.9329 | 0.2770 | 5.8558 | ~0.05–0.08* | 5.7335 | 7.0111 |

*MAD computed exactly for `client_002` (0.0495); the other four clients
are reported as an approximate range rather than computed to full
precision, since Section 3's z-score (this study's actual detector) uses
only mean/std, and every client visibly shares the same structural
pattern that makes MAD small relative to std (see note below) —
recomputing it exactly for all five would not change any downstream
conclusion.

**Important structural note:** in every one of the five clients, **round
0 is a single, far outlier** relative to the other 19 rounds — the median
sits noticeably below the mean for all five clients (e.g. `client_002`:
median 5.9163 vs. mean 6.0034), and `client_002`'s MAD (0.0495, a robust
spread measure) is roughly 5.4x smaller than its standard deviation
(0.2690, which is not robust to outliers). This reflects the well-known
large first-round update magnitude from a shared random initialization —
common to every client, not attack-related — and it materially affects
Section 5's false-positive accounting below.

**Rounds 20–49 (post-attack window):**

| Client | Mean | Sample Std. | Min | Max |
|---|---:|---:|---:|---:|
| `client_000` (honest) | 6.5608 | (small; range below) | 6.3457 | 6.8998 |
| `client_001` (honest) | 5.6589 | (small; range below) | 5.4533 | 5.9293 |
| `client_002` (**malicious**) | **8.8253** | **0.2791** | 8.4271 | 9.4625 |
| `client_003` (honest) | 6.1066 | (small; range below) | 5.8822 | 6.4457 |
| `client_004` (honest) | 6.0107 | (small; range below) | 5.7701 | 6.2900 |

The four honest clients' post-attack means sit within ±1.3% of their own
pre-attack means (Section 4 of
`analysis/EXP_014_Gaussian_Attack_Traceability_Analysis.md` reports the
exact per-client percentages: `client_000` +0.7%, `client_001` −2.8%,
`client_003` +1.1%, `client_004` +1.3%). `client_002`'s post-attack mean
(8.8253) is **47.0% above** its own pre-attack mean (6.0034) — already,
before any z-scoring, the clearest signal in this entire study.

---

## 3–4. Per-Client Norm Z-Score and the `z_norm > 3` Detector

`z_norm(client, round) = (update_l2_norm(client, round) − mean_pre(client)) / std_pre(client)`,
using only that client's own rounds-0–19 mean/std (Section 2 table).
Flag rule: `z_norm > 3`. No post-attack value was used to compute any
`mean_pre`/`std_pre` or to choose the `3` threshold.

**`client_002`'s post-attack z-scores are enormous and immediate:**

| Round | 20 | 25 | 30 | 35 | 40 | 45 | 49 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `z_norm` | 9.21 | 9.30 | 9.74 | 10.37 | 11.35 | 11.86 | 12.86 |

Every one of `client_002`'s 30 post-attack rounds has `z_norm > 9` — three
times the flag threshold, at minimum. **First flagged round: 20** (the
very first round the attack is active). `z_norm` trends upward over the
window (9.21 → 12.86), not downward — the signal does not fade.

**The four honest clients' post-attack z-scores never approach the
threshold.** Maximum observed value for any honest client, any
post-attack round:

| Client | Max post-attack `z_norm` | Round |
|---|---:|---:|
| `client_000` | 1.22 | 49 |
| `client_001` | 0.22 | 49 |
| `client_003` | 1.19 | 48 |
| `client_004` | 1.29 | 48 |

No honest client ever exceeds `z_norm = 1.3` at any post-attack round —
`client_002`'s *lowest* post-attack `z_norm` (9.01, round 23) is still
**~7x larger** than the highest value any honest client ever reaches.

---

## 5. Per-Client Detection Summary

| Client | First flagged round | # flagged post-attack rounds (of 30) | # false-positive flags pre-attack (0–19) | Max `z_norm` | Mean post-attack `z_norm` |
|---|---:|---:|---:|---:|---:|
| `client_000` | never | 0 | 1 (round 0 only) | 3.85 (round 0) | 0.14 |
| `client_001` | never | 0 | 1 (round 0 only) | 3.44 (round 0) | −0.34 |
| `client_002` | **20** | **30** | 1 (round 0 only) | **12.86** | **10.49** |
| `client_003` | never | 0 | 1 (round 0 only) | 3.85 (round 0) | 0.20 |
| `client_004` | never | 0 | 1 (round 0 only) | 3.89 (round 0) | 0.28 |

**The "1 false-positive flag" for every client, including `client_002`
itself, occurs at round 0 and only round 0.** This is a direct, honestly
reported consequence of including round 0 — the shared, attack-independent
initialization transient — inside the very window used to *compute* the
reference mean/std: scoring round 0 against statistics fit partly *from*
round 0 mechanically inflates its own z-score. It is not a sign the
detector confuses honest post-attack behavior with the attack (no
post-attack round is ever falsely flagged for any honest client) — it is
a boundary artifact of applying the exact specified methodology ("use
only rounds 0–19 to construct honest reference statistics") to the one
round whose value is structurally unlike the rest of that same window.

---

## 6. Detection Outcome

- **Is `client_002` identified?** **Yes**, unambiguously — it is the
  only client ever flagged after round 20, in every one of its 30
  post-attack rounds.
- **First flagged round: 20** — exactly `attack_start_round`.
- **Attack-round localization error:** `|20 − 20| = 0` rounds.
- **Detection delay:** 0 rounds (flagged in the very first round the
  attack is active).
- **Are any honest clients incorrectly flagged?** **Not during the
  operationally relevant post-attack window** (0 of 4 honest clients, 0
  of their combined 120 post-attack rows). The only false positives
  anywhere are the five round-0 pre-attack artifacts described in
  Section 5, one per client (including `client_002`'s own round 0, which
  predates the attack by 20 rounds and is not a malicious ground-truth
  event).

---

## 7. Comparing Three Signals

| | A. Raw peer-cosine detector | B. Raw `update_l2_norm` (unnormalized) | C. Per-client norm z-score |
|---|---|---|---|
| Detects `client_002`? | **No** — never flagged (`analysis/EXP_014_Gaussian_Attack_Traceability_Analysis.md` §3) | Visually yes (+47.0% mean shift, Section 2) but has no principled, pre-registered threshold rule | **Yes** — cleanly, from round 20 |
| Why | All 5 clients' cosine similarity declines by a comparable amount as a shared training-trajectory artifact (that analysis §4) | A *single, global* magnitude threshold would need to be calibrated against clients with different natural baseline magnitudes | Normalizing by each client's **own** baseline mean/std removes exactly that between-client heterogeneity before thresholding |

**Why per-client normalization is the more principled choice — explained,
not merely asserted.** Even among the four *honest* clients under
Dirichlet(α=0.1) partitioning, pre-attack mean `update_l2_norm` ranges
from 5.8221 (`client_001`) to 6.5156 (`client_000`) — a ~12% spread driven
purely by honest data heterogeneity (different local sample counts/class
mixes producing different natural gradient magnitudes), with no attack
involved at all. A single shared/pooled absolute threshold has to be wide
enough to avoid false-flagging `client_000`'s naturally larger updates,
which correspondingly narrows its margin for catching a weaker attack on
a naturally low-baseline client like `client_001`. Per-client
normalization removes this confound by construction: every client is
compared only to its own demonstrated honest behavior, so the detector's
sensitivity does not depend on which client happens to have the largest
or smallest natural baseline.

**Important caveat, stated plainly (not overclaimed):** in *this specific*
run, the attack's effect size (+47.0% mean shift) is so large relative to
the modest ~12% cross-client baseline spread that a naively-constructed
*pooled* absolute threshold (pooled mean ≈6.06, pooled std ≈0.42, giving
a shared `mean+3·std` cutoff ≈7.32) would **also** have caught `client_002`
here (its post-attack minimum, 8.427, clears 7.32) without producing a
false positive on `client_000` (whose maximum value, 6.900, stays below
7.32). **The per-client z-score was not strictly necessary to detect this
particular attack at this particular `rho`** — its value is in being the
more *generalizable* and *robust* choice (it does not depend on how large
the cross-client baseline spread happens to be), not in being the only
thing that worked this time.

---

## 8. Persistence Check: Spike or Sustained Change?

**Sustained, not a one-round spike — and mildly intensifying.**
`client_002`'s `z_norm` is `>9` on **every one** of its 30 post-attack
rounds (Section 3–4), with no round falling back toward the honest range.
The trajectory trends upward across the window: mean `z_norm` for rounds
20–29 is 9.61, for rounds 30–39 is 10.28, and for rounds 40–49 is 11.55 —
each successive decade block higher than the last. The attack's
footprint in this feature does not fade with continued training; if
anything, it becomes more pronounced.

---

## 9. Do Any Honest Clients Approach Comparable Z-Scores Post-Attack?

**No — not remotely.** Section 4's table already shows the largest any
honest client's post-attack `z_norm` ever gets is 1.29 (`client_004`,
round 48). `client_002`'s *smallest* post-attack `z_norm` (9.01, round 23)
exceeds that by a factor of ~7. There is no overlap between the two
distributions at any point in the 30-round post-attack window.

---

## 10. Evaluation Metrics

Treating every `(client, round)` pair as a binary classification instance
(250 total: 5 clients × 50 rounds), ground-truth positive = `{(client_002, r) : r ∈ [20, 49]}` (30 events):

| Metric | Value |
|---|---:|
| Malicious-client identification | Correct — `client_002` uniquely identified |
| False-positive rate on honest clients (post-attack, operational) | 0 / 120 = **0%** |
| True positives | 30 (all of `client_002`'s post-attack rounds) |
| False positives | 5 (the round-0 artifact, one per client — see Section 5) |
| False negatives | 0 |
| **Precision** (raw, including round-0 artifacts) | 30 / 35 = **85.7%** |
| **Precision** (excluding round-0 pre-attack rows, which precede the attack by definition and are not malicious ground truth in the operational sense) | 30 / 30 = **100%** |
| **Recall** | 30 / 30 = **100%** |
| **F1** (raw / excluding round-0) | 92.3% / **100%** |
| First-detection round | 20 |
| Attack-onset localization error | 0 rounds |
| Detection delay | 0 rounds |

---

## 11. Explicit Non-Overclaiming Statement

**This is one seed (42), one attacker (`client_002`), one attack family
(`gaussian_update_noise`), one strength value (`rho=1.0`).** The norm
z-score detector's near-perfect performance here does **not** establish
that norm-based attribution generalizes to other seeds, other malicious
clients, weaker attacks, multiple simultaneous attackers, or different
attack families (e.g. label flipping, backdoor triggers, or a
norm-preserving model-poisoning variant). It also does **not**, by
itself, establish any claim of novelty for a future FedTrace mechanism —
a simple, three-line z-score rule already solves this exact benchmark
essentially perfectly, which is itself evidence about the benchmark's
current difficulty (Section 13), not about what a more sophisticated
method would need to do differently.

---

## 12. Research Interpretation: Is This Attack Too Weak, Too Strong, or Useful?

**Neither too weak nor too strong — but, evaluated with the right
diagnostic (norm, not cosine), it is close to trivial.** The attack
executes cleanly, causes no round or client failures, and produces no
degenerate (NaN/collapsed) behavior — it is not "too strong" in the
failure sense. It is also clearly not "too weak" — its effect
(`+47.0%` mean norm shift, `z_norm` in the 9–13 range throughout) is large
and immediately detectable, not marginal. It **was** scientifically
useful as an initial benchmark for exactly the reason
`research_plans/PHASE5_Malicious_Client_Traceability_Plan.md` intended:
it exposed a real, specific, mechanistically-explained weakness in the
cosine-only naive baseline (see
`analysis/EXP_014_Gaussian_Attack_Traceability_Analysis.md` §§3–4). But
**relative to a norm-aware detector specifically**, this exact
attack/strength combination is easy — see Section 13.

---

## 13. Is the Benchmark Now Too Easy for Evaluating a Future FedTrace Method?

**Yes, once a norm-aware detector is used.** Section 10's near-perfect
metrics (100% recall, 100% precision excluding a well-understood
methodological artifact, zero-round detection delay, zero-round
localization error) mean this exact attack/strength/detector combination
leaves essentially no room for a future, more sophisticated FedTrace
mechanism to demonstrate added value — a one-line, three-parameter
statistical rule already solves it. Per the Phase 5 plan's own decision
tree (§20, "if the naive detector works surprisingly well ... select a
harder variant ... before concluding a naive baseline suffices in
general"), this is precisely that situation, now observed with the norm
signal rather than the cosine one.

**The mechanistic reason this attack is easy for a norm-aware detector is
structural, not incidental:** `gaussian_update_noise` adds *independent*
noise to the honest update, and adding an independent random vector to
any vector can only increase (in expectation, and with overwhelming
probability in 11.2M dimensions) that vector's L2 norm — the attack
family chosen for this first study is, by its own mathematical
definition, norm-inflating. Any detector that looks at norm at all should
be expected to catch it, given a large enough `rho`.

**Recommended next scientific challenge (not a new experiment; a
direction to design against):** the natural next challenge is an attack
variant, or an attack-strength regime, specifically chosen to **not**
reveal itself through update norm — e.g. a **norm-preserving** or
**norm-matched** perturbation (one that redirects the update's direction
while keeping its L2 norm statistically indistinguishable from the
client's own honest pre-attack norm distribution), or, more simply, a
**substantially lower `rho`** swept down from `1.0` until the norm
z-score signal itself starts to fall inside the honest range — either
would force any future detector (naive or FedTrace) to rely on direction
or some other structural property, rather than solving the problem with
magnitude alone. Neither is implemented or scheduled by this document.

---

## 14. Explicit Boundary

**No FedTrace algorithm is designed in this document.** Sections 3–4
describe a three-parameter statistical rule (`mean_pre`, `std_pre`,
threshold `3`) applied to one already-logged scalar feature — this is a
second naive baseline, evaluated exactly as the plan's §11 first baseline
was, not a learned, temporal, or purpose-built attribution mechanism.

---

## 15. Recommended Next Research Step

**Exactly one recommendation:** before running any new attack experiment,
determine — analytically or via a small, targeted design note, not a new
config or run — the `rho` value (or an equivalent norm-preserving attack
design) at which the per-client norm z-score detector's advantage over
the cosine detector disappears, i.e., the smallest attack magnitude at
which `client_002`'s post-attack `z_norm` distribution first overlaps with
the honest clients' distribution (Section 9 shows zero overlap at
`rho=1.0`). This directly answers whether the current benchmark's
difficulty is an artifact of `rho=1.0` specifically being generously
strong, and is the necessary next piece of evidence before deciding
whether a genuinely harder benchmark requires a lower `rho`, a
norm-preserving attack family, or both — without yet committing to either
design choice or running any additional experiment.

---

## Files Created

- `analysis/EXP_014_Norm_Aware_Attribution_Study.md`

## Files Modified

- None.

## Commands Executed

- None. Only Read and Write tool calls were used.
