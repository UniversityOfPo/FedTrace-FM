# FedTrace Method Design Plan

**Status:** Design document only. No code, config, test, README, or
Research Database file was created or modified in producing it. No
experiment was run. This document proposes a design and a validation
sequence; it does **not** implement FedTrace, and Stage A (§14) is the
only immediately-actionable next step this plan identifies — as pure
offline analysis of already-collected data, not a new experiment.

**Evidence base:** `research_plans/PHASE5_Malicious_Client_Traceability_Plan.md`,
`research_plans/EXP_015_Norm_Preserving_Directional_Attack_Plan.md`,
`analysis/EXP_014_Gaussian_Attack_Traceability_Analysis.md`,
`analysis/EXP_014_Norm_Aware_Attribution_Study.md`,
`analysis/EXP_014_Attack_Difficulty_Calibration.md`,
`analysis/EXP_015_Directional_Attack_Traceability_Analysis.md`, and the
current implementation (`src/attacks/traceability.py`,
`client_round_trace.csv`'s field set) as inspected, not modified.

---

## 1. Research Problem

**Formal statement.** Given a federated learning run under strong,
realistic Non-IID heterogeneity (this project's established
Dirichlet(α=0.1), 5-client, FedAvg, CIFAR-100/ResNet-18 configuration),
observing only each selected client's per-round submitted model update
(never its raw training data), determine, for each communication round:

**A. Which client (if any) is behaving maliciously** — a client
identification / ranking problem across the (typically 5) clients
selected that round and across the run.

**B. When the malicious behavior begins** — a round-level onset
localization problem, distinct from mere detection: identifying not just
*that* an attack is present in the run, but the specific round at which
a specific client's behavior changed.

**Both without confusing honest heterogeneity with malicious behavior.**
This project's own evidence (§2 below) has already shown this is the
crux of the difficulty: under `alpha=0.1`, honest clients' updates
already diverge substantially from each other and drift substantially
over the course of training, for reasons that have nothing to do with
any attack.

---

## 2. Evidence Motivating FedTrace

**`EXP_014` (`gaussian_update_noise`, ρ=1.0) established:**
- Additive independent noise at this strength produced a large,
  immediate, sustained **magnitude** anomaly (`update_l2_norm` +47.0%
  mean shift; a trivial per-client norm z-score detector flagged
  `client_002` on every one of 30 active rounds, `z_norm > 9` throughout
  — `analysis/EXP_014_Norm_Aware_Attribution_Study.md` §§4–6).
- The pre-registered **peer-cosine** 3-sigma baseline
  (`traceability.py::apply_naive_detector`) never flagged anyone —
  `peer_mean_cosine_similarity` declines by a comparable amount for
  **every** client (honest and malicious alike) as an artifact of the
  shared training trajectory, not the attack
  (`analysis/EXP_014_Gaussian_Attack_Traceability_Analysis.md` §4).
- **The norm signal alone trivially "solved" this attack** — leaving no
  room for any more sophisticated method to demonstrate added value
  (`analysis/EXP_014_Norm_Aware_Attribution_Study.md` §13).

**`EXP_015` (`norm_preserving_directional`, target_cosine=0.0)
established:**
- An exact-rotation attack that preserves `update_l2_norm` to ~1e-10
  precision removed the magnitude shortcut almost entirely — the same
  norm z-score rule now only crosses its threshold at round 49 (a
  29-round detection delay, one correct flag out of 30 active rounds)
  (`analysis/EXP_015_Directional_Attack_Traceability_Analysis.md` §5).
- The peer-cosine baseline again never flagged anyone, for a related but
  distinct reason (`client_002`'s peer-cosine pins near exactly zero and
  crosses from the *lowest*-of-five to the *highest*-of-five client
  partway through the attack window — no fixed absolute threshold
  captures this) (same document, §6).
- A **not-yet-formalized** diagnostic already being logged,
  `self_prev_round_cosine` (a client's cosine similarity to its own
  immediately-preceding submitted update), showed a clean, immediate,
  zero-overlap separation at the exact attack-onset round (same
  document, §§7–9).
- The attack also produced a **real, sustained, boundary-localized
  global-model degradation** (≈5–6pp final accuracy), unlike `EXP_014`'s
  essentially neutral outcome (same document, §§10–11).

**Why no single observed scalar should automatically become the full
FedTrace method.** This is the central lesson of the two-attack series,
stated as the constraint this design plan must honor: **each attack
family lit up exactly one different signal, and defeated the other two.**
`update_l2_norm` was decisive for `EXP_014` and nearly useless for
`EXP_015`. `self_prev_round_cosine` was decisive for `EXP_015` and was
never even evaluated against `EXP_014` (it did not exist as a logged
field then). `peer_mean_cosine_similarity`, in its current
absolute-threshold form, failed against both. Defining FedTrace as a
threshold on whichever single feature happened to work best against the
attacks tried so far would be curve-fitting to two data points, not a
general method — the explicit scientific constraint this plan is
required to honor.

---

## 3. Threat Model

**Assumptions (design choices, not experimentally established facts):**
- The server observes every selected client's submitted model update
  (`ClientUpdate.model_state`, per `src/federated/contracts.py`) each
  round. It never inspects any client's raw training data.
- Exactly one client is malicious in this first design's benchmark scope
  (matching every experiment run so far — `EXP_014`, `EXP_015`).
  Multi-attacker/colluding threat models are out of scope for this
  design (see §17).
- A malicious client may begin attacking at any round, potentially after
  an honest "warm-up" period during which its behavior is
  indistinguishable from an honest client's (both benchmarked attacks
  used a delayed onset, per
  `research_plans/PHASE5_Malicious_Client_Traceability_Plan.md` §3's own
  justification for why a delayed onset is required for onset-detection
  research to be meaningful at all).
- Malicious behavior may alter a submitted update's **magnitude**,
  **direction relative to peers**, **temporal self-consistency**, or any
  combination — the design must not presuppose which dimension a future
  attack will manipulate.

**Facts established experimentally, not merely assumed, by this
project's own prior work:**
- Honest clients under `alpha=0.1` are naturally, substantially
  heterogeneous in both sample composition (§7 of
  `analysis/EXP_009_Three_Seed_Baseline_Analysis.md`, client
  sample-count CV up to ≈31%) and in their peer-cosine trajectory (a
  large, shared, round-dependent decline common to all five clients,
  `analysis/EXP_014_Gaussian_Attack_Traceability_Analysis.md` §4) — this
  is not a hypothesis FedTrace's design must take on faith; it is a
  directly measured, repeatedly confirmed property of this exact
  configuration.
- Honest clients' `update_l2_norm` is comparatively stable (pre-attack
  CVs of 4.5%–8.3% across clients,
  `analysis/EXP_014_Attack_Difficulty_Calibration.md` §3) and their
  `self_prev_round_cosine` stays solidly positive and smoothly evolving
  throughout training (never below ≈0.12 in either benchmarked run).

---

## 4. Candidate FedTrace Signal Families

| Signal | Measures | `EXP_014` result | `EXP_015` result | Strength | Weakness | Attack-specific shortcut risk |
|---|---|---|---|---|---|---|
| `update_l2_norm` | Update magnitude relative to the client's own honest baseline | Decisive: every active round `z_norm > 9`, zero honest false positives | Nearly defeated: only 1/30 rounds crosses `z_norm > 3`, 29-round delay | Cheap, already logged, strong against magnitude-perturbing attacks | Provably uninformative against any exactly norm-preserving attack, by construction | **High** if used alone — any future attacker who preserves norm defeats it completely, as demonstrated |
| `peer_mean_cosine_similarity` (raw, absolute-threshold form) | Directional alignment with the sample-weighted mean of co-selected peers, vs. a fixed pre-attack-derived threshold | Failed: dominated by a shared, attack-independent secular decline | Failed: malicious client's rank among peers changes sign mid-window | Directly targets the flagship "malicious vs. honest heterogeneity" distinction in principle | The specific *absolute-threshold* implementation cannot separate a shared trend from a client-specific deviation | Low risk of over-fitting to one attack (it failed against both) but its current form is simply not useful as implemented |
| `self_prev_round_cosine` | A client's own temporal self-consistency, round to round | Not evaluated (field did not exist when `EXP_014` was run) | Decisive: zero-overlap separation, immediate at onset | Directly targets *round-level* onset localization, which the other two signals cannot do well (§5 below) | Untested against any magnitude- or peer-direction-based attack; a future attacker who deliberately correlates consecutive rounds' perturbations (e.g. a fixed or slowly-drifting malicious direction, rather than `EXP_015`'s fresh-per-round draw) would defeat it, as already flagged in `analysis/EXP_014_Attack_Difficulty_Calibration.md` §7's discussion of a norm-preserving *directional* attack's own future escalation path | **High** if used alone — this is exactly the risk this plan's explicit constraint forbids building around |

**Conclusion driving §5:** the three signals are functionally
complementary (each was decisive for a different, disjoint subset of the
two attacks tried), none is individually general, and their known
failure modes do not overlap. This is the direct evidentiary
justification for combining them rather than picking one.

---

## 5. Multi-Signal FedTrace Representation

**One compact per-client, per-round trace vector, built entirely from
already-logged (or a same-round re-normalization of an already-logged)
diagnostic — no new raw trace field is proposed:**

```
v(k, r) = [ m(k, r), p(k, r), t(k, r) ]
```

- **`m(k, r)` — magnitude behavior:** `update_l2_norm(k, r)` (already
  logged, unchanged).
- **`p(k, r)` — peer-relative directional behavior:**
  `peer_mean_cosine_similarity(k, r)` (already logged, unchanged value)
  — but, per §4's finding that the *absolute* value is not usable,
  re-normalized in §6 using a **same-round, cross-client** statistic
  (computed across that round's selected clients) rather than a
  per-client historical one. This is a re-processing of the existing
  field at scoring time, not a new trace column.
- **`t(k, r)` — temporal/self-consistency behavior:**
  `self_prev_round_cosine(k, r)` (already logged, unchanged; `None` on a
  client's first observed round, per its own documented semantics).

No fourth component (e.g. a robust-center distance or a layer-wise
breakdown, both explicitly deferred in
`research_plans/PHASE5_Malicious_Client_Traceability_Plan.md` §10) is
added — three complementary, already-evidenced families is the minimum
that covers every failure mode observed so far, and adding more without
evidence would violate this task's explicit "do not add arbitrary
features" constraint.

---

## 6. Honest-Reference Calibration

**Per-client, longitudinal calibration (for `m` and `t`).** For each
client `k`, compute `mean_pre(k)`/`std_pre(k)` from that client's own
values over a fixed **calibration window** `[0, T_ref)`, exactly as
`EXP_014`'s offline norm z-score study already did
(`analysis/EXP_014_Norm_Aware_Attribution_Study.md` §3). Standard
deviation (not a robust alternative like MAD) is used for consistency
with every prior baseline in this series, given a stated limitation (see
below).

**Per-round, cross-sectional calibration (for `p`).** For each round `r`,
compute the mean and standard deviation of `peer_mean_cosine_similarity`
**across that round's selected clients themselves** (no historical
window at all) — this is the direct fix for §4's finding that `p`'s
*absolute* value is dominated by a trend shared by all clients equally:
normalizing against the same round's own cross-client spread cancels a
shared trend by construction, since the trend shifts every client's
value together.

**Why per-client normalization, not one pooled/shared threshold, for `m`
and `t`.** Already justified with data in
`analysis/EXP_015_Norm_Preserving_Directional_Attack_Plan.md` §7: honest
clients under `alpha=0.1` have measurably different natural baselines
(e.g. pre-attack `update_l2_norm` means ranging 5.82–6.52 across the five
clients in `EXP_007`/`EXP_014`/`EXP_015`'s shared seed-42 partition) — a
single shared threshold would need to be tuned to the largest-baseline
client, starving sensitivity for smaller-baseline ones.

**Preventing future leakage.** `mean_pre(k)`/`std_pre(k)` are computed
**only** from rounds `< T_ref`; no round `>= T_ref` is ever used to fit
either statistic, mirroring `apply_naive_detector`'s already-verified
discipline (`analysis/PHASE5_Implementation_Verification.md` item 14).
`T_ref` itself must **not** be read from the attack configuration's
`attack_start_round` field (that would be direct ground-truth leakage);
it is instead a fixed **deployment policy** — a bootstrap/warm-up window
length chosen from this project's own already-established, attack-independent
convergence-phase boundary (the round-block convention used since
`analysis/EXP_004_Convergence_Analysis.md`, under which rounds 0–19 are
consistently the fast-changing phase and round ≈20 onward is
consistently calmer, for *every* honest-only baseline this project has
run, `EXP_004`/`EXP_006`/`EXP_007`/`EXP_008`/`EXP_009`, entirely
independent of any attack). That this policy value (`T_ref=20`) happens
to numerically coincide with `EXP_014`/`EXP_015`'s pre-registered
`attack_start_round` is because the latter was *itself* chosen using the
former (`research_plans/PHASE5_Malicious_Client_Traceability_Plan.md`
§7), not the reverse — the calibration procedure would be identical for
any hypothetical run, whether or not it contains an attack at all.

**Round-0 boundary artifact.** `analysis/EXP_014_Norm_Aware_Attribution_Study.md`
§5 documented that scoring round 0 against a reference window that
includes round 0 itself produces a spurious flag for every client
(a shared initialization transient, not attack-related). FedTrace's
calibration window is used **only to fit `mean_pre`/`std_pre`**, never to
*score* client-rounds against those same statistics — scoring begins
strictly at round `T_ref`, so this artifact cannot occur by construction
(the clean convention `analysis/EXP_014_Attack_Difficulty_Calibration.md`
§9 already recommended).

**Clients with insufficient history.** `t(k, r)` is `None` on client
`k`'s first observed round (no prior submitted update exists yet, per
`build_trace_row`'s already-implemented semantics,
`analysis/PHASE5_Implementation_Verification.md` item 23). `m(k, ·)`'s
calibration requires at least 2 pre-`T_ref` observations for `client k`
(matching `apply_naive_detector`'s existing `len(pre_attack_values) >= 2`
rule) — a client with fewer is excluded from scoring entirely for that
signal (never assigned a fabricated threshold), and the combined score
(§7) degrades gracefully to the remaining available signals for that
client-round rather than failing.

---

## 7. FedTrace Anomaly Score

**Standardized per-signal scores**, each a magnitude of a z-score against
its own calibration reference (§6):

```
z_m(k, r) = | m(k, r) − mean_pre_m(k) | / std_pre_m(k)

z_p(k, r) = | p(k, r) − mean_round_p(r) | / std_round_p(r)
            (mean/std computed across clients selected in round r)

z_t(k, r) = | t(k, r) − mean_pre_t(k) | / std_pre_t(k)
```

Each `z_·(k, r)` is `None`/excluded wherever its underlying signal or
reference statistic is undefined (§6's "insufficient history" cases).
Absolute value is used for all three (not a one-signed test) — the two
attacks tried so far happened to push `m`/`t` in one direction each, but
nothing in the threat model (§3) guarantees a future attack would do the
same, and an undirected test is the more defensible default absent
stronger evidence.

**Combined anomaly score — an unweighted maximum over available
components:**

```
AnomalyScore(k, r) = max( z_m(k, r), z_p(k, r), z_t(k, r) )
                      over whichever of the three are defined for (k, r)
```

**Why `max`, not a weighted sum or a learned combiner.** This is the
direct, minimal formalization of §2's central lesson: `EXP_014` was
caught almost entirely by `z_m`; `EXP_015` was caught almost entirely by
`z_t`; in neither case did the *other* signals contribute much
evidence. A `max` combiner requires only that **one** complementary
signal fire strongly to raise the combined score — exactly the property
needed for a method meant to remain effective against attacks that
manipulate only one behavioral dimension, without requiring the
detector to know in advance *which* dimension a given attack will
choose. A weighted sum would need attack-specific weight tuning to avoid
diluting a single strong signal with two near-zero ones (precisely the
"attack-specific shortcut" risk this plan is constrained to avoid,
relocated into the weights instead of the feature choice); a learned
combiner is explicitly not justified by two data points (per this
task's own instruction, and per §17's contribution-boundary framing) and
is not proposed. `max` is fully interpretable (the responsible signal is
always identifiable after the fact) and requires no fitting beyond §6's
already-specified per-client/per-round statistics.

---

## 8. Attribution Rule

**Raw flag:** client-round `(k, r)` is **raw-anomalous** if
`AnomalyScore(k, r) > 3` (the `3`-sigma convention already used
throughout this benchmark series, kept for direct comparability with
every prior baseline — not re-tuned per attack).

**Persistence requirement (reduces isolated false positives).** A client
`k` is **detected** starting at the first round `r*` such that
`AnomalyScore(k, r)` exceeds `3` for **every round in `[r*, r* + P − 1]`**
(`P` consecutive rounds, proposed `P = 3`, justified in §9). A single
isolated raw-anomalous round, with no persistence, is *not* sufficient
to declare detection — directly answering this task's §8 question in the
affirmative, and directly motivated by `EXP_015`'s own observation that
even purely honest clients occasionally cross intermediate z-score levels
by chance in a 50-round run (`analysis/EXP_015_Directional_Attack_Traceability_Analysis.md`
§5's honest-client `z_norm` values reaching as high as ≈1.2–1.3 without
any attack at all — a persistence requirement provides margin against
this kind of ordinary fluctuation eventually exceeding `3` by chance
over many rounds, without needing a higher, less interpretable
single-round threshold).

**Predicted malicious client:** the client with the **earliest**
detected onset round `r*` (§ above); ties broken by the higher mean
`AnomalyScore` over its persistent run. If no client is ever detected,
FedTrace outputs "no malicious client identified" rather than forcing a
guess.

**Suspicious-client ranking:** all clients ordered by
`(has_detection descending, r* ascending, mean AnomalyScore during the
persistent run descending)` — detected clients first (earliest onset
first), then any never-detected clients ordered by their single highest
observed `AnomalyScore` anywhere in the run.

**First detected malicious round** (reported separately from the onset
*estimate*, §9): `r*` itself, i.e. the first round of the confirmed
persistent run — not `r* + P − 1` (the round persistence was
*confirmed*), since §9 explains why the estimate should be attributed
back to the start of the run, not the confirmation point.

---

## 9. Onset Localization

**The detector never has access to `attack_start_round`.** Every
statistic and rule in §§6–8 is computable using only `AnomalyScore(k, r)`
values for `r <= current round`, and the calibration statistics
(§6) are fixed once, from a window that ends before scoring ever begins
— no rule anywhere in this design looks ahead to a future round's value
to decide an earlier round's status.

**Onset estimate: `onset_estimate(k) = r*`**, the first round of the
confirmed `P`-round persistent run (§8) — **not** the round at which
persistence was *confirmed* (`r* + P − 1`). This is a deliberate design
choice: confirmation necessarily lags behind the true onset by up to
`P − 1` rounds (since `P` consecutive anomalous rounds must be observed
*before* persistence can be confirmed), but the anomalous behavior
itself began at `r*`, which is knowable in hindsight (once round
`r* + P − 1` has been observed) without ever having consulted a future
round *to score* an earlier one — the detector only ever *retroactively
labels* a round it already scored, using evidence that existed at
scoring time (`AnomalyScore(k, r*)` was already computed and already
exceeded `3` at round `r*` itself; only the *decision to act on it* is
delayed until persistence is confirmed).

**Choice of `P = 3`:** small enough to keep detection delay bounded
(at most 2 rounds beyond the true onset, given `EXP_014`'s and
`EXP_015`'s own onset behavior — both showed anomaly onset coincide
exactly with `attack_start_round` once the *correct* signal fires,
§§5, 7), while large enough to require more than a single-round
coincidence. This is a provisional, explicitly-not-yet-validated choice
— Stage A (§14) is where it is checked against the two already-collected
runs before being frozen.

---

## 10. Evaluation Metrics (Pre-Registered)

- **Malicious-client identification accuracy** — whether §8's predicted
  client equals the true `malicious_client_id`.
- **Detection precision / recall / F1** — treating every
  `(client, round)` pair with `round >= attack_start_round` as ground-truth
  positive (restricted to the true malicious client), scored against
  every round flagged raw-anomalous (§8's first tier) — the same
  `(client, round)`-event framing already used in
  `analysis/EXP_014_Norm_Aware_Attribution_Study.md` §10, for direct
  comparability.
- **False-positive rate on honest clients** — fraction of honest
  client-rounds ever raw-anomalous, and, separately, fraction of honest
  clients that ever reach persistent detection (§8) at all (the more
  operationally relevant of the two).
- **Detection delay** — `(round persistence was confirmed) − attack_start_round`.
- **Onset localization error** — `|onset_estimate(malicious client) − attack_start_round|`.
- **Client-ranking quality** — the true malicious client's position in
  §8's suspicious-client ranking (ideally rank 1); reported because a
  ranking is more informative than a binary flag when evaluating whether
  a method is "close" even in a case where it does not cross the
  persistence threshold.

---

## 11. Required Baselines

- **A. Existing peer-cosine 3-sigma detector** — `traceability.py::apply_naive_detector`,
  unmodified, exactly as already evaluated against both `EXP_014` and
  `EXP_015`.
- **B. Norm-only detector** — the offline `z_norm(k, r) = |m(k,r) − mean_pre_m(k)| / std_pre_m(k) > 3`
  rule, formalized with §8's same persistence rule (`P=3`) for a fair
  comparison (`analysis/EXP_014_Norm_Aware_Attribution_Study.md`'s
  original single-round rule is reproduced as a variant, not the primary
  comparison point, since §8 changes the flagging rule for all methods
  uniformly).
- **C. Self-prev-round-cosine-only detector** — `z_t(k, r) > 3` with the
  same persistence rule. **Evaluable on `EXP_015` only** — `EXP_014`'s
  `client_round_trace.csv` predates the `self_prev_round_cosine` field
  and does not contain it (§14 addresses this constraint directly).
- **D. FedTrace combined detector** — §7's `max(z_m, z_p, z_t)` with
  §8's persistence rule.

**This comparison is the primary evidence for §15/§16's success/failure
determination** — FedTrace (D) must be compared against each single-signal
baseline (A/B/C) on the *same* evaluation protocol and metrics (§10), not
against a different, more favorable framing for D.

---

## 12. Ablation Plan

| Ablation | Signals used | Scientific question answered |
|---|---|---|
| Magnitude only | `z_m` | Reproduces baseline B; establishes the magnitude-only floor/ceiling on each attack. |
| Peer direction only | `z_p` (cross-sectional form, §6/7 — **not** identical to baseline A, which uses the original absolute-threshold form) | Does the *improved*, same-round-normalized peer signal (not yet evaluated in any prior document) add value on its own, independent of the other two? |
| Temporal consistency only | `z_t` | Reproduces baseline C; establishes the temporal-only floor/ceiling (on `EXP_015` only, per §14). |
| Magnitude + peer | `max(z_m, z_p)` | Does adding peer-direction evidence help when temporal information is unavailable (directly answers the `EXP_014`-only evaluability constraint, §14) or is not the relevant signal? |
| Magnitude + temporal | `max(z_m, z_t)` | Does combining the two signals that were each individually decisive for one attack recover both attacks without the peer-direction component at all? |
| Peer + temporal | `max(z_p, z_t)` | Is the norm/magnitude signal actually necessary, or do the two directional/temporal signals alone suffice — directly tests whether §7's inclusion of `z_m` is load-bearing. |
| Full FedTrace | `max(z_m, z_p, z_t)` | The proposed method; must outperform every ablation on at least one benchmarked attack without underperforming any of them on the other, to justify the three-signal combination over any simpler subset (§15). |

---

## 13. Leakage and Fairness Safeguards

- **No attack ground truth is used to fit any threshold.** The `3`-sigma
  flagging threshold, the `P=3` persistence length, and every
  calibration statistic (§6) are fixed *before* being applied to either
  benchmarked attack's trace, and are not re-tuned per attack.
- **No post-attack (i.e. `round >= T_ref`) sample is ever used for
  honest calibration** — `mean_pre_·(k)`/`std_pre_·(k)` are computed
  exclusively from `[0, T_ref)`, per §6.
- **No future round is used to score an earlier round.** `AnomalyScore(k, r)`
  depends only on `(k, r)`'s own signal values and the fixed,
  once-computed calibration statistics — never on `AnomalyScore(k, r')`
  for `r' > r`. The onset-estimate retroactive-labeling step (§9)
  re-labels round `r*` only after round `r* + P − 1` has been
  *independently scored on its own terms*, not by borrowing its score.
- **The same detector settings are used across attack families.**
  `T_ref=20`, threshold `=3`, `P=3`, and the `max` combiner are applied
  identically to `EXP_014` and `EXP_015` (and to every ablation/baseline
  in §§11–12) — no per-attack tuning.
- **Thresholds cannot be tuned separately to make `EXP_014` or `EXP_015`
  look good.** This is enforced structurally, not just by intention: §6's
  calibration is per-client and per-round, entirely a function of each
  run's own honest reference data, with no attack-identity-dependent
  branch anywhere in §§6–9's rules.

---

## 14. Validation Sequence

**Stage A — Retrospective validation on already-collected trace data
(the only stage this plan identifies as immediately actionable; still
not performed by this document).** Apply §§6–9's fixed rules and §§11–12's
baselines/ablations directly to the already-saved
`client_round_trace.csv` files for `EXP_014` and `EXP_015` — a pure
offline computation over existing CSV data, requiring no new federated
round, no new attack, and no GPU.

**A concrete, honest constraint on Stage A's scope, found during this
design's own evidence review:** `EXP_014`'s trace predates the
`self_prev_round_cosine` field (it was added specifically for `EXP_015`'s
implementation) and does not contain it, and — critically — the raw
per-round submitted-update tensors needed to compute it retroactively
were never persisted for `EXP_014` (`analysis/PHASE5_Implementation_Verification.md`
item 18 already noted this: only two scalar summaries are saved per
client-round, never the raw update itself). **The temporal signal `z_t`,
baseline C, and every ablation that includes `z_t` therefore cannot be
evaluated on `EXP_014` from existing data — only on `EXP_015`.** `EXP_014`
can still be fully evaluated on baselines A/B and the magnitude-only,
peer-only, and magnitude+peer ablations. This constraint is reported
here as a finding, not resolved by proposing a new `EXP_014`-equivalent
run in this document (that would be exactly the kind of experimental-matrix
expansion this task instructs against without a specific justification
beyond "it would be convenient").

**Stage B — Only after `EXP_015`-informed Stage A results, freeze
FedTrace's exact formulation (thresholds, `T_ref`, `P`, combiner), then
replicate on additional seeds.** Per both benchmarks' own pre-registered
seed strategies (`research_plans/PHASE5_Malicious_Client_Traceability_Plan.md`
§16, `research_plans/EXP_015_Norm_Preserving_Directional_Attack_Plan.md`
§13), the next paired seeds (123, 2026) for each attack family were
already anticipated future work, independent of this design plan. Only
once FedTrace's own parameters are frozen from Stage A should any new
run be considered, and only to evaluate the already-frozen method — not
to re-tune it.

**Stage C — Only if Stage B's results justify it, additional attack
severity/family tests.** `analysis/EXP_014_Attack_Difficulty_Calibration.md`
§§5–8 already identified concrete candidate next steps (a calibrated
lower-`rho` Gaussian region, or a norm-preserving attack that also
persists/correlates its direction across rounds to specifically stress
`z_t`) — these remain candidates only, not committed to by this plan, and
are explicitly deferred until Stage B provides evidence that the frozen
method is worth stress-testing further.

**No experimental matrix expansion beyond this three-stage sequence is
proposed.**

---

## 15. Success Criteria

FedTrace will be considered to **improve over the three single-signal
baselines** only if, in Stage A/B evaluation, **both** of the following
hold simultaneously:

- **Reliable attribution:** the full FedTrace combiner (§11-D) correctly
  identifies the true malicious client (§10's identification-accuracy
  metric) on `EXP_015`, and on `EXP_014` restricted to the
  temporal-signal-unavailable ablations (§14's constraint) — matching or
  exceeding whichever single baseline/ablation performs best on that
  same run, not merely beating the worst one.
- **Useful onset localization:** onset-localization error (§10) is no
  worse, on each benchmarked run, than the best-performing single-signal
  baseline evaluable on that run, **and** detection delay stays bounded
  (consistent with §9's `P=3` design target) rather than degrading to
  `EXP_015`'s own norm-only baseline's 29-round delay.

**Both together, with honest-client false positives controlled:** zero
honest clients reach persistent detection (§8) on either benchmarked
run, under the single, un-tuned-per-attack parameter setting (§13).

---

## 16. Failure Criteria

The proposed FedTrace combination will be considered **not to add
meaningful value** over a simple single-feature detector if any of the
following holds:

- On **every** evaluable run, some single ablation (§12) performs at
  least as well as full FedTrace on **all** of §10's metrics — i.e. the
  three-signal combination never demonstrates an advantage the best
  individual signal did not already provide on its own for that run.
- FedTrace's honest-client false-positive rate is **worse** than the
  best-performing single baseline on the same run (the combination
  introducing more false alarms than any component alone would have).
- Onset-localization error for FedTrace is **not** meaningfully better
  than a trivial baseline that simply guesses "onset = `T_ref`" (the
  calibration-window boundary) regardless of any observed signal — i.e.
  the method's localization is no more informative than the deployment
  policy choice itself.
- The `max` combiner's behavior is dominated by one signal on every run
  evaluated so far (i.e. `z_p` never contributes the maximum on either
  run) — this would not by itself be a failure of attribution, but would
  indicate the three-signal design is not yet justified by evidence
  beyond the two-signal (`m`, `t`) case, and should be reported as such
  rather than silently treated as validating all three.

---

## 17. Scientific Contribution Boundary

**What can potentially be claimed, if Stage A/B succeed:** a
traceability method that identifies a malicious client and localizes the
round its behavior began, under the specific strong-Non-IID,
single-attacker, delayed-onset threat model this project has
characterized in detail (§3), by combining complementary
magnitude/direction/temporal signals such that it remains effective
against attacks that individually defeat any one of those signals alone
— validated (contingent on Stage A/B) across two qualitatively different
attack families (a magnitude-perturbing attack and an exactly
norm-preserving directional attack) at, eventually, multiple seeds.

**What cannot be claimed:**
- **Not** universal Byzantine robustness or universal attack detection —
  the method is validated only against the specific attack families
  actually tested, under one dataset/architecture/aggregation/client-count/heterogeneity
  configuration.
- **Not** robustness to an adaptive attacker who knows FedTrace's exact
  design (e.g. one that deliberately keeps `update_l2_norm`, peer-cosine,
  *and* round-to-round self-consistency all within the honest range
  simultaneously — a strictly harder attack than either benchmarked so
  far, and not yet constructed or tested).
- **Not** a multi-attacker or colluding-attacker result (§3's
  single-attacker scope).
- **Not** generalization beyond CIFAR-100/ResNet-18/FedAvg/5-clients/`alpha=0.1`
  without further validation on other configurations.

**Framing:** the contribution is traceability **specifically under
heterogeneous federated learning** — the ability to separate malicious
signal from the honest heterogeneity noise floor this project has
independently, extensively characterized (`EXP_006`–`EXP_013`) — not a
general-purpose robust-aggregation or anomaly-detection claim.

---

## 18. Decision Gate

**Is existing evidence sufficient to implement the proposed FedTrace
method next?** **Yes, for Stage A specifically** — the three signal
families are motivated by direct, already-collected evidence from two
qualitatively different attacks (§2), the combination rule and
calibration procedure are fully specified and require no new data
collection for `EXP_015` (and a defined subset of `EXP_014`, per §14's
documented constraint), and Stage A is pure offline analysis of
already-saved artifacts. **This document does not implement Stage A** —
per this task's explicit instruction, that remains a future step, to be
undertaken only once this design is reviewed and agreed. No `EXP_016`
config, and no new experiment of any kind, is proposed or justified by
this document; Stages B and C are explicitly gated behind Stage A's own
results (§14) and are not committed to here.
