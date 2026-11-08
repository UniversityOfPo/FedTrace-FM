# Phase 5 — Malicious-Client Detection, Attribution, and Traceability: Study Plan

**Status:** Planning document only. No attack code, no attribution code, and
no new config exist yet. `src/attacks/` and `src/attribution/` remain empty
placeholders (`__init__.py` only). This plan pre-registers the threat
model, the first controlled attack experiment's design, the naive baseline
detector, the evaluation protocol, and the decision criteria **before**
any of that code is written, per this project's standing "protocol before
implementation" workflow (already applied to Phase 4's FedProx study —
`research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md`).

**This document does not implement anything.** No code, config, test,
README, or Research Database file was created or modified in producing
it. No command was executed.

---

## 1. Final Scientific Motivation From EXP_007–EXP_013

`analysis/EXP_013_FedAvg_vs_FedProx_ThreeSeed_Synthesis.md` closed out this
project's heterogeneity/optimization-characterization track:

- **A three-seed (42/123/2026) strong-Non-IID FedAvg baseline is now
  established** (`analysis/EXP_009_Three_Seed_Baseline_Analysis.md`):
  Dirichlet(α=0.1) FedAvg degrades test top-1 by a mean 10.26pp
  (range 8.84–11.69pp) relative to the IID reference, with a
  **2.85-percentage-point purely-honest seed-to-seed noise floor** on test
  top-1 — the single most load-bearing number for everything in this plan.
- **Client sample-count imbalance under Dirichlet(α=0.1) is itself large
  and highly seed-sensitive** (coefficient of variation ≈10.9%–30.9%
  across the three seeds, `analysis/EXP_009_Three_Seed_Baseline_Analysis.md`
  §7) — honest clients already look substantially different from each
  other in both data volume and (by construction of the Dirichlet
  partition) class composition.
- **FedProx (μ=0.01), tried as the next controlled method across the same
  three paired seeds, did not clear its own pre-registered bar**
  (`analysis/EXP_013_FedAvg_vs_FedProx_ThreeSeed_Synthesis.md` §§7, 11–12):
  mean test top-1 difference −0.11pp, direction disagreeing between seeds
  on 5 of 8 metrics, formally meeting the plan's Failure Criteria. FedAvg
  remains this project's primary strong-Non-IID baseline; no μ sweep or
  `local_epochs` change was found justified.
- **The heterogeneity/optimization track has therefore run its
  pre-registered course** for this configuration (§§14–15 of that
  synthesis). Per `research_plans/PHASE4_RESEARCH_STRATEGY.md` §§9, 12,
  and `FEDTRACE_FM_MASTER.md`'s own Research Hypothesis and Novel
  Contributions, the project's actual central, still-completely-untested
  question is whether malicious client contributions leave a detectable
  signature distinguishable from the honest heterogeneity noise floor this
  project has now spent seven experiments (`EXP_006`–`EXP_013`)
  rigorously characterizing. Zero attack, attribution, or repair code
  exists anywhere in this codebase.

This plan is the pre-registered protocol for the next phase: **stop
characterizing honest heterogeneity further, and begin building the
minimum attack + ground-truth + naive-detection infrastructure needed to
test that question directly**, using the already-frozen α=0.1 FedAvg
baseline (seeds 42/123/2026) as the honest reference this entire phase is
built against.

---

## 2. Exact New Research Question

> **Can malicious model updates be distinguished from the honest update
> variability already observed under `alpha=0.1`?**

This is deliberately narrower than "can we detect attacks in federated
learning" in general. It is scoped to this project's exact, already-proven
configuration (CIFAR-100, ResNet-18, 5 clients, Dirichlet α=0.1, 50 rounds,
`local_epochs=1`, FedAvg, seeds 42/123/2026) and it is explicitly framed
*relative to* the honest-heterogeneity noise floor this project has
already measured (§1) — not against a hypothetical IID or noise-free
reference. Per `research_plans/PHASE4_RESEARCH_STRATEGY.md` §5, any
attribution method must be evaluated against a **heterogeneity-aware**
threat model from the outset, because `EXP_009`/`EXP_013` have already
shown honest clients under this exact partition diverge from each other by
an amount that is not small (§1). A method that "detects" outliers under
IID data, where any divergence is presumptively suspicious, would not be
answering this project's actual question.

---

## 3. Threat Model

- **Number of malicious clients (first study): exactly one, out of five.**
  A single-attacker setting is the minimal case that still poses the full
  attribution question ("which one of the five is it") without the
  additional confound of multiple simultaneous attackers interacting with
  each other's signals or with the aggregation weighting in ways this
  project has no prior data to reason about. Scaling to multiple malicious
  clients is explicitly deferred to a later, separate study once the
  single-attacker case is understood.
- **Attacker knowledge assumption: local-only, semi-honest-but-corrupted
  client.** The malicious client has access only to what any ordinary
  `FederatedClient` already has: its own local partition, its own locally
  trained model update each round, and the global model weights it
  receives via `load_global_weights()` (identical to what every honest
  client receives). It has **no** visibility into other clients' data,
  other clients' updates, the server's aggregation logic internals, or
  future rounds. This is the standard, weakest-assumption federated
  threat model and matches `PROJECT_ROADMAP.md` Phase 4's scope (label
  flipping / backdoor / model-gradient poisoning, each a local-knowledge
  attack).
- **Attacker control assumption: the malicious client controls only its
  own submitted model update for the rounds it is selected in.** It
  cannot impersonate another client, submit more than one update per
  round, or alter the server's aggregation, selection, or global-model
  code. Per §8 below, this is enforced architecturally, not just
  assumed.
- **Server trust assumption: the server (`FederatedServer` +
  `FedAvg`/`FederatedRunner`) is fully honest and uncompromised.** This
  phase studies detecting a malicious *client*, not a malicious or
  colluding server — server-side trust is out of scope entirely.
- **Attack timing: begins at a fixed round strictly after round 0, not at
  round 0.** See §7 for the exact chosen round and full justification.
  Every round before that boundary is genuinely, entirely honest — this
  is a deliberate design choice, not an oversight.
- **Why delayed attack onset is required for round-level traceability
  research:** the flagship goal (§14/`FEDTRACE_FM_MASTER.md` Research
  Goal) is not merely "detect that an attack exists" but **identify the
  specific round in which malicious behavior begins**. That question is
  only meaningful, and only testable, if there exists a genuine
  pre-attack period during which the same client behaved honestly — an
  attack active from round 0 has no "before" state to distinguish a
  detector's round-localization signal from simply "this client always
  looked different." A delayed onset also lets every client's early,
  volatile-training-dynamics behavior (rounds 0–19 in this project's own
  round-block convention, `analysis/EXP_009_Three_Seed_Baseline_Analysis.md`
  §6) be observed under fully honest conditions first, so any detector's
  pre-attack reference statistics (§11) are not contaminated by the
  attack itself — an essential requirement for the naive baseline's
  thresholding rule in §11 to be legitimate (derived from labels never
  seen, not from the attack window).

---

## 4. First Attack Family: Additive Gaussian Noise Injection (Model/Update Poisoning)

**Chosen: a single attack type — additive Gaussian noise injected directly
into the malicious client's reported model update, scaled relative to that
update's own L2 norm.** No other attack type is introduced in this plan.

Rationale for this choice over the alternatives named in
`PROJECT_ROADMAP.md` Phase 4 (label flipping, backdoor trigger injection,
model/gradient poisoning):

- **It directly perturbs the model/update, not the data or the loss
  function** — per this task's explicit preference. Label flipping is a
  *data*-poisoning attack: its effect on the submitted update is indirect
  (mediated by an entire local training run on corrupted labels), harder
  to attribute a controlled "strength" to, and conflates two research
  questions (does corrupted local data change training dynamics? does the
  resulting update look different?) that this first study does not need
  to conflate. A backdoor attack requires trigger-pattern design and a
  separate attack-success metric (trigger accuracy) not needed for the
  primary detection/attribution question, and is a strictly more complex
  first experiment. Additive Gaussian noise is the simplest member of the
  "model/gradient poisoning" family already named in the roadmap.
- **Strength is naturally, continuously controllable by a single scalar**
  (the noise-to-update-norm ratio, §5) — unlike a sign-flip attack (binary,
  no meaningful strength dial) or a fixed-magnitude scaling attack (whose
  natural strength parameter, the scale factor, does not have as clean an
  interpretation relative to this project's already-established
  update-norm-based diagnostic features, §10).
- **It is scientifically interpretable in closed form**: the perturbed
  update is `w_malicious = w_honest + N(0, sigma^2 I)` for a
  precisely-defined `sigma` (§5), so any downstream detection signal can
  be reasoned about mathematically, not just empirically.
- **It requires no new hyperparameters beyond the strength scalar** (no
  trigger pattern, no poisoned-label mapping, no target class) — the
  minimal attack that still directly and controllably perturbs the
  submitted update.

Explicitly not chosen for this first study (deferred to later, separate
plans if this study's evidence justifies them): label flipping, backdoor
trigger injection, sign-flipping, update scaling/amplification, and any
multi-attacker or colluding-attacker variant.

---

## 5. Attack-Strength Parameterization

**Strength is represented by a single, dimensionless scalar `rho >= 0`**
("noise-to-update ratio"), defined so that the injected noise's expected
L2 norm is `rho` times the honest update's own L2 norm:

```
delta_honest = w_honest_local - w_global_pre_round   (trainable parameters only)
sigma = rho * ||delta_honest||_2 / sqrt(num_trainable_parameters)
noise ~ N(0, sigma^2 I)   (elementwise, i.i.d., same shape as delta_honest)
delta_malicious = delta_honest + noise
w_malicious_submitted = w_global_pre_round + delta_malicious
```

`rho = 0` recovers plain FedAvg exactly (a required no-op property,
mirroring `mu = 0` recovering plain FedAvg for FedProx,
`research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md` §5). The
perturbation is defined over **trainable parameters only** (never
BatchNorm buffers), matching the FedProx proximal term's own scope
(§5 of that plan) and this project's `FedAvg._max_across_clients`
handling of `num_batches_tracked` (`src/federated/aggregation/fedavg.py`)
— non-trainable buffers are not a meaningful target for a "model update"
perturbation and altering them would risk corrupting BatchNorm running
statistics in a way unrelated to the actual research question.

**Relative (norm-based), not absolute, scaling** is used because this
project's model has parameters spanning very different natural magnitudes
across layers (a 3x3 conv stem vs. a final fully-connected layer); an
absolute noise magnitude tuned for one layer would be meaningless for
another. Scaling by the honest update's own norm keeps the perturbation's
effective size self-calibrating to whatever that client's actual local
training produced that round.

**Frozen at a single value for the first experiment: `rho = 1.0`** — the
injected noise's expected magnitude equals the honest update's own L2 norm
(a substantial, unambiguous perturbation, not a marginal one). No broad
`rho` sweep is proposed; per §19–20's failure criteria and decision tree,
`rho` is only revisited (by one deliberate step, not a search) if the
pilot result is uninformative (too weak or too strong). This mirrors the
single-`mu`-value discipline already applied to FedProx
(`research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md` §11).

---

## 6. Malicious-Client Identity: `client_002`

**Chosen: `client_002`, fixed for this first controlled experiment.**

Justification, using the already-verified per-client sample counts across
all three baseline seeds (`analysis/EXP_009_Three_Seed_Baseline_Analysis.md`
§7):

| Client | Seed 42 | Seed 123 | Seed 2026 | Spread (max−min) |
|---|---:|---:|---:|---:|
| client_000 | 10,883 | 6,024 | 5,233 | 5,650 |
| client_001 | 8,003 | 9,628 | 12,318 | 4,315 |
| **client_002** | **8,868** | **7,202** | **9,171** | **1,969 (smallest)** |
| client_003 | 8,574 | 10,361 | 6,556 | 3,805 |
| client_004 | 8,672 | 11,785 | 11,722 | 3,113 |

`client_002` has the smallest cross-seed spread of any client (1,969
samples) and the sample count closest to the per-client average (9,000)
in every one of the three seeds. Because FedAvg weights each client's
contribution to aggregation by `num_samples / total_samples`
(`src/federated/aggregation/fedavg.py::_weighted_average`), the
*aggregation weight* the attack gets to exploit would otherwise vary
substantially by which client and which seed is chosen — confounding
"how visible is this attack" with "how much sample-weight does this
particular client happen to have in this particular partition draw."
Choosing the client with the most seed-stable sample count and weight
keeps that confound as small as this project's own already-realized
partitions allow, so that if this pilot (§16) is later extended to seeds
123/2026, the malicious client's aggregation influence stays comparable
across all three paired runs — a deliberate echo of the paired-seed
discipline already used for FedProx.

---

## 7. Attack Start Round: 20

**Chosen: round 20 (0-indexed), fixed for this first controlled
experiment.**

Justification, using this project's own established round-block
convention (decade blocks at 0–9, 10–19, 20–29, 30–39, 40–49, used
throughout `analysis/EXP_004_Convergence_Analysis.md` through
`analysis/EXP_013_FedAvg_vs_FedProx_ThreeSeed_Synthesis.md`):

- **Pre-attack honest reference behavior (rounds 0–19, 20 rounds):**
  spans both of the two fastest-changing decade blocks
  (`analysis/EXP_009_Three_Seed_Baseline_Analysis.md` §6 block-gain
  tables: rounds 0–9 ≈3.2pp/round, rounds 10–19 ≈1.4pp/round for FedAvg
  at this alpha) — a full 20-round honest window captures both the
  earliest, most volatile training dynamics and the beginning of the
  slower, more stable convergence phase, giving any per-client baseline
  statistic (§11) a reference period that is not itself atypically
  smooth or atypically volatile relative to the rest of the run.
- **Post-attack detection (rounds 20 onward):** round 20 is exactly the
  boundary where this project's own analyses have repeatedly observed
  validation dynamics becoming markedly smoother and lower-variance
  (block-to-block marginal gains fall by roughly half or more between the
  10–19 and 20–29 blocks in every prior FedAvg run at this alpha). A
  perturbation introduced at the start of this calmer phase is easier to
  distinguish from ordinary per-round noise than one introduced during
  the volatile first two decade blocks, directly serving the detection
  half of the research question.
- **Enough remaining rounds to observe consequences (rounds 20–49, 30
  rounds):** a 30-round post-attack window is 1.5x the length of the
  pre-attack window and spans three full decade blocks (20–29, 30–39,
  40–49) — enough to observe not just an initial detection signal but
  whether it persists, grows, or decays over sustained attacked
  participation, and enough to measure any accumulated global-model
  degradation (§13) against the matched honest baseline's own late-run
  behavior.

`round_id >= 20` is used as `attack_active` for `client_002` in every
subsequent round through round 49 (the attack, once begun, does not turn
itself off partway through this first study — an on/off or intermittent
attack schedule is a separate, more complex design deferred to a later
study).

---

## 8. Attack Injection Point in the Current Architecture

**Exact, safest location: inside `FederatedRunner.run_round()`
(`src/federated/runner.py`), per selected client, immediately after
`epoch_metrics = client.train_one_round()` completes and immediately
before the resulting weights are wrapped into a `ClientUpdate` — i.e.,
intercepting only the return value of `client.get_model_weights()` for
the one designated malicious `client_id`, before it is passed into
`ClientUpdate.create(...)` and `self.server.receive_update(update)`.**

Concretely, in the current per-client loop (`runner.py` lines ~416–438):

```python
client.load_global_weights(pre_round_global_state, round_id=round_id)
epoch_metrics = client.train_one_round()
...
update = ClientUpdate.create(
    client_id=client_id,
    round_id=round_id,
    model_state=client.get_model_weights(),   # <-- interception point
    num_samples=num_samples,
    train_metrics=client_train_metrics,
    metadata={"local_epochs": len(epoch_metrics)},
)
self.server.receive_update(update)
```

A future implementation would (not built by this plan) insert an optional,
injectable attack hook that, only for the designated malicious
`client_id` and only when `attack_active` for that round (§7), replaces
the honest `model_state` dict with the perturbed one (§5) before it
reaches `ClientUpdate.create(...)`.

**Why this is the safest location, satisfying both explicit constraints:**

- **"Do not modify the global model directly."** The global model is only
  ever changed by `self.server.load_global_weights(aggregation_result.aggregated_state)`
  (`runner.py` line ~526), which happens strictly *after*
  `self.aggregation_strategy.aggregate(...)` combines **all** accepted
  updates — including the malicious client's, but through the same
  unmodified `FedAvg.aggregate()` code path every honest run already
  uses. The attack never writes to the global model, the server, or the
  aggregation logic; it only changes what one client *reports*, and the
  server processes that report exactly as it would any other client's.
- **"Do not contaminate honest clients."** `client.load_global_weights(...)`
  and `client.train_one_round()` run **unmodified**, for every client
  including the malicious one — the malicious client's own local training
  is genuine, honest SGD on its own real local data (this is a
  model/update-poisoning attack, not a data-poisoning or training-process
  attack, per §4). The interception happens strictly per-`client_id`
  inside the existing sequential loop, touching only that one client's
  already-computed, already-detached, already-CPU-cloned weights dict
  (`get_model_weights()` already returns an independent copy, per its own
  docstring — "never a live reference into this client's GPU memory").
  No other client's `FederatedClient` instance, dataset, model, optimizer,
  or reported update is read or written by this interception.
- It requires **zero changes** to `FederatedClient`, `FederatedServer`, or
  `FedAvg` — exactly as FedProx required zero changes to `FedAvg`
  (`research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md` §8). A poisoned
  `ClientUpdate` is structurally indistinguishable from an honest one (same
  keys, shapes, dtypes) to every downstream component, so no special-casing
  of the attack is needed anywhere in the aggregation or server code.

---

## 9. Ground-Truth Logging Requirements

Every completed round's record must include, in addition to everything
`round_history.csv` already logs (`selected_client_ids`,
`successful_client_ids`, etc. — already produced unmodified by
`_round_result_row()`), the following ground-truth fields, self-contained
in every row so no later analysis needs to cross-reference the config file
to reconstruct ground truth:

- `malicious_client_id` — fixed for the whole run, but logged on every
  row.
- `attack_active` — boolean per round: `True` iff
  `round_id >= attack_start_round`.
- `attack_type` — fixed string identifier for the whole run (e.g.
  `"gaussian_noise"`, per §4).
- `attack_strength` — the frozen `rho` value (§5), fixed for the whole run.
- `attack_start_round` — fixed for the whole run.

These fields must be unambiguous and available even for rounds *before*
`attack_start_round` (with `attack_active=False`), so the full ground-truth
record is self-describing end to end — this is what later makes it
possible to evaluate a detector's round-localization accuracy (§13)
without any ambiguity about what "true" looks like at every round, not
only during the attack window.

---

## 10. Trace Features to Record (Cheap Baseline, Not Yet FedTrace)

A minimal, cheap diagnostic feature set computed per selected client, per
round, from quantities already available at the exact injection point
(§8) — i.e., `delta_client = model_state_submitted - w_global_pre_round`,
trainable parameters only:

**Cheap baseline signals (this phase, feed the naive detector, §11):**

- **Update L2 norm**: `||delta_client||_2`.
- **Cosine similarity to the same client's own previous-round update**
  (self-trajectory consistency): `cos(delta_client^t, delta_client^{t-1})`.
- **Cosine similarity to the sample-weighted mean update of the other
  selected clients that round** ("peer-agreement"):
  `cos(delta_client, mean_{other selected clients} delta)`.
- **L2 distance from the coordinate-wise (per-parameter) median of that
  round's selected clients' updates** — a simple, cheap robust-center
  reference that does not require any trained/learned model of its own.

**Explicitly deferred to a future FedTrace design (not built, not used by
the naive baseline in this phase):**

- **Layer-wise decomposition** of any of the above (per-layer norm/cosine
  signatures instead of one whole-model-aggregate number per feature) —
  plausibly more informative, but adds a design dimension (which layers,
  how to combine them) this phase's minimal baseline does not need.
- **Any temporal/sequential or learned model** over the per-round feature
  time series (e.g., a trained classifier, an anomaly-detection model
  fit across rounds) — this is the actual attribution *mechanism* FedTrace
  is meant to be; per §21, it is explicitly out of scope for this phase.
- **Validation-impact features** (e.g., per-client held-out-validation
  loss/accuracy before vs. after applying a client's individual update in
  isolation) are allowed **only** insofar as they reuse the project's
  existing, already-leakage-free held-out validation split (never the
  test set, which per this project's standing discipline is touched
  exactly once, at the very end, for final reporting only). No such
  feature is included in the cheap baseline (§11) for this first study —
  it is flagged here only as a legitimate future option, not built now.

---

## 11. Naive Attribution Baseline

**One simple detector, using only the "peer-agreement" cosine-similarity
feature from §10, with a threshold derived exclusively from the pre-attack
window.**

- **Feature**: for each selected client `k` and round `r`,
  `s_k(r) = cos(delta_k(r), mean_{selected clients != k} delta(r))` — how
  aligned client `k`'s update direction is with its co-selected peers'
  average update direction that round. Chosen over raw update norm as the
  primary signal because, per this project's own findings (§1), honest
  clients under Dirichlet(α=0.1) already differ substantially in *update
  norm* purely from unequal sample counts and class composition — a
  norm-only detector would be expected to confuse that honest variability
  with an attack far too easily (this is precisely the failure mode §12
  anticipates and §14 must test for). Directional agreement with peers is
  a materially different signal from magnitude.
- **Threshold rule, derived only from pre-attack honest behavior:** for
  each client `k`, compute `mu_k` and `sigma_k` — the mean and sample
  standard deviation of `s_k(r)` over rounds `r` in `[0, attack_start_round)`
  (i.e., rounds 0–19 only, §7), using **only** that client's own
  observations, entirely before any label of "malicious" could possibly
  be known or used. For every round `r >= attack_start_round`, flag client
  `k` as suspicious in round `r` if `s_k(r) < mu_k - 3 * sigma_k` (a fixed,
  conventional 3-sigma one-sided threshold, chosen as a standard,
  unexciting statistical convention — not tuned against the injected
  attack's actual strength or any post-hoc knowledge of the true
  malicious client).
- **Output**: (a) **suspicious client** — the selected client with the
  most flagged rounds over `[attack_start_round, 49]` (ties broken by the
  most extreme single-round z-score); (b) **suspicious round / first
  flagged round** — the earliest round `>= attack_start_round` at which
  the eventual suspicious client was first flagged.

This is deliberately the simplest possible instantiation of the
"update-norm outlier detection" style baseline already named in
`PROJECT_ROADMAP.md` Phase 5's completion criteria — a per-client rolling
z-score threshold on one cheap directional feature, computed with zero
knowledge of ground truth.

---

## 12. Why the Naive Method Is Expected to Be Imperfect Under Non-IID Heterogeneity

This weakness is essential, not incidental: `research_plans/PHASE4_RESEARCH_STRATEGY.md`
§9 requires a naive baseline be **found to have a specific, documented
weakness** before a purpose-built FedTrace mechanism is justified. Two
concrete, evidence-grounded reasons this baseline is expected to struggle
here, not merely "detectors are imperfect" in the abstract:

- **Every client's own pre-attack reference window (rounds 0–19) is drawn
  from a period this project has already shown to be the fastest-changing,
  highest-marginal-gain phase of training** (§7; `analysis/EXP_009_Three_Seed_Baseline_Analysis.md`
  §6 block-gain tables). A `mu_k`/`sigma_k` baseline estimated there may
  not represent the client's own *honest* peer-agreement behavior during
  the calmer rounds 20–49 — i.e., the threshold could be miscalibrated
  for reasons that have nothing to do with any attack, purely because
  honest training dynamics are not stationary across the run.
- **Honest clients under Dirichlet(α=0.1) already diverge substantially
  from their peers by construction** (§1; client sample-count coefficient
  of variation up to ≈30.9%, and by construction of Dirichlet partitioning,
  correspondingly skewed per-client class composition). An honest client
  with an unusually skewed local class distribution has a real,
  legitimate reason for its update direction to disagree with the
  sample-weighted peer mean in a given round — indistinguishable, from
  this one cheap feature alone, from a client whose update direction was
  deliberately perturbed. This is precisely the scenario
  `FEDTRACE_FM_MASTER.md`'s Research Hypothesis identifies as the open
  problem: "even when [malicious contributions are] individually small
  enough to evade naive outlier detection."

If this baseline instead performs surprisingly well and shows neither
weakness, that is itself an important, reportable result — see the
decision tree (§20).

---

## 13. Evaluation Metrics

Computed against the ground-truth log (§9), separately for each of the
listed items — no metric here substitutes for another:

- **Malicious-client identification accuracy**: whether the naive
  detector's single "suspicious client" output (§11) equals the true
  `malicious_client_id`, for this one run (binary — correct/incorrect,
  since there is exactly one malicious client and one detector output per
  run in this first study).
- **False-positive rate on honest clients**: the fraction of the four
  honest clients' round-level flags (§11's per-round-per-client rule)
  that fire during rounds `>= attack_start_round`, out of all
  honest-client-rounds evaluated — i.e., how often the detector flags a
  client it should not.
- **Attack-round localization error**: `|first_flagged_round -
  attack_start_round|` for whichever client the detector ultimately
  identifies as suspicious (only meaningful if that client is the correct
  one; reported separately from identification accuracy, not conflated
  with it).
- **Detection delay**: number of rounds between `attack_start_round` and
  the first round the *correct* malicious client was actually flagged
  (undefined/reported as "never" if it is never flagged).
- **Precision/recall/F1 for malicious client-round events**: treating
  every `(client, round)` pair with `round >= attack_start_round` as a
  binary classification instance (positive iff `client ==
  malicious_client_id`), compute precision, recall, and F1 of the
  detector's per-round flags against that ground truth — a finer-grained
  metric than the single-client identification accuracy above, capturing
  partial/intermittent detection.
- **Global model top-1/top-5 degradation**: the attacked run's final
  test top-1/top-5 (identical evaluation protocol already used by
  `EXP_007`–`EXP_012`) compared against the matched honest FedAvg
  baseline at the same seed (`EXP_007` for seed 42, per §16's pilot
  choice) — reported as a signed difference, not a claim of significance.
- **Clean-performance preservation before attack onset**: the attacked
  run's own validation trajectory for rounds `0`–`19` (before
  `attack_start_round`) compared against the matched honest baseline's
  rounds `0`–`19` — this should show **no** meaningful difference (both
  runs are identical up to `client_002`'s randomness in local
  training at rounds it is honestly participating in before the attack
  begins), and is a sanity check on the injection point's correctness
  (§8) as much as a research result.

---

## 14. Honest-vs-Malicious Separation Criterion

Reusing this project's established noise-floor methodology (§1): evidence
that the attack signal rises above the honest heterogeneity noise floor
requires comparing the malicious client's post-attack (`round >= 20`)
feature values not against some external or theoretical baseline, but
against the **within-run**, **same-round** spread of the four
simultaneously-participating honest clients in that same attacked run —
an internal control group that is always available and requires no
additional experiment. Concretely: for each post-attack round, compute the
min–max range of `s_k(r)` (§11) across the four honest clients selected
that round. **Separation evidence** exists for a round if `client_002`'s
`s_{002}(r)` falls *outside* that honest min–max range. **No separation**
(the attack is not distinguishable from honest heterogeneity by this
feature, for that round) if `client_002`'s value falls *inside* the range
spanned by its honest, simultaneously-heterogeneous peers. This directly
operationalizes §2's research question at the level of a single,
falsifiable per-round test, and is the criterion §20's decision tree acts
on.

---

## 15. First Controlled Experiment Design

**Provisional ID: `EXP_014`.**
**Provisional name: `cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed42`.**

Exactly **one** scientific variable changes relative to the honest
Dirichlet(α=0.1) FedAvg baseline at the matching seed (`EXP_007`, seed
42): **the presence of the attack scenario defined in §§4–9** — one
malicious client (`client_002`, §6), one attack type (additive Gaussian
noise, §4), one frozen strength (`rho=1.0`, §5), one frozen start round
(round 20, §7). Every other setting is frozen exactly as `EXP_007`:
CIFAR-100, `resnet18_cifar`, 5 clients, `clients_per_round=5`,
`minimum_updates=5`, `communication_rounds=50`, `local_epochs=1`, SGD
(`learning_rate=0.01`, `momentum=0.9`, `weight_decay=0.0005`),
`mixed_precision=true`, `aggregation=fedavg`, `partition=dirichlet`,
`dirichlet_alpha=0.1`, `validation_fraction=0.1`, `evaluate_every=1`,
`checkpoint_every=1`, `seed=42`. No config file for `EXP_014` is created
by this plan (per this task's explicit "do not create configs"
instruction) — this section only fixes what that future config must
specify.

---

## 16. Seed Strategy: Seed 42 Pilot First, Not All Three Immediately

**Recommendation: run `EXP_014` at seed 42 only, as a pilot, before
committing to seeds 123 and 2026.**

Justification:

- **This is the first time attack-injection, ground-truth logging, and
  detector code will ever execute in this codebase.** Every prior
  single-variable change in this project's history (round-budget
  extensions, the cosine scheduler, Dirichlet partitioning, FedProx) was
  built on top of already-proven infrastructure; this one introduces an
  entirely new code path (§8's injection hook, §9's logging, §11's
  detector) with correctness properties (§22) that must first be verified
  on real hardware, not only in CPU tests, before three full ≈70–100
  minute GPU runs (per-run durations observed across `EXP_007`–`EXP_012`)
  are committed to.
- **This project has direct, recent evidence that single-seed results can
  mislead** — `analysis/EXP_010_FedProx_Seed42_Analysis.md` alone read as
  "effectively neutral," but `analysis/EXP_013_FedAvg_vs_FedProx_ThreeSeed_Synthesis.md`
  §6 later showed seeds 42 and 123 disagree in sign on 5 of 7 compared
  metrics. The same caution applies here in reverse: a single pilot seed
  is being deliberately used **only** to validate that the pipeline
  executes correctly, produces sane ground-truth logs, and yields *some*
  measurable signal (§17) — not to draw any conclusion about detection
  efficacy in general. No claim about detector performance "in general"
  is licensed by a seed-42-only pilot; §20's decision tree explicitly
  gates full three-seed replication behind the pilot succeeding as a
  pipeline-correctness check first.
- **Seed 42 is this project's own established first-of-kind pilot seed**
  — every genuinely new capability before the FedProx paired-seed
  discipline began (`EXP_002`–`EXP_006`) was first run at seed 42 alone;
  the three-seed discipline (`EXP_007`–`EXP_009` onward) was adopted only
  once the underlying mechanism (plain FedAvg under Dirichlet
  partitioning) was already known to work. Attack injection has not yet
  reached that bar.
- Once the seed-42 pilot confirms (a) the run completes 50/50 rounds
  without error, (b) the ground-truth log (§9) is correct and complete,
  and (c) some non-degenerate signal exists (neither §19's "invisible"
  nor "the run crashed" failure modes), the two remaining paired seeds
  (123, 2026, matched to `EXP_008`/`EXP_009`) should be run to bring this
  question up to the same three-seed standard as every other claim in
  this project since `EXP_007`.

---

## 17. Success Criteria for the Attack Simulation Itself

`EXP_014` will be considered a **successful attack simulation** (a
prerequisite for any detection claim, independent of whether the naive
detector succeeds) if:

- The run completes all 50/50 communication rounds with 5/5 successful
  client updates every round (matching the reliability record of every
  prior experiment `EXP_002`–`EXP_012`) — the attack must not crash the
  pipeline or produce non-finite values.
- The ground-truth log (§9) is complete and internally consistent for
  every round (correct `attack_active` boolean transition exactly at
  round 20, fixed `malicious_client_id`/`attack_type`/`attack_strength`/
  `attack_start_round` on every row).
- `client_002`'s own per-round update statistics (§10) are measurably
  different between the pre-attack window (rounds 0–19, where they must
  be statistically indistinguishable from an honest run — see §13's
  "clean-performance preservation" metric) and the post-attack window
  (rounds 20–49) — i.e., the perturbation is verifiably taking effect,
  not silently a no-op.
- Rounds 0–19 of `EXP_014` show no meaningful deviation from `EXP_007`
  (same seed, same partition, attack not yet active) — this is a
  correctness check on the injection point (§8), not a research finding.

---

## 18. Success Criteria for the Naive Detector

The naive detector (§11) will be considered to have **produced a usable,
interpretable result** (not necessarily a "correct" or "good" one — see
§12) if:

- It outputs a single, well-defined suspicious client and first-flagged
  round (§11) — no crash, no undefined behavior (e.g., division by zero
  if a client's pre-attack `sigma_k` happens to be numerically
  degenerate; this must be handled explicitly, not silently produce a
  meaningless flag).
- Its behavior is fully explainable from §13's metrics — every reported
  metric (identification accuracy, false-positive rate, localization
  error, detection delay, precision/recall/F1) is computable and
  non-degenerate (not literally undefined because, e.g., the detector
  never flags anyone).
- Whether it correctly identifies `client_002`, or not, is itself the
  scientific result this phase is measuring — both outcomes are "usable"
  for the purposes of this criterion; only a crash, an undefined output,
  or a result that cannot be evaluated against §13's metrics counts as
  failure at this stage.

---

## 19. Failure Criteria

`EXP_014` (attack + naive detector, jointly) will be considered to have
produced an **uninformative** first result under any of the following:

- **Too weak / invisible**: `client_002`'s post-attack feature values
  (§10) show no measurable deviation from its own pre-attack distribution
  and no separation from the honest peer range (§14) at any round —
  `rho=1.0` was not a strong enough perturbation to matter, or the
  injection is not taking effect (must first rule out an implementation
  defect per §22's tests before concluding the attack itself is too weak).
- **Too strong / degenerate**: the perturbed update causes non-finite
  loss, a failed round (`round_status="failed"`), a rejected update (a
  shape/dtype mismatch — should not occur given §5's design, but must be
  checked), or global-model collapse (test accuracy catastrophically
  worse than any prior run in this project) — `rho=1.0` was too
  aggressive for a first, interpretable study.
- **Trivial**: the naive detector achieves perfect identification with
  zero false positives and one-round detection delay, *and* §14's
  separation criterion holds with a very large margin at every post-attack
  round for every honest-vs-malicious comparison — the attack was too
  easy to be a useful benchmark for later demonstrating any purpose-built
  method's added value (per `research_plans/PHASE4_RESEARCH_STRATEGY.md`
  §9's requirement that the naive baseline have a *specific, documented
  weakness*).
- **Not scientifically useful**: any outcome where the pre-attack window
  (rounds 0–19) itself already shows a deviation from `EXP_007` at the
  same seed — this would indicate a bug in the injection point (§8)
  contaminating supposedly-honest rounds, invalidating the entire premise
  of a "delayed onset" study (§3) regardless of what happens after round
  20.

---

## 20. Decision Tree

- **If the attack is invisible** (§19's "too weak" case): increase `rho`
  by one deliberate step (e.g., to `rho=2.0`) and re-run the seed-42 pilot
  only — not a sweep, not all three seeds — to check whether a stronger,
  still-singly-chosen value produces a measurable signal before deciding
  anything further.
- **If the attack is trivially obvious** (§19's "trivial" case): decrease
  `rho` by one deliberate step (e.g., to `rho=0.5`) and re-run the seed-42
  pilot only, **or** retain the `rho=1.0` result as a documented
  "positive control" (proof the pipeline and detector both work correctly
  end to end) while separately planning a harder/subtler configuration —
  do not treat a trivially-detected attack as evidence the malicious/honest
  separation question (§2) is answered in general.
- **If the attack is detectable but confused with honest Non-IID
  variation** (§11's detector shows a non-trivial false-positive rate on
  honest clients, or §14's separation criterion fails for some
  post-attack rounds despite §17's attack-simulation success criteria
  being met): **this is the expected, target outcome** per
  `FEDTRACE_FM_MASTER.md`'s Research Hypothesis and
  `research_plans/PHASE4_RESEARCH_STRATEGY.md` §9's evidence bar. Proceed
  to (a) replicate at seeds 123/2026 (§16) before drawing any conclusion
  from more than one seed, and (b) formally document this naive-baseline
  weakness as satisfying the evidence gate for beginning FedTrace design
  in a later, separate phase — not started by this plan.
- **If the naive detector works surprisingly well** (perfect or
  near-perfect identification with low false positives across all three
  seeds once replicated): do **not** conclude a purpose-built attribution
  mechanism is unnecessary from this alone, and do **not** immediately
  design FedTrace either (§21). Per §12's requirement that a weakness be
  demonstrated, the correct next action is to select a harder variant —
  e.g. a smaller `rho`, or (in a later, separate plan) a genuinely
  different attack type expected to be harder to detect via this cheap
  feature set — as the next controlled experiment, rather than concluding
  the naive baseline suffices in general from one easy configuration.

---

## 21. Explicit Boundary: What This Phase Does Not Do

**This plan does not design the final FedTrace attribution algorithm.**
Its entire purpose is to (a) build a trustworthy, ground-truth-labeled
attack benchmark (§§3–9) on top of the already-established honest
heterogeneity baseline, and (b) expose — not fix — the naive baseline
detector's specific weakness under honest Non-IID heterogeneity (§12),
per the evidence-gate logic already laid out in
`research_plans/PHASE4_RESEARCH_STRATEGY.md` §9 and
`PROJECT_ROADMAP.md` Phase 5's own completion criteria (which require a
naive-baseline comparison to exist *before* FedTrace is evaluated against
it). Any purpose-built attribution mechanism, learned detector, repair
mechanism, layer-wise or temporal feature engineering beyond §10's cheap
baseline, multi-attacker threat models, or attack types beyond §4's single
choice are all explicitly out of scope for this plan and deferred to
subsequent, separately pre-registered plans once this phase's evidence
(§20) justifies them.

---

## 22. Required Implementation Tests Before Any GPU Run

All fast, CPU-only, tiny-synthetic-data tests, per this project's existing
testing rules (`FEDTRACE_FM_MASTER.md` → Testing Rules) — no GPU required
to pass, mirroring the discipline already applied in
`research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md` §18:

1. **Selective-perturbation test**: on a tiny synthetic multi-client round,
   the attack hook perturbs only the designated malicious client's
   `model_state`; every other (honest) client's `model_state` is
   byte-identical to what it would be with no attack hook present at all.
2. **Pre-attack no-op test**: with `round_id < attack_start_round`, the
   attack hook produces a `model_state` byte-identical to the unperturbed
   input, for the malicious client too — verifying rounds before the
   attack start are genuinely untouched (directly supports §19's "not
   scientifically useful" failure check).
3. **Closed-form strength test**: given a fixed `rho`, a fixed honest
   `delta_honest` on a tiny synthetic tensor, and a fixed RNG seed, the
   perturbed update's expected noise L2 norm matches the closed-form value
   in §5 within a defined numerical tolerance.
4. **`rho=0` recovers plain FedAvg test**: with `rho=0`, the "attacked"
   `model_state` is byte-identical to the unperturbed one, at any round
   including rounds `>= attack_start_round` — a required no-op property
   (§5), mirroring `mu=0` for FedProx.
5. **Ground-truth logging test**: a tiny synthetic multi-round run produces
   log rows with the exact expected `malicious_client_id`/`attack_active`/
   `attack_type`/`attack_strength`/`attack_start_round` values for rounds
   strictly before, exactly at, and strictly after `attack_start_round`.
6. **Aggregation non-interference test**: a poisoned `ClientUpdate`
   (perturbed `model_state`, same keys/shapes/dtypes as an honest one)
   passes through `FedAvg.aggregate()` (unmodified) exactly like an honest
   update — no special-casing needed, no exception raised solely because
   the tensor values are perturbed.
7. **Honest-client non-contamination test**: on a tiny synthetic multi-round
   federated run, every honest client's own `FederatedClient` internal
   state (model weights, optimizer state, `round_history`) is identical
   whether or not the attack hook is active for the malicious client that
   round — proving the injection point (§8) is a pure post-hoc
   report-time interception with zero side effects on any other client.
8. **Naive-detector correctness test**: given a tiny synthetic set of
   per-round `s_k(r)` feature values engineered with a known-injected
   outlier `(client, round)`, the §11 threshold rule correctly flags the
   expected pair and does not flag any of the honest synthetic clients.
9. **Threshold-window isolation test**: on synthetic data engineered so
   that including post-attack-window values in the baseline computation
   would produce a materially different (and wrong) threshold, verify the
   detector's `mu_k`/`sigma_k` are computed **only** from
   `[0, attack_start_round)` and are provably unaffected by any
   post-attack-window data.

---

## 23. Reproducibility and Logging Requirements

- All attack parameters (`attack_type`, `malicious_client_id`,
  `attack_strength`, `attack_start_round`, and any RNG seed the attack's
  own randomness uses) must be recorded in the experiment's config and
  echoed into `summary.json`, exactly as every other experiment parameter
  already is (`experiment_id` through `git_commit`, per
  `src/experiments/run_federated_cifar100.py::main()`'s existing
  `summary` payload).
- The attack's own Gaussian noise draws (§5) must use a reproducible,
  explicitly-seeded RNG (derived deterministically from the experiment's
  main `seed`, not from an unseeded global RNG state), so re-running an
  identical config reproduces an identical sequence of perturbed updates
  — matching this project's existing reproducibility-first discipline
  (`FEDTRACE_FM_MASTER.md` → Major Design Decisions).
- `run.log` must record a clear, greppable line identifying attack
  activation at `round_id == attack_start_round`, and thereafter on every
  active round, mirroring the existing `round_callback` logging
  convention (`run_federated_cifar100.py::round_callback`).
- The ground-truth fields (§9) must be persisted to a durable output
  artifact (CSV or JSON, analogous to `round_history.csv`/
  `client_partition_summary.csv`), not left implicit in `run.log` text
  alone — later attribution-evaluation analysis must never need to
  re-derive ground truth by parsing log text.
- The existing test-set-touched-exactly-once discipline is preserved
  without exception: neither the attack hook, the trace features (§10),
  nor the naive detector (§11) may read the held-out test set at any
  point during training or detection — only the already-existing,
  leakage-free held-out validation split (or the client updates
  themselves) may be used, and the test set continues to be evaluated
  exactly once, after training, exactly as every prior experiment in this
  project has done.

---

## 24. PI Approval Checklist

Before any part of this plan is implemented or executed, confirm:

- [ ] This plan has been reviewed and agreed as the correct next phase,
      given `analysis/EXP_013_FedAvg_vs_FedProx_ThreeSeed_Synthesis.md`
      §§14–15's assessment that the heterogeneity/optimization track has
      run its course and the project's own FedTrace-design gate
      (`research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md` §21) is not
      yet met.
- [ ] The single-attacker, local-knowledge-only, honest-server threat
      model (§3) is agreed as the correct first threat model, with
      multi-attacker and server-trust extensions explicitly deferred.
- [ ] The choice of additive Gaussian noise injection as the sole first
      attack type (§4), over label flipping/backdoor/sign-flip/scaling,
      is reviewed and agreed.
- [ ] The frozen first-experiment choices — `rho=1.0` (§5), `client_002`
      (§6), attack start round 20 (§7) — are agreed as reasonable,
      literature-and-project-grounded starting points, explicitly not
      claimed optimal, mirroring the FedProx plan's own `mu=0.01`
      framing.
- [ ] The injection point (§8) — inside `FederatedRunner.run_round()`,
      intercepting only `client.get_model_weights()`'s return value for
      the designated malicious client, before `ClientUpdate.create(...)`
      — is reviewed and agreed as architecturally safe (no global-model or
      honest-client contamination) before any code is written.
- [ ] The ground-truth logging schema (§9) and the cheap trace-feature set
      (§10), including the explicit boundary against layer-wise/temporal/
      learned features for this phase, are agreed.
- [ ] The naive detector's exact definition (§11) — peer-agreement cosine
      similarity, pre-attack-only 3-sigma threshold — and its expected
      weakness (§12) are agreed before any results are seen, so the
      evaluation cannot be retroactively rationalized.
- [ ] The evaluation metrics (§13) and the honest-vs-malicious separation
      criterion (§14) are agreed in advance.
- [ ] The seed-42-pilot-first strategy (§16), success/failure criteria
      (§§17–19), and decision tree (§20) are agreed in advance, before any
      GPU run.
- [ ] The explicit boundary (§21) — no FedTrace algorithm design in this
      phase — is agreed and will be respected regardless of how
      favorable or unfavorable the pilot result looks.
- [ ] All required tests (§22) will be implemented and passing on CPU
      before any GPU run is attempted.
- [ ] No other file (code, config, test, prior experiment result, or
      Research Database entry) will be modified as part of implementing
      this plan until its own results are ready for documentation.
