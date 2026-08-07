# EXP_010 — FedProx Under Strong Non-IID Heterogeneity (α=0.1) Study Plan

**Status:** Planning document only. `EXP_010` has **NOT** been implemented or
executed. No config file exists yet. FedProx does not exist in this
codebase yet — `src/federated/aggregation/fedavg.py` is the only concrete
`AggregationStrategy`, and no proximal-term logic exists anywhere in
`src/federated/client.py` or `src/training/engine.py`. This plan
pre-registers the design, the paired-seed protocol, and the decision
criteria **before** any of that code is written, per this project's
standing "protocol before implementation" workflow.

---

## 1. Scientific Motivation

`analysis/EXP_009_Three_Seed_Baseline_Analysis.md` established, across
three independent seeds (42, 123, 2026), that plain FedAvg under
Dirichlet(α=0.1) heterogeneity produces a substantial, reproducible
accuracy degradation relative to both the IID reference (`EXP_004`,
68.52% test top-1) and the centralized baseline (`EXP_001`, 62.71%) — a
mean gap of 10.26pp to `EXP_004`, with every individual seed falling
below `EXP_001`. `research_plans/PHASE4_RESEARCH_STRATEGY.md` § 7 set an
explicit evidence bar for implementing a heterogeneity-mitigation
strategy, and § 11 of that memo already named FedProx as the next
controlled method once that bar was met. This plan is the pre-registered
protocol for that experiment: test whether FedProx's proximal-term
correction measurably narrows the α=0.1 gap, using the same three seeds
already characterized, so the comparison is paired rather than a fresh,
uncontrolled sample.

## 2. Exact Research Question

**Does FedProx reduce the α=0.1 degradation relative to paired FedAvg
baselines?** Precisely: for each seed `s ∈ {42, 123, 2026}`, does
`FedProx(mu, seed=s)`'s final test top-1 accuracy exceed
`FedAvg(seed=s)`'s (`EXP_007`/`EXP_008`/`EXP_009` respectively) by a
margin large enough to be distinguishable from the seed-to-seed noise
already measured in the FedAvg-only baseline (2.85pp range, § 3 of
`analysis/EXP_009_Three_Seed_Baseline_Analysis.md`)? This is a narrower,
more precise question than "does FedProx help under heterogeneity in
general" — it is scoped to this exact configuration and these three
specific seeds.

## 3. Paired-Seed Design Using Seeds 42, 123, 2026

`EXP_010` consists of **three FedProx runs**, one per seed, each paired
against the already-completed FedAvg run at the same seed:

| Seed | FedAvg baseline (already run) | FedProx run (planned) |
|---|---|---|
| 42 | `EXP_007` (`configs/cifar100_fedavg_dirichlet_a01_50rounds.yaml`) | `EXP_010a` (not yet created) |
| 123 | `EXP_008` (`configs/cifar100_fedavg_dirichlet_a01_seed123_50rounds.yaml`) | `EXP_010b` (not yet created) |
| 2026 | `EXP_009` (`configs/cifar100_fedavg_dirichlet_a01_seed2026_50rounds.yaml`) | `EXP_010c` (not yet created) |

Per `analysis/EXP_009_Three_Seed_Baseline_Analysis.md` § 12, reusing these
exact three seeds (rather than a fresh set) is required so that each
FedProx run shares its **identical Dirichlet partition realization** with
its paired FedAvg baseline (same client sample counts, same per-client
class composition) — isolating the aggregation/local-objective change as
the only source of any observed difference, rather than confounding it
with a new, uncontrolled partition draw. § 7 of that analysis showed
partition-realization variance alone (client sample-count coefficient of
variation ranging ≈10.9%–30.9% across these three seeds) is large enough
to matter; an unpaired comparison would not be able to separate that
effect from a genuine FedProx effect.

## 4. Everything Frozen From EXP_007–EXP_009

Every value in `configs/cifar100_fedavg_dirichlet_a01_50rounds.yaml` /
`_seed123_50rounds.yaml` / `_seed2026_50rounds.yaml` remains unchanged
for the corresponding paired FedProx run **except** the new FedProx-related
field(s) (§ 5–9): `partition: dirichlet`, `dirichlet_alpha: 0.1`,
`num_clients: 5`, `clients_per_round: 5`, `minimum_updates: 5`,
`communication_rounds: 50`, `local_epochs: 1`, `batch_size: 32`,
`num_workers: 2`, `model: resnet18_cifar`, `num_classes: 100`,
`optimizer: sgd`, `learning_rate: 0.01`, `momentum: 0.9`,
`weight_decay: 0.0005`, `scheduler: none`, `mixed_precision: true`,
`validation_fraction: 0.1`, `evaluate_every: 1`, `checkpoint_every: 1`,
`device: auto`, `fail_round_on_client_error: true`,
`continue_after_failed_round: false`, and — critically — the matching
`seed` per pair (42 / 123 / 2026). The client-state reset-every-round
policy (`FederatedClient.reset_training_state()`, unconditionally
rebuilding a fresh optimizer with zero momentum every round) is also
**not** changed by this plan — see § 6/17.

## 5. Exact FedProx Objective and Proximal Term

Per Li et al., 2020 ("Federated Optimization in Heterogeneous
Networks"), FedProx modifies only the **local** training objective. For
client `k` at communication round `t`, instead of minimizing the plain
local loss `F_k(w)`, the client minimizes:

```
h_k(w; w^t) = F_k(w) + (mu / 2) * ||w - w^t||^2
```

where:
- `w` is the client's local model parameters, being optimized by local
  SGD during the round.
- `w^t` is the **global** model's parameters at the start of round `t` —
  fixed for the duration of that round's local training (not updated as
  `w` changes).
- `mu >= 0` is the proximal-term coefficient. `mu = 0` recovers plain
  FedAvg exactly.
- `F_k(w)` is this project's existing local loss — `nn.CrossEntropyLoss`
  (with `label_smoothing`, per `FederatedClientConfig`) — unchanged.

The proximal penalty is defined over **trainable parameters only**
(`model.parameters()`), not over non-trainable buffers (BatchNorm
`running_mean`/`running_var`/`num_batches_tracked`). This matches the
original FedProx formulation (`w` denotes the model's parameter vector)
and deliberately excludes the exact buffers `FedAvg._max_across_clients`
(Sprint 6.1) already treats specially — a proximal penalty on
`num_batches_tracked` would be mathematically meaningless (it is an
integer counter, not a continuously-optimized parameter).

## 6. Where the Proximal Term Should Be Computed

**In `src/training/engine.py::train_one_epoch`, added to the per-batch
loss immediately after the existing `loss = criterion(outputs, targets)`
line**, as two new optional parameters (`proximal_mu: float = 0.0`,
`global_params: list[torch.Tensor] | None = None`), both defaulting to a
fully inert no-op:

```python
if proximal_mu > 0 and global_params is not None:
    proximal_term = sum(
        (p - g).pow(2).sum() for p, g in zip(model.parameters(), global_params)
    )
    loss = loss + (proximal_mu / 2) * proximal_term
```

This is the correct location for three reasons: (1) mathematically, the
proximal term is a per-batch loss addition, not a per-epoch or per-round
operation; (2) `train_one_epoch` is the **only** place local gradients are
computed and backpropagated in this codebase (Sprint 3.1's design
explicitly reuses it, unmodified, for every client), so adding the term
here — rather than duplicating a training loop inside
`FederatedClient` — follows this project's established "shared behavior
lives in one place, is imported, not copy-pasted" rule; (3) both new
parameters default to values that make the added code path a true no-op,
so `src/train_centralized.py` (which also calls `train_one_epoch`) is
**provably unaffected** — the same backward-compatibility discipline
already used for `round_id`, `round_lr_schedule`, and every other
optional addition in this series.

**Do not** implement the proximal term inside `FederatedClient` by
hand-rolling a separate training loop — that would duplicate
`train_one_epoch`'s existing batch loop, AMP handling, and gradient-clip
logic, violating this project's own coding rules.

## 7. Required Global-Model Snapshot Behavior on Each Client

`FederatedClient.load_global_weights(state_dict, round_id)` is the
single point where a client receives the round's global model. It must
be extended to also **snapshot** that global state as the FedProx
reference point `w^t`, immediately after `self.model.load_state_dict(state_dict)`:

- Store a new attribute, e.g. `self._global_reference_params: list[torch.Tensor]
  | None`, built from **`self.model.parameters()`** (trainable parameters
  only, in the same order `train_one_epoch` will iterate them) —
  `.detach().clone()`'d so it is a true independent copy, not a live
  reference into `self.model`'s own tensors (mirroring the existing
  `get_model_weights()` convention: `detach().cpu().clone()`, or
  on-device if avoiding a host round-trip is preferred for speed —
  an implementation detail to settle when this is actually built, not
  fixed by this plan).
- This snapshot **must be retaken every round** (i.e., every call to
  `load_global_weights`), since `w^t` changes every round by definition.
  A stale snapshot from an earlier round would silently implement a
  different, incorrect algorithm.
- The snapshot must **not** be affected by the existing
  `reset_training_state()` call that immediately follows within
  `load_global_weights()` — that method only rebuilds the
  optimizer/scheduler/scaler, and does not touch `self.model`'s
  parameters or any new snapshot attribute, so no interaction is expected,
  but this must be verified by a test (§ 18) rather than assumed.
- `train_one_round()` passes `self._global_reference_params` and
  `self.config.fedprox_mu` (§ 9) into every `engine.train_one_epoch(...)`
  call for that round (all `local_epochs` calls use the **same** frozen
  snapshot — with `local_epochs=1` in this project's configuration, this
  is only one call per round in practice, but the design must be correct
  for `local_epochs > 1` too, since that is a plausible future SCAFFOLD
  stress test per `research_plans/PHASE4_RESEARCH_STRATEGY.md` § 8).

## 8. Aggregation Remains Sample-Weighted FedAvg

**Yes — unchanged.** This is an important architectural clarification:
FedProx, as defined in § 5, modifies only each client's **local**
objective. The server-side combination step — how accepted
`ClientUpdate`s are combined into a new global model — is **identical**
sample-count-weighted averaging in both FedAvg and FedProx per the
original paper. `src/federated/aggregation/fedavg.py`'s `FedAvg` class
requires **zero changes** for this plan; the same `FedAvg()` strategy
object already used by `EXP_007`–`EXP_009` is reused unmodified for the
FedProx runs.

This refines a forward-looking statement in `FEDTRACE_FM_MASTER.md` →
*Major Design Decisions*, which lists "future aggregation strategies
(FedProx, SCAFFOLD, etc.) plug in via the `AggregationStrategy`
interface" as an example of this project's strategy-agnostic design.
That statement is not literally correct for FedProx specifically:
`AggregationStrategy.aggregate(global_state, updates)` only ever sees
already-computed `ClientUpdate`s — it has no visibility into, or
influence over, how a client computed its local update in the first
place, so it is architecturally **incapable** of expressing a
local-objective change like FedProx's proximal term. FedProx must be
implemented as a client-side (`FederatedClient`/`train_one_epoch`)
change, as specified in §§ 6–7, with the experiment config's
`aggregation` field remaining `fedavg` throughout. (SCAFFOLD, by
contrast, genuinely does require new information to flow through the
aggregation step — its control variates — so that forward-looking
statement remains accurate for SCAFFOLD; only the FedProx case is being
corrected here.)

## 9. A Principled Initial `mu` Value

**Proposed initial value: `mu = 0.01`.**

## 10. Justification for `mu = 0.01` (Not Claimed Universal)

- It sits within the range Li et al. (2020) tested in their original
  FedProx experiments (`mu ∈ {0.001, 0.01, 0.1, 1}` across benchmarks),
  and is the value most commonly cited in follow-up literature as a
  reasonable first default when no prior tuning exists for a new
  architecture/dataset combination.
- It is the same order of magnitude as this project's own fixed
  `learning_rate=0.01` — a defensible heuristic anchor in the absence of
  any tuning: a proximal-term gradient contribution `mu*(w - w^t)` on a
  similar scale to the primary task gradient is unlikely to either
  overwhelm learning early on or be too small to have any measurable
  effect.
- With `local_epochs=1` (frozen from `EXP_007`–`EXP_009`, § 4), each
  client only takes one local epoch's worth of SGD steps before the next
  synchronization — the local model cannot drift arbitrarily far from
  `w^t` within a single round regardless of `mu`. This argues against
  starting with an aggressive (large) `mu`, since the drift FedProx
  corrects for is inherently bounded here; a moderate value is more
  appropriate than either an extreme value or `mu ≈ 0`.

**This is explicitly not claimed to be the optimal `mu` for this
project's specific architecture, dataset, or heterogeneity level.** No
tuning or prior sweep has been performed to justify optimality — `mu =
0.01` is a literature-grounded, defensible starting point for a first
controlled study, nothing more.

## 11. Is One `mu` Sufficient for the First Controlled Study?

**Yes.** Consistent with this project's single-variable-change discipline
applied throughout `EXP_002`–`EXP_009` (one round-budget change at a
time, one scheduler change, one `alpha` value at a time, one seed at a
time), `EXP_010` changes exactly one thing relative to the paired FedAvg
baselines: `fedprox_mu: 0 → 0.01`. Per the explicit instruction governing
this plan, **a broad `mu` sweep is not recommended unless the evidence
from this single-`mu` study genuinely requires it** — e.g., a highly
ambiguous or borderline result (§ 19) might justify a small, targeted
follow-up, but a sweep is not proposed here as a default next step.

## 12. Success Criteria

`EXP_010` will be considered evidence that FedProx **measurably helps**
under this configuration if:

- The mean paired difference (`FedProx test top-1 − paired FedAvg test
  top-1`, averaged over the three seeds) is **positive and clearly larger
  than the 2.85-percentage-point range already observed as pure
  seed-to-seed FedAvg-only noise** (`analysis/EXP_009_Three_Seed_Baseline_Analysis.md`
  § 3) — a smaller improvement cannot be confidently distinguished from
  noise already characterized in this exact configuration.
- The paired difference is positive in **at least 2 of the 3 seeds**
  (ideally 3 of 3) — not driven by a single outlier seed.
- All three FedProx runs complete 50/50 rounds with 5/5 successful
  clients and 0 rejected updates, matching the reliability record of
  every prior experiment in this series.

## 13. Failure Criteria

`EXP_010` will be considered evidence that FedProx (at `mu=0.01`) **does
not help** under this configuration if:

- The mean paired difference is negative, zero, or smaller than the
  2.85pp noise floor (§ 12) — i.e., not distinguishable from ordinary
  seed variance.
- The paired difference is positive in only 0 or 1 of the 3 seeds.
- Any FedProx run is unstable (non-finite loss, a failed round, or
  materially worse final accuracy than its paired FedAvg baseline by a
  large margin) — this would need root-causing (e.g. a `mu` too large
  for this setup) before being treated as evidence against FedProx in
  general.

## 14. Per-Seed Paired-Comparison Rules

- Compute `paired_diff(s) = FedProx_test_top1(s) − FedAvg_test_top1(s)`
  for each `s ∈ {42, 123, 2026}` individually, using `EXP_007`/`EXP_008`/
  `EXP_009` as the respective FedAvg baselines.
- **Report all three paired differences individually before any
  averaging** — do not lead with the mean. If the three differences
  disagree in sign, that disagreement must be reported and discussed
  explicitly, not smoothed over (matching the standard already set by
  `analysis/EXP_009_Three_Seed_Baseline_Analysis.md` § 6/8 when a
  2-seed pattern failed to hold at the third seed).
- Each pair must be compared using **identical** partition realization
  (verified via matching `client_partition_summary.csv` client counts
  between the FedAvg and FedProx run at the same seed) before any
  accuracy comparison is trusted — this is the entire point of the
  paired design (§ 3) and must be checked, not assumed.

## 15. Aggregate-Comparison Rules

- Report the mean and the range (min–max) of the three paired
  differences — no other aggregate statistic.
- **Do not compute or report a standard error, confidence interval, or
  p-value for the paired differences.** n=3 pairs is not sufficient for
  any such estimate to be meaningful (§ 22).
- Any aggregate comparison must also report the mean and range of the
  underlying FedProx test top-1 values themselves (not just the deltas),
  exactly as `analysis/EXP_009_Three_Seed_Baseline_Analysis.md` § 3 did
  for the FedAvg baselines, so the two three-seed summaries remain
  directly comparable.

## 16. Runtime and Communication-Cost Expectations

- **Communication cost: unchanged.** FedProx does not alter what is
  transmitted between client and server — the same model `state_dict`
  shape/size is sent each round, exactly as in `EXP_007`–`EXP_009`. No
  new tensors (e.g. control variates) cross the client/server boundary,
  unlike SCAFFOLD.
- **Per-batch compute overhead: expected to be small.** The proximal
  term requires one extra elementwise-squared-difference-and-sum pass
  over all 11,220,132 trainable parameters per batch, plus its
  backward-pass contribution — cheap relative to a full ResNet-18
  forward/backward pass, expected to add at most a few percent to
  per-round wall-clock time.
- **Total wall-clock time per run: expected to be comparable to
  `EXP_007`–`EXP_009`** (observed range 4,008.92 s – 5,094.58 s for 50
  rounds on the project's RTX 5060 Laptop GPU), plus the modest overhead
  above. Three FedProx runs are expected to cost roughly the same GPU-hour
  budget as the three FedAvg runs already completed — a known, bounded
  cost, not an open-ended sweep.

## 17. Risks and Implementation Traps

- **Snapshotting the wrong thing.** Including non-trainable buffers
  (BatchNorm running stats, `num_batches_tracked`) in the proximal
  penalty, or snapshotting `w^t` after local training has already begun
  rather than at the exact start of the round, would silently implement
  an incorrect variant of FedProx. Must be caught by tests (§ 18), not
  assumed correct by inspection.
- **Stale snapshots.** Reusing a snapshot from a previous round (e.g., a
  bug that snapshots once at client construction instead of every
  `load_global_weights()` call) would silently break the method — the
  proximal term would anchor to the wrong reference point every round
  after the first.
- **Mixed-precision interaction.** This project always runs with
  `mixed_precision: true`. The proximal term's addition to `loss` must be
  computed consistently with (or deliberately outside) the existing
  `torch.amp.autocast` context in `train_one_epoch` — an unexamined
  interaction here could cause numerical instability or silently
  incorrect scaling under `GradScaler`. Requires explicit verification,
  not assumption.
- **Gradient-clipping interaction.** `gradient_clip_norm` (already
  supported, currently unused by `EXP_007`–`EXP_009`'s configs) clips the
  **total** gradient, which will now include the proximal term's
  contribution. This is expected/correct behavior but must be confirmed
  by a test rather than left implicit.
- **Do not also change the client-state reset policy.** FedProx does not
  require, and this plan does not propose, switching away from the
  existing reset-optimizer-every-round policy
  (`FederatedClient.reset_training_state()`, unconditional). Changing
  both the local objective and the state-persistence policy in the same
  experiment would confound two variables and break paired
  comparability with `EXP_007`–`EXP_009`.
- **Do not let this leak into centralized training.** `train_one_epoch`
  is shared with `src/train_centralized.py`. The new parameters must
  default to values that make the change a true no-op there — verified
  by a regression test (§ 18), not by inspection alone.

## 18. Required Tests Before Any GPU Run

All fast, CPU-only, tiny-synthetic-data tests, per this project's
existing testing rules — no GPU required to pass:

1. **Backward-compatibility regression test:** `train_one_epoch` with
   `proximal_mu=0` (or unset) produces byte-identical loss values and
   parameter updates to the current, unmodified behavior, on a fixed
   seed and tiny synthetic dataset.
2. **Zero-penalty-at-snapshot test:** with `proximal_mu > 0` and
   `global_params` set equal to the model's own current parameters (zero
   distance), the added proximal loss term is exactly zero.
3. **Closed-form correctness test:** with `proximal_mu > 0` and
   `global_params` set to a known, different tensor set, the added loss
   term and its gradient contribution match the closed-form
   `(mu/2)*||w-w^t||^2` / `mu*(w-w^t)` values on a tiny synthetic model.
4. **Snapshot-independence test:** `FederatedClient`'s stored global
   reference parameters do not change when `self.model`'s parameters are
   subsequently updated by local training — a true detached copy, not a
   live reference (mirroring the existing `test_output_tensors_are_cpu_cloned_detached`-style
   pattern already used in `tests/test_aggregation.py`).
5. **Snapshot-exclusion test:** the stored global reference parameters
   exclude non-trainable buffers (verified against a tiny model with a
   real `BatchNorm` layer, matching `_TinyModelWithBatchNorm` already
   used in `tests/test_aggregation.py`).
6. **Snapshot-freshness test:** across two consecutive rounds, the
   client's stored snapshot changes to match the newly-loaded global
   weights each time `load_global_weights()` is called — no staleness.
7. **Centralized-training non-interference test:** `src/train_centralized.py`'s
   existing test suite continues to pass unmodified, confirming the new
   `train_one_epoch` parameters have no observable effect when unused.
8. **Tiny end-to-end federated round with FedProx:** a full tiny
   synthetic federated round (mirroring
   `test_one_tiny_end_to_end_iid_federated_experiment`) with
   `fedprox_mu > 0` completes successfully and produces finite,
   different-from-plain-FedAvg results on the same tiny setup.

## 19. Decision Tree After the Three FedProx Runs

- **If Success Criteria (§ 12) are met:** treat `FedProx(mu=0.01)` as an
  improved reference for the α=0.1 condition. Proceed toward Phase 4
  (attack simulation) using it, rather than plain FedAvg, as the
  heterogeneity-mitigated baseline for that phase. A small confirmatory
  `mu` ablation may be considered later (e.g. for the eventual paper's
  robustness section) but is not required to proceed.
- **If Failure Criteria (§ 13) are met:** do not adopt FedProx for this
  configuration. This becomes one input toward
  `research_plans/PHASE4_RESEARCH_STRATEGY.md` § 8's SCAFFOLD gate (§ 20
  below) — but not sufficient on its own.
- **If the result is ambiguous** (e.g., mixed-sign paired differences
  with no clear majority, or an improvement close to but not clearly
  above the 2.85pp noise floor): do not conclude either direction.
  Consider whether the ambiguity itself suggests `mu=0.01` is poorly
  tuned (motivating one or two additional, specifically-chosen `mu`
  values — still not a broad sweep) before drawing any conclusion about
  FedProx's suitability.

## 20. Clear Condition for Moving to SCAFFOLD

Per `research_plans/PHASE4_RESEARCH_STRATEGY.md` § 8, **both** of the
following must hold before SCAFFOLD design work begins — `EXP_010`
alone, even if it fails, is not sufficient:

1. FedProx (this plan) is tried and found insufficient (§ 13's failure
   criteria met, or an ambiguous result that a follow-up `mu` adjustment
   also fails to resolve).
2. A separate, not-yet-run `local_epochs > 1` stress test under Non-IID
   data shows sharply worse degradation than this project's
   `local_epochs=1` results — the specific compounding-drift failure
   mode SCAFFOLD targets, untested anywhere in this project to date.

## 21. Clear Condition for Beginning FedTrace-FM Design

Per `research_plans/PHASE4_RESEARCH_STRATEGY.md` § 9, FedTrace design
begins only once **both** of the following exist — completing `EXP_010`
(regardless of outcome) does **not**, by itself, satisfy this gate:

1. A working Phase 4 attack simulation (label flipping, backdoor, or
   gradient/model poisoning) that measurably degrades the global model,
   evaluated against whichever heterogeneity baseline (plain FedAvg or,
   if § 19's success branch is reached, FedProx) is current at that time.
2. A naive baseline attribution/anomaly-detection method evaluated
   against that attack and shown to have a specific, documented
   weakness — most importantly, an inability to reliably distinguish
   malicious clients from honestly-heterogeneous ones, a risk this
   project's own α=0.1 results (this plan's motivation) make concrete
   rather than hypothetical.

`EXP_010` is a heterogeneity-mitigation baseline step that precedes and
is independent of this gate.

## 22. No Statistical-Significance Claim From n=3

Consistent with `analysis/EXP_009_Three_Seed_Baseline_Analysis.md` § 9,
this plan requires that **no analysis of `EXP_010`'s results claim
statistical significance**. Three paired differences support a
descriptive mean and range (§ 15) and a qualitative majority-sign check
(§ 12/13) — nothing more. No standard error, confidence interval,
p-value, or "statistically significant improvement" language is
permitted in any resulting analysis document, regardless of how clean
the three-seed pattern looks.

## 23. PI Approval Checklist

Before any part of `EXP_010` is implemented or executed, confirm:

- [ ] This plan has been reviewed and FedProx (rather than SCAFFOLD or a
      fresh Dirichlet-alpha experiment) is agreed to be the correct next
      controlled method, per `research_plans/PHASE4_RESEARCH_STRATEGY.md`
      § 7/11.
- [ ] `mu = 0.01` (§ 9–10) is agreed as the starting proximal-term
      coefficient, understood explicitly as a literature-grounded
      starting point, not a tuned-optimal value.
- [ ] The design in §§ 5–8 — proximal term added inside
      `train_one_epoch`, snapshot taken in `load_global_weights()`,
      `AggregationStrategy`/`FedAvg` left unmodified, `aggregation: fedavg`
      unchanged in configs — is reviewed and agreed before any code is
      written.
- [ ] All required tests (§ 18) are implemented and passing on CPU before
      any GPU run is attempted.
- [ ] The paired-seed design (§ 3–4: seeds 42/123/2026, everything else
      frozen from `EXP_007`–`EXP_009`) is agreed, and no additional seeds
      or `alpha` values are introduced without a separate plan.
- [ ] Success/failure criteria (§ 12–13) and the noise-floor comparison
      (2.85pp, from `analysis/EXP_009_Three_Seed_Baseline_Analysis.md`)
      are agreed upon in advance, before results are seen.
- [ ] The Decision Tree (§ 19) and the SCAFFOLD/FedTrace gating
      conditions (§ 20–21) are agreed upon in advance.
- [ ] No broad `mu` sweep will be run without a separate, explicit
      justification (§ 11).
- [ ] No claim of statistical significance will appear in any resulting
      analysis document (§ 22).
- [ ] No other file (code, config, test, prior experiment result, or
      Research Database entry) will be modified as part of running
      `EXP_010` until its own results are ready for documentation.
