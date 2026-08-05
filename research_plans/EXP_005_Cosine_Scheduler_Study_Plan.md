# EXP_005 — Cosine Scheduler Study Plan

**Status:** Planning document only. `EXP_005` has **NOT** been executed. No
config file for this experiment has been created yet.

---

## 1. Objective

Following `EXP_004`'s formal convergence classification
(`analysis/EXP_004_Convergence_Analysis.md`, Outcome B — "convergence
slows significantly"), and the pre-registered decision tree in
`research_plans/EXP_004_Convergence_Study_Plan.md` Section 9, test whether
introducing a learning-rate scheduler — rather than a further
communication-round extension — improves final accuracy and/or
late-training stability relative to `EXP_004`'s constant-learning-rate
policy.

## 2. Background from the EXP_004 Convergence Analysis

Per `analysis/EXP_004_Convergence_Analysis.md` (verified from
`results/federated/cifar100_fedavg_iid_50rounds/round_history.csv`,
`summary.json`, `final_test_metrics.json`):

- `EXP_004` (FedAvg IID, 50 rounds, constant `learning_rate=0.01`, no
  scheduler) reached **final test top-1 68.52%, test top-5 91.27%, test
  loss 1.1049** — the current project best, and the first federated result
  to exceed the centralized `EXP_001` baseline (62.71% top-1).
- Round-block analysis showed per-round validation top-1 gain shrinking
  monotonically across every decade block: ≈3.964pp/round (rounds 0–9) →
  ≈1.668pp/round (10–19) → ≈0.688pp/round (20–29) → ≈0.440pp/round (30–39)
  → ≈0.318pp/round (40–49) — a ~92% reduction in marginal efficiency from
  the first block to the last, with no block reversing this decline.
- The final-ten-round window (rounds 40–49) still showed a clear net
  improvement (val top-1 +3.18pp, val loss −0.1022) but with two isolated
  single-round top-1 regressions (round 41→42: −0.70pp; round 43→44:
  −0.30pp) and, for the first time in this project's round histories, a
  single-round top-5 regression at the very end (round 48→49: −0.40pp).
- Applying `EXP_004`'s own pre-registered Section 6/7 criteria, the
  analysis classified this behavior as **Outcome B: convergence slows
  significantly** (not Outcome A "continues", not Outcome C "plateaus").
  Per `research_plans/EXP_004_Convergence_Study_Plan.md` Section 9,
  Outcome B's prescribed next action is: *"introduce exactly one new
  controlled variable — a learning-rate scheduler — as the next
  single-variable experiment, since round-budget increases are shown to
  be losing efficiency and a complementary lever should now be tested in
  isolation."*
- `EXP_005`, as specified below, is that prescribed experiment.

## 3. Controlled Variable

**`scheduler: none → cosine`. This is the only scientific change from
`EXP_004`.** Every other configuration value — model, optimizer, learning
rate, momentum, weight decay, batch size, local epochs, aggregation
strategy, client count, clients per round, partition, seed, validation
fraction, evaluation protocol, checkpoint format, and client-state reset
policy — remains identical to `EXP_004`
(`configs/cifar100_fedavg_iid_50rounds.yaml`).

### 3.1 Critical implementation note (verified from source, not assumed)

The `scheduler` field is not applied at the level of the full 50-round
run. Per `src/federated/client.py` (`FederatedClient._build_local_state()`
/ `reset_training_state()`), each client independently constructs its own
scheduler via `src.training.engine.build_scheduler(optimizer,
config.scheduler, epochs=config.local_epochs)`, and this scheduler —
along with the optimizer and AMP scaler — is **discarded and rebuilt from
scratch at the start of every communication round**
(`FederatedClient.reset_training_state()`, invoked from
`load_global_weights()`; unchanged policy, see `DECISIONS.md`).

With `local_epochs=1` (identical to `EXP_004`, held fixed by this plan),
`build_scheduler(..., epochs=1)` constructs `CosineAnnealingLR(optimizer,
T_max=1)`. Per `src/training/engine.py`, the scheduler is stepped exactly
once, **after** the round's single local epoch of training batches has
already completed at the un-decayed base learning rate — and that
scheduler state is then thrown away at the next round's reset, before any
further training batch is ever run under the decayed rate. Consequently,
**under the current codebase and with `local_epochs=1` held fixed, setting
`scheduler: cosine` is expected to have no effect on any gradient update
relative to `scheduler: none`** — every training batch in every round
still runs at the constant configured `learning_rate=0.01`, because the
one-and-only scheduler step per round never precedes any subsequent
training within that round.

This is a verified property of the existing, unmodified code
(`src/training/engine.py::build_scheduler`, `src/federated/client.py`), not
a new result and not an invented outcome. It is documented here so the
PI can decide, before any GPU time is spent, whether to (a) run `EXP_005`
exactly as literally specified — a valid experiment whose expected outcome
is "no meaningful difference from `EXP_004`," itself a meaningful
negative-control data point — or (b) treat this plan as blocked pending a
separate, explicitly-scoped code change (e.g., round-level rather than
local-epoch-level scheduling) before `EXP_005` is run. **This plan does
not propose or make that code change; it only documents the fact for PI
decision-making.** See Section 11 (Risks) and Section 12 (PI Approval
Checklist).

## 4. Hypothesis

**Primary hypothesis (as literally specified — `scheduler: none → cosine`,
`local_epochs` unchanged at 1):** Given Section 3.1's verified code
behavior, `EXP_005`'s final test top-1/top-5/loss and full round history
are expected to be statistically indistinguishable from `EXP_004`'s
(`round_history.csv` values matching within ordinary run-to-run floating
point/CUDA nondeterminism, not a systematic shift), because the cosine
schedule cannot alter any training batch's effective learning rate under
`local_epochs=1` and the every-round reset policy.

**Falsifiable framing:** If `EXP_005`'s results *do* differ materially
from `EXP_004`'s — either better or worse, on final test top-1 or on the
final-ten-round noise pattern — that would itself be a notable and
currently-unexplained finding requiring its own follow-up investigation
into why the scheduler had an effect despite Section 3.1's analysis,
rather than confirmation of the intended "smooth 50-round anneal" premise.

## 5. Exact Frozen EXP_004 Configuration

Copied verbatim from `configs/cifar100_fedavg_iid_50rounds.yaml` (the
actual committed config that produced `EXP_004`'s frozen results):

| Field | EXP_004 (verified, frozen) |
|---|---|
| `experiment_id` | `EXP_004` |
| `experiment_name` | `cifar100_fedavg_iid_50rounds` |
| `dataset_config` | `configs/cifar100_data.yaml` |
| `num_clients` | 5 |
| `clients_per_round` | 5 |
| `minimum_updates` | 5 |
| `communication_rounds` | 50 |
| `local_epochs` | 1 |
| `batch_size` | 32 |
| `num_workers` | 2 |
| `model` | `resnet18_cifar` |
| `num_classes` | 100 |
| `optimizer` | `sgd` |
| `learning_rate` | 0.01 |
| `momentum` | 0.9 |
| `weight_decay` | 0.0005 |
| `scheduler` | `none` |
| `mixed_precision` | `true` |
| `aggregation` | `fedavg` |
| `partition` | `iid` |
| `validation_fraction` | 0.1 |
| `evaluate_every` | 1 |
| `checkpoint_every` | 1 |
| `seed` | 42 |
| `device` | `auto` |
| `fail_round_on_client_error` | `true` |
| `continue_after_failed_round` | `false` |

**EXP_005's planned configuration is identical to every row above except
`scheduler` (`none → cosine`), plus new, non-scientific bookkeeping values
for `experiment_id`, `experiment_name`, `output_dir`, and
`checkpoint_dir`** — the same bookkeeping-only convention already used for
`EXP_002 → EXP_003 → EXP_004`.

## 6. Success Criteria

`EXP_005` will be considered evidence that the scheduler **materially
helps** if, relative to `EXP_004`'s verified final test metrics (top-1
68.52%, top-5 91.27%, loss 1.1049):

- Final test top-1 accuracy exceeds 68.52% by a margin clearly outside
  ordinary seed/nondeterminism noise (the project's frozen results have
  not yet characterized this noise band via repeated runs at the same
  seed — see Section 11), **and**
- The final-ten-round (rounds 40–49) validation top-1 trajectory shows
  fewer or smaller-magnitude single-round regressions than `EXP_004`'s
  equivalent window (two regressions, −0.70pp and −0.30pp; see
  `analysis/EXP_004_Convergence_Analysis.md` Section 5), **and**
- Validation loss over the final ten rounds shows an equal or more
  consistent (less oscillating) downward trend than `EXP_004`'s.

## 7. Failure Criteria

`EXP_005` will be considered evidence that the scheduler, as implemented
under the current round-scoped/reset-every-round policy, **does not
help** (consistent with the Section 3.1/4 no-op hypothesis) if:

- Final test top-1, top-5, and loss are statistically indistinguishable
  from `EXP_004`'s (68.52% / 91.27% / 1.1049), **or**
- The full round-by-round `validation_top1`/`validation_loss` history is
  numerically identical or near-identical to `EXP_004`'s round history
  (the expected outcome per Section 3.1's code-level analysis), **or**
- Final test top-1 is *lower* than `EXP_004`'s 68.52% — which would
  indicate the scheduler introduced a regression rather than a neutral
  no-op or an improvement.

## 8. Expected Learning-Curve Behavior

Per Section 3.1's verified code analysis, the most probable outcome is
that `EXP_005`'s round-by-round validation loss/top-1/top-5 curve is
**effectively a re-run of `EXP_004`'s curve** (same monotonic-with-noise
shape, same approximate per-block marginal gains as documented in
`analysis/EXP_004_Convergence_Analysis.md` Sections 2–5), because the
cosine schedule cannot change any batch-level learning rate under
`local_epochs=1`. Any visible divergence from `EXP_004`'s curve would
itself be the notable finding (see Section 4's falsifiable framing) and
should not be assumed away — this expectation is a prediction to be
tested, not a foregone conclusion, and the run must still be evaluated
against Sections 6–7 rather than assumed to match Section 8 in advance.

## 9. Decision Tree After EXP_005

- **If `EXP_005` matches Section 7 (statistically indistinguishable from
  `EXP_004`, confirming the Section 3.1 no-op prediction):** do not pursue
  local-epoch-scoped cosine scheduling further under this client
  architecture. The next candidate lever becomes either (a) a
  round-level learning-rate schedule (an explicitly-scoped code change to
  `src/federated/`, not yet designed or approved), or (b) proceeding to
  Non-IID (Dirichlet/shard) partitioning, since the round-budget and
  naive-scheduler levers would both be exhausted at this point.
- **If `EXP_005` matches Section 6 (materially better than `EXP_004`,
  contradicting the Section 3.1 prediction):** treat the mechanism as an
  open question requiring its own dedicated investigation (why did a
  cosine schedule scoped to a single local epoch produce a measurable
  effect?) before relying on the result, and only then consider it as a
  candidate default for future federated experiments.
- **If `EXP_005` is worse than `EXP_004` on final test top-1 (Section 7's
  third bullet):** do not adopt `scheduler: cosine` under this
  configuration; revert to `scheduler: none` as the working baseline and
  treat the regression itself as requiring explanation before any further
  scheduler-related experiment is planned.

Exactly one of these three branches applies to any observed `EXP_005`
outcome.

## 10. Fair-Comparison Rules

- `EXP_005` must change **only** `scheduler` (`none → cosine`) plus the
  non-scientific bookkeeping fields (`experiment_id`, `experiment_name`,
  `output_dir`, `checkpoint_dir`) relative to `EXP_004`'s exact config
  (Section 5). `local_epochs` must remain 1 — changing it alongside the
  scheduler would confound two variables and invalidate the comparison,
  even though Section 3.1 shows `local_epochs=1` is precisely what makes
  the scheduler change likely inert.
- Seed must remain 42, identical to `EXP_001`–`EXP_004`, so any observed
  difference cannot be attributed to a different partition or
  initialization.
- The comparison baseline is `EXP_004`'s already-frozen, verified results
  (`results/federated/cifar100_fedavg_iid_50rounds/`) — `EXP_005` must not
  overwrite these, and must write to its own new `output_dir`/
  `checkpoint_dir`.
- No metric from `EXP_005` may be reported until the run has fully
  completed all 50 configured rounds (`communication_rounds_completed ==
  50`, matching `EXP_004`'s completion standard).
- Analysis of `EXP_005` must reuse the same round-block methodology
  applied to `EXP_004` in `analysis/EXP_004_Convergence_Analysis.md`
  (decade blocks, previous-block-endpoint baseline) so the two analyses
  are directly comparable.

## 11. Risks

- **Wasted computation on a predicted no-op:** Section 3.1's code-level
  analysis predicts `EXP_005` will reproduce `EXP_004`'s results almost
  exactly, meaning the full ≈68-minute run (`EXP_004`'s total experiment
  time was 4055.95 s) may spend significant GPU time confirming a
  prediction that could, in principle, be checked more cheaply (e.g. a
  short-round smoke test, or direct unit inspection of the per-round
  effective learning rate) before committing to a full 50-round run. This
  plan does not resolve that trade-off — it is a PI decision (see Section
  12).
- **Ambiguous "materially different" threshold:** this project has not
  yet run repeated same-seed trials of any experiment to characterize
  ordinary run-to-run noise (CUDA nondeterminism, floating-point
  accumulation order), so Sections 6–7's "statistically indistinguishable"
  language currently relies on qualitative judgment (comparing full round
  histories and final metrics) rather than a formal statistical test.
- **Scope creep temptation:** if `EXP_005` confirms the no-op prediction,
  there will be a natural temptation to immediately design and implement
  round-level scheduling as a follow-up. That is explicitly **out of
  scope** for this plan and for `EXP_005` itself — it would be a new,
  separately-scoped code change and a new pre-registered plan, not a
  continuation of this single-variable experiment.
- **Opportunity cost:** as with `EXP_004`'s plan, time spent here delays
  Non-IID partitioning and the attack/attribution phases already queued
  behind the round-budget/scheduler question.

## 12. PI Approval Checklist

Before `EXP_005` is executed, confirm:

- [ ] This plan has been reviewed and `scheduler: none → cosine` (with
      `local_epochs` held at 1) is agreed to be the correct next step per
      `EXP_004`'s Outcome B decision-tree branch.
- [ ] Section 3.1's code-verified no-op prediction has been read and
      understood: **as literally specified, this experiment is expected
      to reproduce `EXP_004`'s results, not to produce a genuine 50-round
      cosine anneal.** The PI explicitly chooses one of:
      - [ ] Run `EXP_005` anyway, as specified, to empirically confirm
            the no-op prediction (a valid negative-control result), **or**
      - [ ] Pause this plan and separately scope a code change (round-level
            scheduling) before running any cosine-scheduler experiment —
            in which case `EXP_005` as currently specified is not executed.
- [ ] Section 5's configuration table has been used to create
      `configs/cifar100_fedavg_iid_50rounds_cosine.yaml` (or equivalent),
      with every non-`scheduler` field verified identical to `EXP_004`'s
      config.
- [ ] The bookkeeping identity fields (`experiment_id`, `experiment_name`,
      `output_dir`, `checkpoint_dir`) are set to new, non-colliding values
      so `EXP_004`'s frozen results are not overwritten.
- [ ] Estimated runtime (comparable to `EXP_004`'s ≈4055.95 seconds, since
      round count and local epochs are unchanged) is acceptable given
      available compute time.
- [ ] Success/failure criteria (Sections 6–7) are agreed upon in advance,
      before results are seen, to avoid post-hoc reinterpretation.
- [ ] The Decision Tree (Section 9) is agreed upon in advance for all
      three branches.
- [ ] No other file (code, config, test, prior experiment result, or
      Research Database entry) will be modified as part of running
      `EXP_005` until its own results are ready for documentation.
