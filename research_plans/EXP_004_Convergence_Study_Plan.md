# EXP_004 — Convergence Study Plan

**Status:** Planning document only. EXP_004 has NOT been executed. No config
file for this experiment has been created yet.

---

## 1. Objective

Does extending the federated communication-round budget for IID FedAvg on
CIFAR-100 continue to close the gap to the centralized baseline (EXP_001),
or has training begun to plateau by round 20?

EXP_003 (20 rounds) showed no plateau: the final five rounds (15–19) still
produced consistent, non-negligible gains in validation loss, top-1, and
top-5 accuracy, and the round-19 gain was not the smallest in that window.
EXP_004 exists to determine whether this trend continues under a
substantially larger round budget, or whether it has, by that point,
converged.

## 2. Background

`EXP_001` is the frozen centralized baseline: CIFAR-compatible ResNet-18
trained for 10 centralized epochs, reaching test top-1 62.71%, test top-5
88.54%, test loss 1.3016. `EXP_002` applied FedAvg under an IID partition
across 5 clients for 10 communication rounds (1 local epoch per round,
client optimizer state reset every round) and reached test top-1 37.81%,
test top-5 70.27%, test loss 2.3611 — a 24.90-percentage-point top-1 gap
versus the centralized baseline at a matched 10-unit data-exposure budget.

`EXP_003` changed exactly one variable from `EXP_002` — communication
rounds, 10 → 20 — with every other hyperparameter, the partition, the
client set, and the seed held identical. This produced test top-1 54.69%,
test top-5 83.75%, test loss 1.6335: a +16.88pp top-1 improvement over
`EXP_002`, narrowing the gap to `EXP_001` from 24.90pp to 8.02pp. Analysis
of `EXP_003`'s full round history showed marginal per-round validation
top-1 gain shrinking from ≈3.96pp/round (rounds 0–9) to ≈1.67pp/round
(rounds 10–19), with the final five rounds (15–19) still averaging
≈1.17pp/round and no flat or negative transition anywhere in that window —
consistent with emerging but not yet exhausted diminishing returns.

## 3. Controlled Variable

Only `communication_rounds` will change, from 20 (`EXP_003`) to 50.
Every other configuration value — model, optimizer, learning rate,
scheduler (none), batch size, local epochs, aggregation strategy, client
count, clients per round, partition, seed, evaluation protocol, checkpoint
format, and result format — remains identical to `EXP_003`. The
experiment identity fields (`experiment_id`, `experiment_name`,
`output_dir`, `checkpoint_dir`) will differ only as non-scientific
bookkeeping, exactly as was done for `EXP_002` → `EXP_003`, so that
`EXP_003`'s frozen results are never overwritten.

## 4. Hypothesis

Extending training to 50 communication rounds will continue to improve
all four tracked metrics, but at a shrinking marginal rate consistent with
the diminishing-returns trend already observed between `EXP_002` and
`EXP_003`:

- **Validation loss:** continues to decrease monotonically on average,
  with per-round reduction magnitude smaller than the `EXP_003` rounds
  10–19 average (≈0.071/round), most plausibly settling toward a much
  smaller per-round reduction by round 50.
- **Validation top-1 / top-5:** continue to increase, with average
  per-round gain smaller than the `EXP_003` rounds 10–19 average
  (≈1.67pp/round for top-1), most plausibly in the low single digits of
  a percentage point per round by the later rounds of this run.
- **Test accuracy** (evaluated once, at the end): expected to improve
  further over `EXP_003`'s 54.69% top-1 / 83.75% top-5, most plausibly
  narrowing the remaining 8.02pp gap to `EXP_001`, though not necessarily
  eliminating it — whether the gap fully closes, partially closes, or
  stalls above zero is exactly the open question this experiment is
  designed to answer, not a foregone conclusion.

## 5. Experimental Configuration

Copied from the verified `EXP_003` configuration
(`configs/cifar100_fedavg_iid_20rounds.yaml`), with the single scientific
change noted below. **This plan does not create the config file — it
specifies what that config must contain when it is created.**

| Field | EXP_003 (verified) | EXP_004 (planned) |
|---|---|---|
| Dataset | CIFAR-100 | CIFAR-100 (unchanged) |
| Model | resnet18_cifar (11,220,132 params) | unchanged |
| Partition | IID, 5 clients, seed 42, 9,000 samples/client | unchanged |
| Aggregation strategy | FedAvg | unchanged |
| Clients per round / minimum updates | 5 / 5 | unchanged |
| Local epochs | 1 | unchanged |
| Optimizer | SGD | unchanged |
| Learning rate | 0.01 | unchanged |
| Momentum | 0.9 | unchanged |
| Weight decay | 0.0005 | unchanged |
| Scheduler | none | unchanged |
| Batch size | 32 | unchanged |
| Mixed precision | enabled | unchanged |
| Validation fraction | 0.1 | unchanged |
| Client-state policy | reset optimizer/scheduler/scaler every round | unchanged |
| Seed | 42 | unchanged |
| **Communication rounds** | **20** | **50 (the only scientific change)** |
| experiment_id / output_dir / checkpoint_dir | `cifar100_fedavg_iid_20rounds` | new identity only (bookkeeping, non-scientific) |

## 6. Success Criteria (evidence of continued convergence)

EXP_004 will be considered evidence of **continued, meaningful
convergence** if, at round 50:

- Validation top-1 accuracy at round 49 exceeds round 19's 54.22% by a
  clearly non-trivial margin (materially more than noise — e.g., several
  percentage points), **and**
- The final-ten-round average marginal gain (rounds 40–49) is positive
  and not indistinguishable from zero, **and**
- Validation loss at round 49 is lower than round 19's 1.6727, with no
  sustained (multi-round) increase anywhere in the final ten rounds, **and**
- Final test top-1 accuracy narrows the 8.02pp gap to `EXP_001` (62.71%)
  relative to `EXP_003`, even if it does not close it entirely.

## 7. Failure Criteria (evidence of effective plateau)

EXP_004 will be considered evidence that convergence has **effectively
plateaued** if, at round 50:

- The final-ten-round average marginal gain in validation top-1 (rounds
  40–49) is near zero or within the round-to-round noise band already
  observed in `EXP_003` (i.e., comparable to or smaller than the smallest
  single-round deltas seen in `EXP_003`'s rounds 15–19, such as the
  +0.70pp round 17→18 transition), **or**
- Validation loss shows no sustained decrease over the final ten rounds
  (flat or oscillating without a clear downward trend), **or**
- Validation top-1/top-5 decrease or oscillate without net improvement
  over multiple consecutive rounds late in training (a signal distinct
  from the isolated single-round noise already observed in `EXP_003`).

## 8. Expected Scientific Outcomes

**A. Convergence continues** (gains through round 50 remain comparable in
character to `EXP_003`'s rounds 10–19, i.e., shrinking but still
substantial): supports the conclusion that the round budget, not the
architecture or FedAvg itself, remains the dominant factor separating
federated IID training from the centralized baseline. This would justify
further round-budget extension as still the single highest-value lever.

**B. Convergence slows significantly** (gains are much smaller than in
`EXP_003` but still clearly positive and non-noise): supports the
conclusion that federated IID FedAvg is approaching, but has not yet
reached, a performance ceiling under this configuration — additional
rounds still help, but with rapidly diminishing efficiency, making other
levers (learning-rate schedule, local epochs) increasingly worth
investigating in parallel with, or instead of, further round increases.

**C. Convergence plateaus** (gains in the final rounds are statistically
indistinguishable from noise, per Section 7): supports the conclusion
that the round budget has been exhausted as the primary lever, and that
any remaining gap to `EXP_001` reflects a different, unaddressed cause
(e.g., lack of a learning-rate schedule, the every-round optimizer-state
reset policy, or a structural cost of federated averaging under this
partition) rather than insufficient training time.

## 9. Decision Tree

- **If Outcome A (convergence continues):** run a further controlled
  round-budget extension (e.g., `EXP_005` at a larger round count),
  continuing to hold every other variable fixed, since rounds remain the
  demonstrated dominant lever.
- **If Outcome B (convergence slows significantly):** introduce exactly
  one new controlled variable — a learning-rate scheduler — as the next
  single-variable experiment, since round-budget increases are shown to
  be losing efficiency and a complementary lever should now be tested in
  isolation.
- **If Outcome C (convergence plateaus):** do not extend rounds further;
  instead run a single-variable experiment introducing a learning-rate
  scheduler (all else identical to `EXP_004`), since the data would show
  that rounds are no longer the primary unresolved variable.

Exactly one action follows from each outcome; no scenario recommends more
than one next experiment.

## 10. Reproducibility Notes

Changing exactly one variable (`communication_rounds`) and holding every
other configuration value, the partition, the client set, and the seed
identical to `EXP_003` is what allows any observed difference in outcome
to be attributed unambiguously to the round-budget change. This is the
same discipline applied in `EXP_002` → `EXP_003`, where it allowed rounds
0–9 of `EXP_003` to exactly reproduce `EXP_002`'s per-round values as an
internal consistency check. Without this discipline, an improvement (or
lack of one) at round 50 could not be distinguished from the effect of
any other change made at the same time, and the resulting comparison
would not be scientifically valid.

## 11. Risks

- **Wasted computation:** if convergence has already plateaued well
  before round 50 (Outcome C), the additional ~30 rounds beyond `EXP_003`
  spend significant compute time (`EXP_003`'s 20 rounds took ≈1776.63
  seconds; a 50-round run would be expected to take roughly 2.5x that,
  absent any change in per-round cost) without producing new scientific
  information beyond what a smaller extension might have shown.
- **Overfitting:** longer federated training with a fixed learning rate
  and no scheduler could allow local client models to increasingly overfit
  their local shards before each reset, potentially causing validation
  loss to plateau or rise even while local training loss keeps falling —
  this would need to be checked explicitly against the local training
  loss curve, not assumed.
- **Diminishing returns without a clear stopping signal:** because
  `EXP_003` already showed noisy (non-monotonic) round-to-round deltas, a
  small number of additional rounds could show an apparent slowdown or
  apparent continuation purely from noise, making the Section 6/7 criteria
  important to apply over multi-round windows (not single-round deltas)
  before drawing a conclusion.
- **Opportunity cost:** time spent on a third round-budget experiment
  delays investigation of Non-IID partitioning and the attack/attribution
  phases, both of which are already queued behind this round-budget
  question.

## 12. PI Approval Checklist

Before `EXP_004` is executed, confirm:

- [ ] This plan has been reviewed and the single controlled variable
      (`communication_rounds: 20 → 50`) is agreed to be the correct next
      step (as opposed to a scheduler change or Non-IID partitioning).
- [ ] Section 5's configuration table has been used to create
      `configs/cifar100_fedavg_iid_50rounds.yaml` (or equivalent), with
      every non-round field verified identical to `EXP_003`'s config.
- [ ] The bookkeeping identity fields (`experiment_id`, `experiment_name`,
      `output_dir`, `checkpoint_dir`) are set to new, non-colliding values
      so `EXP_003`'s frozen results are not overwritten.
- [ ] Estimated runtime (~2.5x `EXP_003`'s ≈1776.63 seconds) is acceptable
      given available compute time.
- [ ] Success/failure criteria (Sections 6–7) are agreed upon in advance,
      before results are seen, to avoid post-hoc reinterpretation.
- [ ] The Decision Tree (Section 9) is agreed upon in advance for all
      three outcomes.
- [ ] No other file (code, config, test, prior experiment result, or
      Research Database entry) will be modified as part of running
      `EXP_004` until its own results are ready for documentation.
