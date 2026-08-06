# Phase 4 Research Strategy Memo

**To:** PI
**Date:** 2026-08-05
**Prepared from:** `FEDTRACE_FM_MASTER.md`, `PROJECT_ROADMAP.md`,
`PROJECT_CONTEXT.md`, `RESEARCH_DIARY.md`,
`analysis/EXP_004_Convergence_Analysis.md`,
`analysis/EXP_005_Cosine_Scheduler_Analysis.md`,
`analysis/EXP_006_Dirichlet_A05_Analysis.md`.

**Purpose:** With `EXP_001`–`EXP_006` complete and Phase 3
(`PROJECT_ROADMAP.md`) now satisfied — IID and Non-IID federated
CIFAR-100 baselines both established — this memo synthesizes what has
been learned, what remains open, and recommends how to sequence the move
into Phase 4 (attack simulation) and beyond, toward one flagship paper.
No new experiment is proposed or run here beyond what the source
documents already define (`EXP_007`); this is a planning and synthesis
document only.

---

## 1. What Have We Learned Scientifically From EXP_001–EXP_006?

| Experiment | Configuration | Result | What it established |
|---|---|---|---|
| `EXP_001` | Centralized ResNet-18, 10 epochs | 62.71% test top-1, 88.54% top-5 | The project's leakage-free reference figure (10-epoch data-exposure budget). |
| `EXP_002` | FedAvg IID, 10 rounds | 37.81% top-1 (−24.90pp vs. `EXP_001`) | At a matched 10-unit budget, federated substantially trails centralized. |
| `EXP_003` | FedAvg IID, 20 rounds | 54.69% top-1 (−8.02pp) | Doubling the round budget alone closed roughly two-thirds of the gap — the `EXP_002` shortfall was largely undertraining, not an inherent FedAvg limitation. |
| `EXP_004` | FedAvg IID, 50 rounds | 68.52% top-1 (**+5.81pp**, exceeds `EXP_001`) | First federated result to exceed centralized. Round-block analysis classifies convergence as **Outcome B — "slows significantly"** (marginal top-1 gain fell ~92%, block to block, but never plateaued or reversed through round 49). The centralized-exceeding result is attributed to an *unequal effective training budget* (250 client-epoch-equivalents vs. 10 centralized epochs), explicitly **not** evidence that federated learning is inherently superior (`analysis/EXP_004_Convergence_Analysis.md` § 8). |
| `EXP_005` | FedAvg IID, 50 rounds, round-level cosine LR (`min_learning_rate=0.0001`) | 65.12% top-1 (**−3.40pp vs. `EXP_004`**) | A negative, mechanistically-explained result: the schedule reaches near-zero learning rate well before the round budget is exhausted, and with `local_epochs=1` plus the every-round momentum reset, that produces a late-training near-stall (rounds 40–49 marginal gain ≈8–12% of `EXP_004`'s rate). `EXP_004` remains the frozen IID reference. |
| `EXP_006` | FedAvg Non-IID (Dirichlet α=0.5), 50 rounds | 66.22% top-1 (**−2.30pp vs. `EXP_004`**) | First Non-IID data point. Degradation classified as **mild** — broad, even reduction across all macro metrics, convergence tracks IID closely through round 39, and the only clear slowdown is confined to the final decade block (≈60–78% of IID's marginal efficiency there). Still exceeds centralized `EXP_001` by +3.51pp. |

**Taken together:** this project has built and validated, end to end, a
leakage-free, reproducible federated CIFAR-100 pipeline; established that
communication-round budget is the dominant lever closing the
federated/centralized gap under IID; discovered and documented a real
engineering failure mode (a naive round-scoped LR-decay policy can
actively hurt results under this architecture's every-round reset
policy); produced the project's first Non-IID baseline; and, along the
way, fixed a genuine correctness bug (`FedAvg`'s handling of BatchNorm
`num_batches_tracked` under unequal client batch counts) that was
silently blocking Non-IID experiments from running at all. This last item
is easy to overlook but matters for publication: a bug in aggregation
correctness discovered and fixed *before* any attack/attribution result
depends on it is exactly the kind of infrastructure hardening that
protects the credibility of everything built on top.

---

## 2. Which Hypotheses Have Been Confirmed?

- **The IID federated/centralized gap is primarily a round-budget
  artifact, not an inherent FedAvg limitation** (raised after `EXP_002`,
  confirmed twice: `EXP_002`→`EXP_003` closed two-thirds of the gap;
  `EXP_003`→`EXP_004` reversed it entirely).
- **`EXP_004`'s convergence had not plateaued at 50 rounds** (Outcome B,
  not Outcome C) — round-budget extension was a real, if diminishing,
  lever through round 49.
- **Non-IID (Dirichlet α=0.5) partitioning degrades FedAvg accuracy
  relative to IID**, the primary hypothesis of
  `research_plans/EXP_006_NonIID_Dirichlet_Study_Plan.md` — confirmed,
  though the *magnitude* (mild, not moderate/severe) was an open question
  the plan deliberately did not presuppose.

## 3. Which Hypotheses Have Been Rejected?

- **`EXP_005`'s primary hypothesis** — that a round-level cosine schedule
  anchored to decay from `learning_rate` to `min_learning_rate=0.0001`
  over the full 50-round budget would raise final accuracy above
  `EXP_004`'s constant-LR baseline — was **falsified**. It underperformed
  by 3.40pp test top-1, with a mechanistic, data-supported explanation
  (`analysis/EXP_005_Cosine_Scheduler_Analysis.md` § 6).
- **The (never formally adopted, but explicitly guarded-against) reading
  that `EXP_004` exceeding `EXP_001` shows federated learning is
  inherently superior to centralized training** was explicitly rejected
  by its own analysis in favor of the unequal-budget explanation
  (`analysis/EXP_004_Convergence_Analysis.md` § 8). This is not a
  hypothesis anyone on the project formally held, but it is a claim the
  evidence does **not** support and that a paper draft must not imply.

## 4. What Are the Remaining Unknowns?

- **Whether Non-IID degradation scales predictably with heterogeneity.**
  One data point (α=0.5, mild) exists; `EXP_007` (α=0.1) is the next,
  still only a second point.
- **`EXP_006`'s actual per-client class-distribution skew is unmeasured.**
  `client_partition_summary.csv` records only per-client sample totals
  (mild, ≈5.2% coefficient of variation); the per-class breakdown that
  would directly evidence label skew was flagged as a needed instrument
  in `research_plans/EXP_006_NonIID_Dirichlet_Study_Plan.md` but not yet
  built.
- **Seed variance is uncharacterized project-wide.** Every comparison in
  `EXP_002`–`EXP_006` rests on a single seed (42); no repeated-seed trial
  exists anywhere in this project, so no reported gap has been checked
  against ordinary run-to-run noise.
- **Whether a corrected scheduler could still help.** `EXP_005`'s failure
  is specific to one parameterization (`min_learning_rate=0.0001`, no
  warmup) under the every-round momentum-reset policy — untested
  alternatives (higher `min_learning_rate`, warmup, or a persistent
  client-state policy) remain open, but are explicitly deprioritized (see
  § 11).
- **Why `EXP_004` exceeds centralized `EXP_001`**, beyond the
  budget-mismatch explanation — a secondary "federated averaging as
  implicit regularization" hypothesis was raised but never tested
  (`analysis/EXP_004_Convergence_Analysis.md` § 8).
- **The project's actual central research hypothesis is completely
  untested.** Zero attack simulation, attribution, or repair code exists
  (`src/attacks/`, `src/attribution/`, `src/repair/` are empty
  placeholders per `FEDTRACE_FM_MASTER.md`). Nothing in `EXP_001`–`EXP_006`
  bears on whether malicious contributions leave a detectable signature.
- **Generalization beyond this exact setup** — one dataset (CIFAR-100),
  one architecture (ResNet-18), one aggregation strategy (FedAvg), one
  client count (5) — is untested in every direction.

## 5. What Is the Strongest Research Question Now?

Not further baseline characterization for its own sake — Phase 3's
completion criteria are met, and `EXP_006`/`EXP_007` are already the
closing baseline work. The strongest question is the one that directly
determines whether Phase 5's attribution mechanism can work in a
realistic setting:

> **Under non-IID, potentially adversarial conditions, does a malicious
> client's update produce a statistical signature distinguishable from
> the "background" divergence that ordinary, honest Non-IID heterogeneity
> already produces between clients?**

`EXP_006` established, for the first time in this project, that honest
clients under Dirichlet(0.5) partitioning already diverge measurably from
each other (mild but real accuracy/convergence effects, § 8 of its
analysis). Any attribution method (Phase 5) must be able to separate
*malicious* divergence from this now-quantified *honest heterogeneity*
noise floor — not just detect divergence in the IID case, where any
outlier is presumptively suspicious. This reframes Phase 4/5 as
building and evaluating against a **heterogeneity-aware** threat model
from the outset, rather than bolting non-IID robustness on afterward.

## 6. Why Is `EXP_007` (Dirichlet α=0.1) Justified?

Per `analysis/EXP_006_Dirichlet_A05_Analysis.md` §§ 9–12: `EXP_006`
classified `alpha=0.5`'s degradation as mild, consistent with it being a
moderate skew setting in the literature. A single moderate-heterogeneity
data point cannot characterize whether the accuracy cost of heterogeneity
grows gradually or has a sharper threshold at stronger skew — and a lower
`alpha` also produces a more visibly non-IID split, which matters
directly for § 5's research question: characterizing the "honest
heterogeneity noise floor" at a more severe, more realistic-worst-case
setting before attacks are layered on top of it. `EXP_007` is a single,
pre-specified, controlled-variable change (`dirichlet_alpha: 0.5 → 0.1`,
everything else identical to `EXP_006`) with an explicit falsifiable
hypothesis already on record — it is the correct, minimal next baseline
experiment, not scope creep.

## 7. What Evidence Would Justify Implementing FedProx?

FedProx adds a proximal term to each client's local objective specifically
to limit client drift under data heterogeneity. Implementing it is
justified by evidence that plain FedAvg's drift sensitivity is a real
bottleneck at realistic heterogeneity levels — specifically:

- **`EXP_007` (α=0.1) shows a substantially larger degradation than
  `EXP_006`'s mild 2.30pp gap** — e.g., a moderate-to-severe accuracy
  drop, non-monotonic instability, or a convergence slowdown clearly
  beyond what `EXP_006`'s round-block pattern showed.
- **Direct evidence of client drift**, e.g. per-client local training
  loss/accuracy diverging sharply from each other before aggregation
  (this project does not currently instrument per-client train metrics
  beyond the sample-weighted aggregate — building that instrumentation
  would itself be a prerequisite).

Mild degradation alone (`EXP_006`'s actual result) does **not** meet this
bar — implementing a new aggregation strategy to fix a problem that has
not yet been shown to be severe would be solving a problem the data
hasn't demonstrated exists.

## 8. What Evidence Would Justify Implementing SCAFFOLD?

SCAFFOLD uses control variates to correct client drift more aggressively
than FedProx's simple proximal regularization, and its benefit is known
to scale with the number of local update steps taken between
synchronizations. Justifying evidence:

- **FedProx is tried (per § 7's evidence bar) and found insufficient** —
  still a large gap or unstable convergence at the heterogeneity level
  being tested.
- **A `local_epochs > 1` stress test under Non-IID data shows sharply
  worse degradation than `local_epochs=1`.** This project has used
  `local_epochs=1` in every federated experiment to date (`EXP_002`–
  `EXP_007`), so the "many local steps between syncs compounds drift"
  failure mode SCAFFOLD specifically targets has not even been
  stress-tested yet.

SCAFFOLD is a heavier, more complex intervention than FedProx; it should
only be reached for once a simpler correction has been tried and shown
inadequate, not adopted as a first response to `EXP_006`/`EXP_007`.

## 9. What Evidence Would Justify Creating a New FedTrace-FM Algorithm?

This is the project's actual novel contribution (`FEDTRACE_FM_MASTER.md` →
*Novel Contributions*), and the bar is different in kind from §§ 7–8: it
is not about whether FedAvg converges well enough, but about whether
*attribution* (identifying which client) is a problem existing methods
don't solve. Justifying evidence, once Phase 4 attack code exists:

- **A working attack simulation shows malicious contributions measurably
  degrade the global model** (label flipping, backdoor, or
  gradient/model poisoning, at some tested rate) — a precondition, not
  sufficient on its own.
- **A naive baseline attribution/anomaly-detection method (e.g. simple
  update-norm outlier detection, already earmarked in
  `PROJECT_ROADMAP.md` Phase 5) is evaluated against that attack and
  found to have a specific, documented weakness** — most importantly,
  either (a) it cannot reliably distinguish malicious clients from
  honestly-heterogeneous ones (a real risk `EXP_006` already establishes
  the *possibility* of, since honest clients under Non-IID partitioning
  already look measurably different from each other), or (b) it detects
  *that* something is anomalous but cannot correctly attribute it to a
  *specific* client, or fails at low attack rates / subtle attacks.
- **This gap — robustness (down-weighting outliers) is not the same claim
  as attribution (identifying the responsible client) — is itself the
  precise justification.** If no existing baseline correctly attributes
  malicious clients specifically under realistic Non-IID conditions, that
  is direct evidence a purpose-built method is needed, and defines
  exactly what FedTrace must do better than the naive baseline.

## 10. What Risks Could Derail Publication?

- **Single-seed results throughout.** Every accuracy comparison so far
  (`EXP_002`–`EXP_006`) rests on one seed; reviewers will expect variance
  estimates before accepting any gap as a real effect rather than noise —
  already flagged as an open limitation in every analysis document.
- **Misreading the `EXP_004`-exceeds-centralized result.** If the paper
  does not carry forward `analysis/EXP_004_Convergence_Analysis.md` §ᅠ8's
  careful budget-mismatch framing, a reviewer could read it as an
  (incorrect, easily-attacked) claim that federated learning beats
  centralized training in general.
- **Small experimental scale.** 5 clients, one mid-sized CNN, one
  dataset. A flagship claim about attribution robustness will likely need
  a larger client count and/or additional dataset before top-venue
  reviewers find it convincing.
- **Missing external baselines.** No comparison yet to existing
  heterogeneity-mitigation methods (FedProx, SCAFFOLD) or existing
  anomaly-detection/robust-aggregation baselines (e.g. Krum, trimmed
  mean) — a paper claiming a novel attribution mechanism needs to beat
  real prior work, not only an internal naive baseline.
- **Instrumentation gaps.** The per-client class-distribution breakdown
  needed to rigorously characterize Non-IID skew (§ 4) does not exist yet
  and will likely be expected in any Non-IID-focused submission.
- **Compute budget.** Each 50-round CIFAR-100 run takes roughly 65–70
  minutes on a single consumer GPU (`EXP_004`: 4,055.95 s; `EXP_006`:
  4,241.69 s). A full ablation grid (multiple α values × multiple seeds ×
  multiple attack types × multiple rates × multiple defenses) could
  require a very large number of GPU-hours on this project's current
  single-workstation hardware — a real scheduling risk against any
  publication deadline.
- **Scope overreach.** The project's long-term vision spans foundation
  models (Phase 6) and medical imaging (Phase 7) in addition to
  attribution/repair (Phases 4–5). Trying to fit all of this into one
  flagship paper risks an unfocused submission (see § 12's recommended
  scope boundary).

## 11. What Should We Deliberately Avoid?

- **Do not claim "federated exceeds centralized" as a general finding.**
  It is a budget-mismatch artifact in this project's own data
  (§ 2/`EXP_004` analysis § 8).
- **Do not generalize any single-seed result.** Every analysis document
  produced so far already states this explicitly for its own experiment;
  the paper must preserve that discipline rather than smooth it over.
- **Do not chase further IID/scheduler tuning.** `EXP_004` is frozen as
  the reference protocol; `analysis/EXP_005_Cosine_Scheduler_Analysis.md`
  § 9 already recommended deferring further scheduler ablations, and
  nothing since has changed that recommendation.
- **Do not conflate robustness with attribution.** Down-weighting
  anomalous updates (what most robust-aggregation baselines do) is not
  the same contribution as identifying *which* client is responsible
  (what FedTrace must do, § 9). Letting the narrative drift toward "yet
  another robust-FL paper" would undercut the project's actual novel
  claim.
- **Do not implement FedProx, SCAFFOLD, or FedTrace ahead of the evidence
  bars in §§ 7–9.** This project's entire methodology to date has been
  protocol-before-implementation, single-variable-at-a-time; building a
  solution before the data demonstrates the problem it solves would break
  that discipline and weaken the eventual paper's narrative logic.
- **Do not run large ablation sweeps (many α values, many seeds) before
  Phase 4/5 exist.** Characterizing the heterogeneity axis in
  ever-finer detail has diminishing scientific value until there is an
  attack/attribution mechanism to characterize it *for* — compute spent
  here competes directly with compute needed for Phase 4/5.
- **Do not modify frozen baselines or bypass the pre-registration
  workflow.** The project's discipline (frozen reference configs,
  single-variable changes, plans written before configs, configs before
  runs) is itself a reproducibility asset worth preserving intact into
  the paper's methodology section.

## 12. Publication Roadmap — One Flagship Paper

Scoped to keep the flagship paper focused on Phases 3–5
(`PROJECT_ROADMAP.md`); foundation-model (Phase 6) and medical-imaging
(Phase 7) work is recommended as explicitly out of scope for this paper
and reserved for the journal extension already described in
`FEDTRACE_FM_MASTER.md` → *Long-Term Vision* (items 6–7, 9).

1. **Methodology and baseline characterization (near-complete).**
   `EXP_001`–`EXP_006`, plus `EXP_007` (§ 6), close out the paper's
   "federated methodology + IID/Non-IID baseline" section: leakage-free
   split discipline, round-budget convergence behavior, the documented
   scheduler negative result, and a two-point heterogeneity
   characterization (α=0.5, α=0.1).
2. **Attack simulation (Phase 4).** Implement label-flipping, backdoor,
   and gradient/model-poisoning attacks at configurable rates
   (`PROJECT_ROADMAP.md` Phase 4 criteria); measure impact against both
   the IID (`EXP_004`) and Non-IID (`EXP_006`/`EXP_007`) baselines. This
   produces the paper's threat-model and impact-quantification section.
3. **Naive baseline defense.** Implement and evaluate at least one
   existing simple defense (e.g. update-norm outlier detection, already
   named in `PROJECT_ROADMAP.md` Phase 5) against the Phase 4 attacks, to
   establish the § 9 evidence gate — where it succeeds and specifically
   where/why it fails at attribution.
4. **FedTrace attribution (Phase 5) — the paper's central contribution.**
   Design and evaluate the attribution mechanism against the naive
   baseline from step 3, across attack types/rates and both IID and
   Non-IID conditions, quantifying attribution precision/recall.
5. **Repair mechanism.** Use the attribution signal to correct the global
   model, compared against naive alternatives (discard-and-retrain,
   client exclusion) on cost/accuracy trade-off — completing the
   detect→attribute→repair pipeline the paper's abstract will describe.
6. **Targeted robustness ablations, not a full sweep.** Before writing up:
   multi-seed repeats of the headline attribution/repair result (directly
   addressing § 10's variance risk) and, time permitting, one additional
   heterogeneity point or a FedProx/SCAFFOLD comparison *only if* § 7/8's
   evidence bars were met along the way — not a speculative grid search.
7. **Write-up and submission** (`PROJECT_ROADMAP.md` Phase 8), with
   foundation-model/medical-imaging results explicitly deferred to the
   Phase 9 journal extension rather than folded into this paper.
