# FedTrace Stage B: Independent Validation Plan

**Status:** Planning document only. No code, test, config, README, or Research Database file was created or modified in producing it. No experiment was run. This document defines the Stage B validation protocol; it does not execute it, and it does not create `EXP_016`'s (or any other) config.

**Purpose:** Independently validate the now-frozen FedTrace v3 detector (`research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md`, confirmed passing its retrospective development gate in `analysis/FEDTRACE_V3_RETROSPECTIVE_ANALYSIS.md`, and its offline unit test suite at 63/63) on federated runs and seeds that were **not** used to design, diagnose, or tune it. This document exists specifically to keep that boundary strict: development evidence and validation evidence must never be allowed to blur together, in either direction.

---

## 1. Freeze Declaration

The following are frozen, effective immediately, and **no Stage B result may be used to modify any of them**:

**Temporal residual (`z_t_res`):**
```
t(k,r) = self_prev_round_cosine
c(r) = median_j t(j,r)                          (no leave-one-out)
d(k,r) = c(r) - t(k,r)
D_k = { d(k,r) : r in [1, T_ref) },  T_ref = 20
historical_center_k = mean(D_k)
historical_scale_k  = sample stdev(D_k)          (n-1 denominator)
z_t_res(k,r) = max(0, [d(k,r) - historical_center_k] / historical_scale_k)
```
Undefined (never fabricated) if `|D_k| < 2` or `historical_scale_k == 0`.

**Magnitude (`z_m`):** unchanged from v1/v2 — per-client longitudinal calibration, `mean_pre_m(k)`/`std_pre_m(k)` over `round_id < T_ref`, sample stdev.

**Primary score:**
```
FedTraceV3Score(k,r) = max(z_m(k,r), z_t_res(k,r))
```
over whichever component is defined; undefined if both are undefined. **The peer signal (`z_p`) remains outside this primary score** — computed and logged for ablation/monitoring only, exactly as in v1/v2.

**Detection rule:** raw-anomalous iff `FedTraceV3Score(k,r) > 3`. Persistence `P = 3` consecutive anomalous rounds. Confirmation round = onset + 2. Onset estimate = first round of the confirmed run (not the confirmation round).

**Attribution:** predicted malicious client = earliest-confirmed client; ties broken by higher mean `FedTraceV3Score` during the persistent run, then `client_id` ascending. Missing-value handling (never substitute zero, never substitute an epsilon or MAD fallback, an undefined score breaks a persistence streak) is frozen exactly as implemented in `src/analysis/fedtrace_detector.py`'s v3 section and specified in `research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md` §16.

**No exceptions.** If a Stage B run produces a disappointing or ambiguous result, §13 below governs what happens — and tuning any value in this section is explicitly not among the permitted responses.

## 2. Development Data Boundary

`EXP_014` (`gaussian_update_noise`, seed 42) and `EXP_015` (`norm_preserving_directional`, seed 42) are **development data**. They were used, directly and repeatedly, to diagnose v1's and v2's flaws and to design and hand-verify v3's fix (`analysis/FEDTRACE_V2_RETROSPECTIVE_ANALYSIS.md`, `analysis/FEDTRACE_V3_RETROSPECTIVE_ANALYSIS.md`). They may continue to be shown in a method-development section of any future write-up (exactly as they are cited throughout this document), but **they do not count as independent validation evidence for FedTrace v3, and any future document that reports Stage B results must not merge development-run numbers with Stage B numbers into a single combined evidence count.**

## 3. Independent Validation Seeds

Seeds **123** and **2026** — already-established, unseen (with respect to any attacked run) strong-Non-IID seeds, with existing honest FedAvg baselines already on disk:

- `EXP_008` — FedAvg, Dirichlet α=0.1, CIFAR-100/ResNet-18, 50 rounds, **seed 123** (`configs/cifar100_fedavg_dirichlet_a01_seed123_50rounds.yaml`, verified in `analysis/EXP_008_Replication_Analysis.md`).
- `EXP_009` — same configuration, **seed 2026** (`configs/cifar100_fedavg_dirichlet_a01_seed2026_50rounds.yaml`, verified in `analysis/EXP_009_Three_Seed_Baseline_Analysis.md`).

**Why reusing these exact seeds for the attacked paired runs is a strong design, not a shortcut.** `EXP_008`/`EXP_009` already establish, independently of anything to do with FedTrace, exactly what each seed's honest partition realization, honest training trajectory, and honest client heterogeneity look like — this project's own prior work (`analysis/EXP_009_Three_Seed_Baseline_Analysis.md`) already characterizes how much natural, non-malicious variation exists client-to-client and seed-to-seed under this exact configuration. Attacking these same seeds means every Stage B attacked run has a **pre-existing, already-analyzed, attack-independent honest counterfactual** to pair against — precisely the same paired-single-variable-change design this project has used since `EXP_007`→`EXP_008`. Choosing genuinely fresh, never-before-seen seeds instead would save nothing scientifically (the seed itself is not what needs to be "unseen" for a fair validation — only the *attacked* run needs to be unseen to FedTrace's design process) while discarding the ability to cleanly separate "this seed's honest heterogeneity" from "this attack's effect," which is exactly the confound this entire project's earlier work (magnitude-vs-heterogeneity, peer-cosine-vs-secular-trend) has repeatedly had to disentangle by hand. Reusing `EXP_008`/`EXP_009` is the paired design already validated across this project's whole history, applied once more.

## 4. Attack Families to Validate

**Both already-developed attack families, exactly as frozen — no third family introduced.**

**A. Gaussian update-noise attack** (`gaussian_update_noise`): `malicious_client_id = client_002`, `attack_start_round = 20`, `attack_strength_rho = 1.0` — identical to `EXP_014`.

**B. Norm-preserving directional attack** (`norm_preserving_directional`): `malicious_client_id = client_002`, `attack_start_round = 20`, `directional_target_cosine = 0.0` — identical to `EXP_015`.

**Why the attacker identity, onset round, and severity must not change now.** Changing any of these would introduce a second confound alongside the seed change, destroying the paired design: if `EXP_016` (say) used a different `attack_start_round` than `EXP_014`, any difference in FedTrace v3's onset-localization error between the two runs could not be attributed to seed alone — it could equally be an artifact of the different onset round interacting with `T_ref`'s fixed calibration window, or of a different post-onset window length affecting how much post-attack data the persistence rule has to work with. Holding `client_002`/`round 20`/`ρ=1.0`/`target_cosine=0.0` fixed makes seed the **only** varying factor between a Stage B run and its `EXP_014`/`EXP_015` development counterpart — exactly the single-variable-change discipline this project has applied to every experiment since `EXP_007`.

## 5. Exact Experiment Matrix

| Provisional ID | Attack family | Seed | Attack parameters | Paired honest baseline |
|---|---|---|---|---|
| `EXP_016` | Gaussian update-noise | 123 | `client_002`, round 20, `ρ=1.0` | `EXP_008` |
| `EXP_017` | Gaussian update-noise | 2026 | `client_002`, round 20, `ρ=1.0` | `EXP_009` |
| `EXP_018` | Norm-preserving directional | 123 | `client_002`, round 20, `target_cosine=0.0` | `EXP_008` |
| `EXP_019` | Norm-preserving directional | 2026 | `client_002`, round 20, `target_cosine=0.0` | `EXP_009` |

Four new attacked runs, minimum, for a 2-attack-family × 2-seed matrix. No config file is created by this plan (§20 gates config creation behind PI approval). Provisional naming, following this project's existing convention exactly (`EXP_014`/`EXP_015`'s own naming pattern): `cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed123` (`EXP_016`), `..._seed2026` (`EXP_017`), `cifar100_fedavg_dirichlet_a01_direction_attack_client002_round20_seed123` (`EXP_018`), `..._seed2026` (`EXP_019`).

## 6. Scientific Pairing

| Stage B run | Paired honest baseline | Shared, unchanged settings |
|---|---|---|
| `EXP_016` (seed 123, Gaussian) | `EXP_008` | |
| `EXP_017` (seed 2026, Gaussian) | `EXP_009` | CIFAR-100, ResNet-18, 5 clients, all 5 selected every round, Dirichlet α=0.1, FedAvg, 50 rounds, `local_epochs=1`, SGD, `lr=0.01`, identical partition realization for the matching seed |
| `EXP_018` (seed 123, directional) | `EXP_008` | |
| `EXP_019` (seed 2026, directional) | `EXP_009` | |

**The only scientific difference between each Stage B run and its paired honest baseline is the frozen attack scenario itself** (§4) — every other training setting is identical, matching this project's single-variable-change discipline exactly as it was already applied between `EXP_007`/`EXP_014` (seed 42 honest vs. seed 42 Gaussian-attacked) and `EXP_007`/`EXP_015` (seed 42 honest vs. seed 42 directional-attacked).

## 7. Trace Requirements

Every Stage B run must produce the current 18-field `client_round_trace.csv` schema (`src/attacks/traceability.py::TRACE_CSV_FIELDNAMES`), including `update_l2_norm`, `peer_mean_cosine_similarity`, `self_prev_round_cosine`, the attack ground-truth fields (`attack_configured`, `attack_active`, `attack_type`, `attack_strength_rho`, `malicious_client_id`, `attack_start_round`), and, for the directional runs (`EXP_018`/`EXP_019`), the directional verification fields (`directional_target_cosine`, `delta_attack_norm_ratio`, `realized_cosine_to_honest`). **FedTrace v3 scoring happens entirely offline, after each run completes**, via `src/analysis/fedtrace_detector.py`'s already-implemented, unmodified v3 functions — no detector code is touched based on anything observed in a Stage B trace, at any point, for any reason (this is the operational meaning of §1's freeze).

## 8. Required Validation Detectors

Pre-registered, per run:

- **A. Peer-cosine baseline** — the existing `apply_naive_detector`/`evaluate_naive_baseline` output, unchanged.
- **B. Magnitude-only** — `z_m` alone.
- **C. Temporal-residual-only (v3)** — `z_t_res` alone.
- **D. FedTrace v3** — `max(z_m, z_t_res)`, the frozen primary method.

v1's (`z_t`) and v2's (`z_t_rel`) results **may** be reported alongside for historical context (they are cheap to compute from the same trace, already implemented, and directly illustrate why v3 was necessary) but are **not** primary Stage B baselines and do not factor into the PASS/PARTIAL/FAIL decision in §18.

## 9. Per-Run Evaluation Metrics

For every Stage B attacked run:

- Predicted malicious client
- Attribution correctness
- First anomalous round
- Confirmation round
- Onset estimate
- Onset localization error (`|onset_estimate − 20|`)
- Detection delay (`confirmation_round − 20`)
- Malicious client-round precision / recall / F1 (same event framing as every prior document in this series)
- Honest persistent false-positive count, and exact identities of any honest client(s) confirmed
- Honest persistent false-positive rate (confirmed honest clients / 4 honest clients, per run)
- Dominant FedTrace signal (`z_m` vs. `z_t_res`, per anomalous client-round, counted separately per run)
- Global-model test Top-1 / Top-5 / loss, reported alongside the paired honest baseline's corresponding final-test values (`EXP_008`'s or `EXP_009`'s `final_test_metrics.json`), and the signed degradation (attacked − honest)

## 10. Primary Stage B Success Criteria (Per Run)

FedTrace v3 passes an individual Stage B run only if **all** of the following hold:

1. `predicted_client == client_002`.
2. No honest client is confirmed (persistent detection) before `client_002`'s own confirmation round.
3. **Onset localization error `<= 2` rounds.** Justification: `T_ref=20` is fixed, and every development-run success case (`EXP_014` magnitude-only, `EXP_015` v3 temporal-residual) achieved onset error of exactly `0` — the signal separation was immediate and total in both cases. A `<=2`-round tolerance allows for a Stage B seed's honest heterogeneity to make the signal ramp up slightly less instantaneously than at seed 42, without allowing a tolerance so loose it would accept a detector that is only vaguely in the right neighborhood of the true onset.
4. **Detection delay `<= 4` rounds.** Justification: under the frozen `P=3` persistence rule, the *best possible* delay is always exactly `2` (confirmation requires observing 3 consecutive anomalous rounds starting at the true onset, i.e. `onset + 2`) — every development-run success case achieved exactly this minimum. A `<=4`-round bound permits up to two additional rounds of ramp-up before the persistent run begins, directly consistent with (double) the onset-error tolerance in criterion 3, while still ruling out anything resembling `EXP_015`'s magnitude-only development result (a 25-round delay) or `EXP_014`'s never-relevant temporal-signal absence.
5. **Honest persistent false positives controlled**, defined as: zero honest clients confirmed. (Not "materially reduced" — v3's own development-run result on `EXP_015` was exactly zero, §17 of `analysis/FEDTRACE_V3_RETROSPECTIVE_ANALYSIS.md`, so the bar for an independent validation pass is that same standard, not a weaker one.)
6. No detector parameter was changed to obtain this result (structural — enforced by §1's freeze applying uniformly to every run, not run-by-run).

A run that fails even one of these criteria does not pass §10 for that run, regardless of how well it does on the others — see §13.

## 11. Cross-Seed Success Criteria (Per Attack Family)

For **each** attack family (Gaussian, directional) to be considered independently validated:

- §10's criteria met on **both** seed 123 and seed 2026 for that family.
- No honest-before-malicious confirmation in either seed for that family.
- Useful onset localization (§10 criterion 3) in both seeds for that family.
- Controlled honest persistent false positives (§10 criterion 5) in both seeds for that family.

**Even one validation failure, within a family, blocks a strong generalization claim for that family.** A single failed seed (out of two) is not treated as "50% success" or averaged away — per this project's own repeatedly-stated non-overclaiming discipline (already applied to every `n=1`/`n=2`/`n=3` result across `EXP_007`–`EXP_015`), two seeds is too small a sample to average over in the first place, and a failure on either one is direct, first-hand evidence that the method's reliability is seed-dependent for that attack family — the correct response is to report it as a failure for that family (§13, §18), not to claim a fractional or averaged success.

## 12. Attack-Family Coverage Question

**The central scientific question:** does the frozen FedTrace v3 provide reliable traceability across *both* a magnitude-revealing attack and a norm-preserving directional attack, on unseen strong-Non-IID seeds — without knowing in advance which family a given attack belongs to?

This is explicitly **not** the same question as whether FedTrace v3 beats the best attack-specific single detector on that detector's own best attack. Per `analysis/FEDTRACE_V3_RETROSPECTIVE_ANALYSIS.md` §14, FedTrace v3's development-run behavior is already known to be numerically identical to `A_magnitude_only` alone on `EXP_014` and to `E_temporal_residual_only` alone on `EXP_015` — this is expected, already-documented "coverage, not fusion" behavior, and Stage B is **not** designed to test whether that changes. **FedTrace v3 is not required to beat magnitude-only on the Gaussian family, nor temporal-residual-only on the directional family, at Stage B.** What Stage B tests is whether the *single, fixed* `FedTraceV3Score` — deployed without knowing in advance which attack family it will face — reliably tracks whichever single detector happens to be the right one for a given (unseen-seed) run, across both families, without needing to be told which one to use. That is the coverage claim this whole series of documents has built toward, and it is what §18's PASS determination is actually about.

## 13. Validation Failure Protocol

**If any Stage B run fails §10, or any attack family fails §11:**

- **Do not tune FedTrace v3.** No change to threshold, `P`, `T_ref`, the `z_t_res` formula, the `z_m` formula, the combiner, or any missing-value/tie-breaking rule, under any circumstance, based on a Stage B result.
- **Do not select a new formula using the failed validation seed.** If a future redesign is ever undertaken, it must not use the failed Stage B seed as informal design/debugging data — doing so would silently convert validation data into development data for whatever comes next, exactly the boundary this whole document exists to prevent crossing.
- **Record the failure** in full (the same metric suite as §9, computed and reported, not omitted or softened).
- **Classify its cause**, distinguishing at minimum:
  - *Implementation/infrastructure failure* — e.g. the attack did not activate correctly, the trace schema is malformed, a partition-realization or determinism issue produced an unexpected honest-phase divergence from the paired baseline, or the offline detector code itself errored. These are re-runnable once fixed and do not, by themselves, constitute evidence against FedTrace v3's method (§16 distinguishes this from a genuine scientific failure at the point of first evidence, not after the fact).
  - *Scientific detector failure* — the run executed correctly, the trace is valid, and FedTrace v3, applied exactly as frozen, still failed one or more §10 criteria. This **is** evidence against the method as currently specified.
- **Preserve the failed run as permanent validation evidence.** It is not deleted, excluded, or treated as a discardable pilot — once a run is used for Stage B validation (as opposed to a pre-registered implementation-correctness check per §16), its outcome counts, pass or fail.
- **Decide, explicitly and separately from any individual run's outcome:** whether the overall claim must be narrowed (§18 PARTIAL) or whether a future v4 would require a genuinely new development/validation cycle — meaning fresh development data (not `EXP_014`/`EXP_015`, not the failed Stage B run) and, subsequently, entirely new, not-yet-collected validation seeds (not seed 123 or 2026 again, since both would by then have been used to shape the redesign).

**Validation data must never silently become development data.** This is the single rule this entire section exists to enforce.

## 14. Global-Model Damage Analysis

Required, per run, as a paired comparison against the corresponding honest baseline (`EXP_008` or `EXP_009`): final test loss, Top-1, Top-5, and the validation-accuracy trajectory in the rounds immediately surrounding `attack_start_round=20` (matching the round-block marginal-gain convention already used throughout `analysis/EXP_014_Gaussian_Attack_Traceability_Analysis.md` and `analysis/EXP_015_Directional_Attack_Traceability_Analysis.md`). This project's own prior evidence already shows the two attack families produce very different degradation profiles at seed 42 (`EXP_014`: essentially neutral; `EXP_015`: a real, sustained ≈5–6pp accuracy loss) — **Stage B does not require every attack, or every seed, to reproduce a large accuracy drop.** The security/traceability claim this project is building toward does not depend on the attack being catastrophic to global performance; a stealthy, low-damage attack that is nonetheless correctly traced is not a lesser result for this paper's purposes than a damaging one. Degradation is reported for completeness and for the eventual security narrative, not treated as a pass/fail criterion for FedTrace v3 itself (§10/§11 govern that).

## 15. Statistical Discipline

**Stage B provides exactly two unseen seeds per attack family (four attacked runs total).** This materially improves reproducibility evidence over the single seed-42 development runs, but it remains a small sample by any standard, and this document commits, in advance, to **not** claiming broad statistical significance, a success "rate," a confidence interval, or universal robustness from four runs. Every Stage B result must be reported per-seed, transparently, exactly as run — never pooled into an aggregate percentage that would misrepresent `n=2` as if it supported a distributional claim. This mirrors the discipline already applied to `EXP_007`–`EXP_009`'s own three-seed honest baseline (`analysis/EXP_009_Three_Seed_Baseline_Analysis.md` §self-description: "three seeds support descriptive summary statistics only").

## 16. Experiment Execution Order

**Recommended order: `EXP_016` → (implementation-correctness checkpoint) → `EXP_017` → `EXP_018` → `EXP_019`.**

`EXP_016` (Gaussian, seed 123) runs first, for two reasons: (1) the Gaussian attack transform is the simpler of the two (additive, norm-rescaled noise; no Gram-Schmidt rotation or degenerate-direction handling), so if a seed-123-specific implementation surprise exists — e.g. an unexpected interaction between the new partition realization and the deterministic attack-seed derivation, or a trace-schema regression — it is easier to diagnose against the simpler transform; (2) this mirrors the exact order already used when both attacks were first developed at seed 42 (Gaussian piloted in `EXP_014` before the directional attack was designed and piloted in `EXP_015`), keeping the same low-risk-first sequencing this project has already validated as sound practice.

**After `EXP_016` completes, before running `EXP_017`–`EXP_019`:** an **implementation-correctness checkpoint**, distinct from and prior to any scientific detector evaluation:
- Confirm `EXP_016`'s pre-attack rounds (0–19) match `EXP_008`'s honest trajectory exactly where they are expected to (identical partition, identical seed, identical code up to round 20) — any unexpected divergence indicates an infrastructure problem, not a FedTrace finding, and must be resolved before proceeding.
- Confirm the trace CSV has the full 18-field schema, `attack_active` flips to `True` for `client_002` at exactly round 20, and `attack_strength_rho`/`malicious_client_id`/`attack_start_round` are logged as configured.
- Confirm the run completed all 50 rounds without error.

**Only if this checkpoint passes** does `EXP_017` (completing the Gaussian family) proceed, followed by `EXP_018` and `EXP_019` (the directional family). **Do not run all four runs unattended before this checkpoint** — an implementation failure discovered only after all four GPU runs complete would waste the entire Stage B compute budget re-diagnosing the same root cause four times over, and (per §13) would risk conflating an infrastructure problem with a genuine scientific finding if not caught early.

**Distinguishing implementation failure from scientific detector failure, operationally:** an implementation failure is diagnosable *before* FedTrace v3 is ever applied — it shows up in the checkpoint above (schema, activation, honest-phase fidelity), independent of any detection outcome. A scientific detector failure can only be diagnosed *after* a run has already passed that checkpoint and FedTrace v3 has been applied and found wanting against §10. The checkpoint exists specifically to make sure these two categories are never confused with each other.

## 17. Analysis Workflow

For each of the four runs, in the order given in §16:

1. Verify attack correctness (attack activated for `client_002` at round 20, deactivated/absent for every other client, consistent with §4's frozen parameters).
2. Verify trace integrity (§7's schema, no missing expected columns, no unexpected `None`s outside the already-documented semantics — e.g. round 0's `self_prev_round_cosine`).
3. Apply the frozen FedTrace v3 detector (§8) offline, exactly as implemented, computing every §9 metric.
4. Record the full result set (§9) — nothing summarized away or omitted, pass or fail.

**The method itself is not changed between runs, under any circumstance** — this is the same freeze as §1/§13, restated here because it governs the *workflow*, not just the formulas: even an early, favorable `EXP_016` result must not tempt a "small improvement" before `EXP_017`–`EXP_019` run.

After all four runs have valid results (§16's checkpoint passed, §17 steps 1–4 completed for each), **exactly one Stage B synthesis document** is created, applying §10/§11's pre-registered criteria to the full four-run matrix and reaching one of §18's three categories.

## 18. Stage B Final Decision Categories

Defined now, before any Stage B result exists:

**PASS.** FedTrace v3 independently validates sufficiently for the flagship paper: §10 is met on **all four** Stage B runs, and §11's cross-seed criteria are met for **both** attack families.

**PARTIAL.** FedTrace v3 works reliably for one attack family (both its seeds pass §10 and §11) but not the other, **or** cross-seed consistency is limited within a family in a way that does not amount to a clean §11 pass or a clean failure (e.g. one seed passes every §10 criterion and the other narrowly fails exactly one, in a way judged, at synthesis time, not to indicate a structural problem but also not to support an unqualified pass — this judgment must be argued explicitly in the synthesis document, not asserted). The resulting claim must be narrowed accordingly (e.g. "validated for the directional family; Gaussian-family validation is inconclusive/failed") rather than stated as a general FedTrace v3 result.

**FAIL.** Independent validation does not support the frozen traceability method as currently specified — either attack family fails §11 outright (both seeds, or a seed failure judged structural rather than borderline), or a genuine scientific detector failure (§13) is found and judged, at synthesis time, to undermine the core coverage claim (§12).

These three definitions are fixed now and are not to be adjusted, loosened, or reinterpreted once Stage B results exist.

## 19. What Happens After Stage B

Not committed to now — described only conditionally, per this task's explicit restriction against pre-committing to further experiments:

- **If PASS:** the traceability method has independent-seed support across both developed attack families; a plausible next phase (not authorized by this document) would be security-aware aggregation or mitigation building on a validated detector, or extending coverage evaluation to additional attack families under a fresh development/validation cycle of its own.
- **If PARTIAL:** the claim in any resulting write-up must be narrowed to whatever the evidence actually supports (e.g. one attack family only), and a decision would be needed on whether the weaker family is worth a fresh method-development cycle (new diagnosis, new design, new — different — validation seeds) or is reported as a known, honest limitation.
- **If FAIL:** the result is not hidden or reframed — it is reported as a failure of the frozen method as validated, and the FedTrace methodology (not merely its v3 parameters) would need reconsideration, following the same non-overclaiming discipline this entire document series has applied throughout.

No specific next experiment beyond `EXP_016`–`EXP_019` is proposed or authorized here.

## 20. PI Approval Checklist (Required Before `EXP_016` Config Creation)

All of the following must be explicitly confirmed before any Stage B config file is created:

- [ ] This document (`research_plans/FEDTRACE_STAGE_B_INDEPENDENT_VALIDATION_PLAN.md`) has been read and approved by the PI/user.
- [ ] §1's frozen formulas have been checked, line by line, against `src/analysis/fedtrace_detector.py`'s current v3 implementation and against `research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md` §16, and all three are confirmed identical.
- [ ] `tests/test_analysis_fedtrace_detector.py`'s full suite (63 tests, per this task's stated status) has been (re-)run and independently confirmed passing at 63/63 immediately prior to Stage B, not merely cited from an earlier report.
- [ ] `EXP_008`'s and `EXP_009`'s existing results (`round_history.csv`, `summary.json`, `final_test_metrics.json`, `client_round_trace.csv` where applicable) are confirmed present and readable, so each Stage B run has a valid paired baseline already on disk before it starts.
- [ ] GPU time budget for four ≈50-round CIFAR-100/ResNet-18/5-client runs (per-run duration precedent: `EXP_014`/`EXP_015` each took on the order of 90 minutes) has been confirmed available.
- [ ] The provisional experiment IDs and naming convention in §5 are confirmed to not collide with any existing or already-planned experiment.
- [ ] The §16 execution order (and its implementation-correctness checkpoint after `EXP_016`) is explicitly agreed — no request to run all four in parallel or out of this order.
- [ ] The §13 validation failure protocol and §18 PASS/PARTIAL/FAIL definitions are explicitly acknowledged as binding before any result exists.
- [ ] Explicit, written PI/user go-ahead to create `EXP_016`'s config specifically (not a blanket approval to create all four at once) — consistent with §16's one-at-a-time-until-checkpoint discipline.
