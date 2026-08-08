# EXP_015 — Norm-Preserving Directional Attack: Benchmark Design Plan

**Status:** Planning document only. No attack code exists yet for this
design, no `EXP_015` config exists, and no experiment has been run. This
plan pre-registers the design, the deterministic construction, the
ground-truth/diagnostic requirements, and the decision criteria **before**
any of that code is written, per this project's standing "protocol before
implementation" workflow (already applied to `EXP_014`'s
`gaussian_update_noise` pilot —
`research_plans/PHASE5_Malicious_Client_Traceability_Plan.md`).

**This document does not implement anything.** No code, config, test,
README, or Research Database file was created or modified in producing
it. No command was executed. **This is still benchmark construction, not
the FedTrace algorithm** — see Section 20.

---

## 1. Motivation From EXP_014

`analysis/EXP_014_Gaussian_Attack_Traceability_Analysis.md` and
`analysis/EXP_014_Norm_Aware_Attribution_Study.md` established, with
already-collected `EXP_014` data:

- The `gaussian_update_noise` attack at `rho=1.0` produced a
  **~41–47% norm inflation** on `client_002`'s submitted update
  (theoretical `√2 ≈ +41.4%`, observed `+43.4%` immediate / `+47.0%`
  full-window mean).
- A trivial, three-parameter per-client norm z-score detector
  (`z_norm = (update_l2_norm − mean_pre) / std_pre`, flag at `z_norm > 3`)
  achieved **100% recall, effectively 100% precision, zero detection
  delay, and zero localization error** — every one of `client_002`'s 30
  post-attack rounds registered `z_norm > 9`, roughly 7x the highest
  value any honest client ever reached.
- `analysis/EXP_014_Attack_Difficulty_Calibration.md` showed
  analytically that this near-perfect result is not incidental: any
  attack built from *additive independent noise* mechanically inflates
  L2 norm (`‖delta+n‖₂/‖delta‖₂ ≈ √(1+ρ²)` for near-orthogonal noise),
  so the `gaussian_update_noise` family leaks a magnitude signal by
  mathematical construction, at any `rho` large enough to matter.

**Therefore: the next benchmark must suppress magnitude leakage
entirely**, not merely recalibrate it to a harder-but-still-solvable
`rho`. `analysis/EXP_014_Attack_Difficulty_Calibration.md` §§7–8
concluded a norm-preserving directional attack is the scientifically
relevant next challenge (over a lower-`rho` Gaussian recalibration),
because this project's own evidence shows *honest* Non-IID heterogeneity
is dominated by *directional* variation (the failed peer-cosine
detector), not magnitude variation (pre-attack norm CVs of only
4.5%–8.3% across honest clients) — a genuinely useful attribution method
must eventually separate malicious *direction* changes from that same
honest directional drift, which no norm-based signal, however finely
tuned, can be made to test.

---

## 2. Exact Research Question

> **Can a malicious directional perturbation be detected and localized
> when the submitted update norm remains approximately indistinguishable
> from the honest update norm?**

This is the direct successor to `EXP_014`'s question — same threat model,
same client, same onset round, same honest baseline — with exactly one
new scientific variable: the attack no longer perturbs magnitude at all,
isolating whether *any* of the already-collected or newly-proposed cheap
diagnostics (Section 12) carry a usable signal once the trivially
effective norm channel is closed off.

---

## 3. The Honest Update

Unchanged from `EXP_014`/`gaussian_update_noise`
(`src/attacks/traceability.py`), and from every prior experiment in this
project's federated stack:

```
delta = local_model_state - global_model_state   (trainable parameters only)
```

computed at the same injection point as before — see Section 6.

---

## 4. Attack Design: A Norm-Preserving Rotation

**Chosen: the simplest mathematically interpretable norm-preserving
transformation — a controlled rotation of the honest update's direction
within a random, delta-orthogonal 2-plane, exactly rescaled to the
honest norm.** No learned, adaptive, or iteratively-optimized attack is
considered.

Construction (deterministic, pure function of `delta`, a target severity
parameter, and the seed — see Sections 5–7):

1. **Honest direction:** `delta_hat = delta / ‖delta‖₂`.
2. **Random orthogonal direction:** draw raw i.i.d. standard-normal noise
   `u_raw` (same shape as `delta`, same deterministic-seeding mechanism
   already used by `generate_gaussian_attack_noise`, Section 5).
   Gram-Schmidt-orthogonalize against `delta_hat`:
   `u_perp_raw = u_raw − ⟨u_raw, delta_hat⟩·delta_hat`, then normalize:
   `u_perp = u_perp_raw / ‖u_perp_raw‖₂`. `u_perp` is, by construction,
   an exact unit vector orthogonal to `delta_hat`.
3. **Rotate:** for a target angle `theta` (Section 7),
   `direction_attack = cos(theta)·delta_hat + sin(theta)·u_perp`. Since
   `delta_hat` and `u_perp` are an orthonormal pair,
   `‖direction_attack‖₂ = 1` **exactly** (`cos²θ + sin²θ = 1`), and
   `⟨direction_attack, delta_hat⟩ = cos(theta)` **exactly**.
4. **Rescale to the honest norm:**
   `delta_attack = ‖delta‖₂ · direction_attack`.

By construction, `‖delta_attack‖₂ = ‖delta‖₂` exactly (up to
floating-point rounding — no post-hoc rescaling-of-independent-noise
approximation is involved, unlike `gaussian_update_noise`, because
`direction_attack` is an exact unit vector by the orthonormal-rotation
identity, not an empirically-rescaled random draw), and
`cos(delta_attack, delta) = cos(theta)` exactly.

**Only trainable parameters are perturbed**, matching every prior attack
and FedProx convention in this codebase (`traceability.py`'s
`apply_gaussian_update_noise` scope) — non-trainable buffers (BatchNorm
running stats, `num_batches_tracked`) are copied through unchanged.

---

## 5. Determinism

Identical mechanism to `generate_gaussian_attack_noise`
(`traceability.py:_derive_attack_seed`): `u_raw` is drawn from a
dedicated `torch.Generator(device="cpu")` seeded via
`sha256(f"{seed}-{round_id}-{client_id}")` (masked to 63 bits), never the
global RNG. Given identical `(delta, seed, round_id, client_id, theta)`,
the construction in Section 4 is a pure, reproducible function — no other
randomness is introduced anywhere in the transformation.

---

## 6. Preserved Scientific Settings

Every setting matches `EXP_014` exactly, with the local objective now the
directional attack (Section 4) in place of `gaussian_update_noise`:
`malicious_client_id: client_002`, `attack_start_round: 20`, `seed: 42`
(first pilot only — see Section 13), `partition: dirichlet`,
`dirichlet_alpha: 0.1`, `aggregation: fedavg`, `communication_rounds: 50`,
`local_epochs: 1`, and all other training settings identical to
`configs/cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed42.yaml`
(model, optimizer, learning rate, momentum, weight decay, batch size,
client counts, validation/checkpoint settings, device, mixed precision,
failure handling). **The injection point is unchanged**: inside
`FederatedRunner.run_round()`, via the existing, fully generic
`update_transform_callback` hook (`src/federated/runner.py`), applied
after `client.get_model_weights()` (honest local training already
complete) and before `ClientUpdate.create()`/`self.server.receive_update()`
— exactly the same architectural point verified safe in
`analysis/PHASE5_Implementation_Verification.md` items 3–8. No change to
`FederatedRunner`, `FederatedClient`, or `FedAvg` is required by this
design.

---

## 7. Strength/Control Parameter: Target Directional Cosine

**One parameter, with an exact geometric meaning: `target_cosine ∈ [−1, 1]`**
— the intended cosine similarity between `delta_attack` and the client's
own honest `delta` that round (`theta = arccos(target_cosine)` in
Section 4's rotation). This is not an arbitrary magic constant: by
Section 4's construction, `cos(delta_attack, delta) = target_cosine`
**exactly**, so the parameter *is* the quantity it controls, verifiable
directly (Section 10).

- `target_cosine = 1.0` → `theta = 0°` → no-op (recovers the honest
  update exactly, the required inert-parameter property, mirroring
  `rho=0` for `gaussian_update_noise`).
- `target_cosine = 0.0` → `theta = 90°` → the attacked update is exactly
  as different in direction from the honest update as a random orthogonal
  vector, while retaining the honest norm.
- `target_cosine = -1.0` → `theta = 180°` → exact reversal (the most
  extreme, most "obviously adversarial" case — an update pointing
  directly away from the client's own honest gradient).

---

## 8. Recommended Initial Severity: `target_cosine = 0.0`

**Analytical justification, avoiding both failure extremes:**

- **Not near-no-op.** `target_cosine` near `1.0` would leave
  `delta_attack` nearly indistinguishable from `delta` in direction as
  well as magnitude — an uninformative pilot that tests nothing new.
  `target_cosine = 0.0` is maximally far from this failure mode along
  the monotonic `cos(theta)` scale (`1 → 0 → −1` as `theta` runs
  `0° → 90° → 180°`).
- **Not trivially destructive.** `target_cosine = -1.0` (exact reversal)
  is the single most extreme point on the scale — the update points
  directly opposite the client's own honest gradient direction, which
  risks conflating this new attack family with a qualitatively different,
  cruder one (a sign-flip attack) and risks a large, easily-attributable
  global-model impact that could saturate Section 17's "global model not
  immediately destroyed" success criterion before any subtler attribution
  question can be studied.
- **`target_cosine = 0.0` (orthogonal) is the natural, parameter-free
  midpoint** of the meaningful `[-1, 1]` range in cosine terms (not
  merely in angle terms) — it is "as directionally different as
  possible without being defined *relative to* the reversal case,"
  giving a clean, easily-interpreted reference point: an attacked update
  that is a completely different direction, at the honest magnitude,
  with zero residual alignment to what the client would honestly have
  submitted.
- It also gives Section 10's verification the cleanest possible signal:
  measured `cos(delta_attack, delta) ≈ 0.0` is trivially distinguishable
  from an honest client's typical self-consistency (expected to be much
  closer to `1.0` for smoothly-evolving successive honest updates, though
  this is not measured by this plan — see Section 12).

---

## 9. Norm-Preservation Verification

**Invariant:** for every active client-round,

```
| ‖delta_attack‖₂ − ‖delta_honest‖₂ | / ‖delta_honest‖₂  <  tolerance
```

**Tolerance: `1e-4` (relative).** Because Section 4's construction is an
*exact* orthonormal rotation (not an empirical rescaling of independent
noise, as `gaussian_update_noise` requires), the only expected source of
deviation is floating-point rounding across an ~11.2M-dimensional
`float32` vector — `1e-4` relative is generously loose relative to that
expected rounding floor and should hold with wide margin if the
implementation is correct; a violation would indicate an implementation
defect, not an inherent property of the design, and must be caught by
the required tests (Section 21) before any GPU run.

---

## 10. Direction-Change Verification

**Metric:** `cos(delta_attack, delta_honest) = ⟨delta_attack, delta_honest⟩ / (‖delta_attack‖₂ · ‖delta_honest‖₂)`,
computed exactly as `traceability.py::compute_peer_mean_cosine_similarity`
already computes cosine similarity between two flattened, trainable-key
vectors (same helper machinery, applied to a different pair of vectors).

**Proof the attack materially changed direction:** the measured value
must equal `target_cosine` (Section 7–8) within a tight numerical
tolerance (e.g. `1e-3` absolute, looser than Section 9's since this
involves an additional normalization division) — this is both a
correctness check (implementation matches the intended geometry) and,
substantively, direct evidence that `client_002`'s submitted direction at
active rounds is genuinely different from what it would honestly have
submitted (as opposed to, e.g., a near-zero perturbation that happens to
still satisfy the norm invariant).

---

## 11. Why This Attack Defeats `update_l2_norm`-Only Attribution, By Design

Because `‖delta_attack‖₂ = ‖delta_honest‖₂` **exactly** (Section 9), the
`update_l2_norm` diagnostic and any statistic derived purely from it
(including the `z_norm` detector evaluated in
`analysis/EXP_014_Norm_Aware_Attribution_Study.md`) will show **zero**
expected deviation from `client_002`'s own honest reference distribution
on every attacked round — `z_norm ≈ 0`, by mathematical construction, not
by a threshold being merely "hard to reach." This is categorically
different from lowering `gaussian_update_noise`'s `rho`
(`analysis/EXP_014_Attack_Difficulty_Calibration.md` §§4–6): a lower
`rho` still leaves *some* magnitude signal, shrinking but never
literally zero; here there is no magnitude signal to find, at any
detection threshold, because the feature itself carries no information
about the attack.

---

## 12. Cheap Signals That May Still Reveal the Attack (Not Designed Here)

**Named as candidates only — no detection algorithm is designed or
evaluated in this document; this is explicitly not FedTrace design
(Section 20).**

- **`peer_mean_cosine_similarity`** (already logged, Sprint Phase 5): the
  attacked update's *direction* has changed, so its alignment with peers
  that round may shift — though `analysis/EXP_014_Gaussian_Attack_Traceability_Analysis.md`
  §4 already showed this feature is dominated by a large, shared,
  attack-independent secular trend under the *magnitude*-based attack;
  whether it behaves differently under a *directional* attack is an open
  question this plan does not answer.
- **Client historical (self) cosine — proposed as a new diagnostic
  (Section 15):** cosine similarity between a client's current-round
  submitted update and that *same client's own* immediately-preceding
  round's submitted update. Honest updates plausibly evolve smoothly
  round to round (same local data, a slowly-shifting global model); an
  abrupt, sustained drop in this self-consistency measure exactly at the
  attack's onset round would be a genuinely temporal signature that a
  single-round magnitude or peer-comparison snapshot cannot express.
- **Directional residual against a robust reference** (e.g. distance from
  the coordinate-wise median direction of that round's selected clients)
  — named in the original Phase 5 plan's cheap-feature candidate list
  (`research_plans/PHASE5_Malicious_Client_Traceability_Plan.md` §10) but
  not yet implemented; potentially informative here since it does not
  rely on magnitude at all.
- **Temporal change-point behavior more broadly** — any of the above
  features' round-to-round *volatility*, rather than their single-round
  value, evaluated for a discontinuity coinciding with round 20.

---

## 13. First Pilot Experiment

**Provisional ID: `EXP_015`.**
**Provisional name:** `cifar100_fedavg_dirichlet_a01_direction_attack_client002_round20_seed42`.

Seed 42 only, for the same reasons `EXP_014` piloted at seed 42 only
before any paired-seed commitment (`research_plans/PHASE5_Malicious_Client_Traceability_Plan.md`
§16): this is a first-of-kind code path (a new attack transform,
requiring new verification fields) that has not yet executed on real
hardware. `malicious_client_id: client_002`, `attack_start_round: 20`,
`target_cosine: 0.0` (Section 8). No config file is created by this plan.

---

## 14. Ground-Truth Logging Requirements

**Keep every existing `EXP_014` trace field unchanged**
(`traceability.py::TRACE_CSV_FIELDNAMES`): `round_id`, `client_id`,
`selected`, `attack_configured`, `attack_active`, `attack_type`,
`attack_strength_rho`, `malicious_client_id`, `attack_start_round`,
`update_l2_norm`, `peer_mean_cosine_similarity`, `naive_threshold`,
`naive_flag`, `first_flagged_round`. (`attack_type` becomes the new
attack's identifier, e.g. `direction_rotation`; `attack_strength_rho`
remains present for schema stability but is not the severity parameter
for this attack family — see Section 15 for the field that is.)

**Minimum additional fields required, specifically to verify Sections
9–10's invariants:**

- `directional_target_cosine` — the configured `target_cosine` (Section
  7), constant per run, logged on every row (mirrors how
  `attack_strength_rho` is logged on every row today).
- `delta_attack_norm_ratio` — `‖delta_attack‖₂ / ‖delta_honest‖₂` for the
  malicious client on active rounds (ground-truth verification of
  Section 9's invariant; `None` for honest clients/inactive rounds).
- `realized_cosine_to_honest` — the measured
  `cos(delta_attack, delta_honest)` (Section 10) for the malicious client
  on active rounds (`None` otherwise) — ground-truth verification that
  the realized direction change matches the configured target.

---

## 15. Required New Trace Diagnostic (Minimum Only)

**Exactly one new diagnostic feature is proposed, no more:**
`self_prev_round_cosine` — cosine similarity between a client's current-
round submitted update and that same client's own previous-round
submitted update. This is the minimum addition needed to make Section
12's "client historical cosine" candidate signal observable at all; it
requires only a one-round lookback buffer of each client's already-
computed submitted delta (no new tensor persistence, no raw-update
storage, reusing the same flatten/cosine helper already used for
`peer_mean_cosine_similarity`). No other new diagnostic is proposed —
layer-wise features, robust-center distance, and validation-impact
features remain explicitly deferred, exactly as they were for `EXP_014`
(`research_plans/PHASE5_Malicious_Client_Traceability_Plan.md` §10).

---

## 16. Naive Baseline Evaluation

**Both existing `EXP_014` naive baselines must be re-evaluated on
`EXP_015`'s trace, unmodified:**

1. **Peer-cosine 3-sigma detector** (`traceability.py::apply_naive_detector`)
   — run exactly as implemented for `EXP_014`, no code change.
2. **Norm z-score detector** (`analysis/EXP_014_Norm_Aware_Attribution_Study.md`'s
   offline method) — recomputed exactly as before, on `EXP_015`'s
   `update_l2_norm` column.

**Expected outcome, stated as a prediction to be checked, not a claimed
result:** per Section 11, the norm z-score detector should lose its
`EXP_014` advantage — by construction, `client_002`'s post-attack
`update_l2_norm` should be statistically indistinguishable from its own
pre-attack reference distribution, so `z_norm` should **not** exceed the
`3`-threshold on active rounds (beyond the same round-0 self-referential
artifact already documented in `analysis/EXP_014_Norm_Aware_Attribution_Study.md`
§5, unrelated to this attack). Whether the peer-cosine detector performs
any differently than it did under `EXP_014` is an open, unanswered
question this plan does not predict either way.

---

## 17. Success Criteria for the Attack Benchmark

`EXP_015` will be considered a **successful benchmark construction** if:

- The run completes 50/50 rounds with 5/5 successful client updates every
  round (matching every prior experiment's reliability record).
- Section 9's norm-preservation invariant holds within tolerance on every
  active client-round (`delta_attack_norm_ratio` within `1e-4` relative
  of `1.0`).
- Section 10's direction-change invariant holds (`realized_cosine_to_honest`
  within `1e-3` absolute of the configured `target_cosine`).
- The global model is **not immediately destroyed** — no non-finite loss,
  no catastrophic post-round-20 accuracy collapse relative to the paired
  honest baseline (`EXP_007`), consistent with a subtle, attributable
  attack rather than a training-breaking one.
- The **norm-only detector is no longer trivially perfect** — `z_norm`
  does not exceed `3` on `client_002`'s active rounds (beyond the
  documented round-0 artifact), i.e. Section 16's prediction is confirmed
  by the actual run, not merely predicted.
- **Attribution remains non-trivial** — neither existing naive baseline
  achieves the same clean, immediate, zero-false-positive identification
  `EXP_014`'s norm z-score detector achieved, confirming this benchmark
  is harder than its predecessor.

---

## 18. Failure Criteria

`EXP_015` will be considered **uninformative** under any of:

- **Too weak / no measurable effect:** the run shows no detectable
  difference from `EXP_007` in *any* logged diagnostic, at any round —
  would suggest `target_cosine = 0.0` was insufficient or the
  construction itself has no behavioral consequence worth studying (must
  first rule out an implementation defect via Section 21's tests before
  concluding this about the design itself).
- **Too destructive:** non-finite loss, a failed round, or a
  catastrophic, immediate global-model collapse right at round 20 —
  would indicate the rotation, despite preserving norm, is too
  aggressive for this configuration (e.g. if `target_cosine = 0.0`
  proves too extreme in practice, a future revision might consider a
  less extreme value, though not as a broad sweep).
- **Norm preservation broken:** `delta_attack_norm_ratio` measurably
  deviates from `1.0` beyond Section 9's tolerance — an implementation
  defect, not a property of the attack family; must be fixed before any
  scientific conclusion is drawn from the run.
- **Simple norm detector still trivially identifies the attacker:** would
  falsify Section 11's core design claim and require re-examining the
  construction (e.g. a subtle bug reintroducing a magnitude bias) before
  treating this as a harder benchmark than `EXP_014`.
- **No measurable behavioral effect anywhere** (distinct from "too weak"
  above): every diagnostic, including the new `self_prev_round_cosine`
  and `peer_mean_cosine_similarity`, is indistinguishable from `EXP_007`
  — would mean this benchmark cannot yet be used to evaluate *any*
  attribution method, naive or future, and a stronger `target_cosine`
  (still not a reversal) or a different diagnostic would need
  consideration.

---

## 19. Decision Tree After `EXP_015`

- **If Section 17's success criteria are met and both naive baselines
  fail (or perform materially worse than on `EXP_014`):** this is the
  target outcome — a genuinely harder benchmark now exists. Document the
  result and treat Section 12's candidate signals (self-cosine, peer
  cosine, directional residual, temporal change-points) as the motivated
  starting point for the eventual FedTrace design phase (still not begun
  by this plan).
- **If norm preservation or direction change fails verification
  (Section 18, "too destructive" or "norm preservation broken"):**
  treat as an implementation defect first (re-check against Section 21's
  tests); only reconsider the geometric design itself once implementation
  correctness is independently confirmed.
- **If the attack has no measurable effect on any diagnostic
  ("too weak"/"no measurable behavioral effect"):** consider a
  moderately less-orthogonal or moderately-more-adversarial single
  `target_cosine` value as the next single-variable pilot change (e.g.
  one alternative value, not a sweep) — but only after confirming via
  Section 21's tests that the construction itself is behaving as
  designed.
- **If the norm detector still trivially succeeds despite the
  invariant holding:** re-examine whether some other logged quantity
  (not norm itself) is inadvertently correlated with the attack in a way
  that leaks into the z-score computation — a targeted debugging
  question, not a broad re-design.

---

## 20. Explicit Boundary: Still Benchmark Construction, Not FedTrace

**This plan does not design the FedTrace attribution algorithm.** Its
purpose is identical in kind to `EXP_014`'s: construct one more
controlled, ground-truth-labeled attack scenario and evaluate the
*already-existing* naive baselines against it — nothing here proposes a
learned, temporal, or purpose-built detector. Section 12's candidate
signals are named, not designed, evaluated, or implemented. Any future
attribution mechanism built from those candidates is explicitly deferred
to a separate, later, separately pre-registered plan, consistent with
the original Phase 5 plan's §21 boundary.

---

## 21. Required Implementation Tests Before Any GPU Run

All fast, CPU-only, tiny-synthetic-data tests, mirroring
`research_plans/PHASE5_Malicious_Client_Traceability_Plan.md` §22's
discipline:

1. **No-op test:** `target_cosine = 1.0` produces `delta_attack` exactly
   equal to `delta_honest` (`theta = 0`, `cos(theta) = 1`,
   `direction_attack = delta_hat` exactly).
2. **Exact norm-preservation test:** for a range of `target_cosine`
   values (e.g. `-1.0, -0.5, 0.0, 0.5, 1.0`) on a tiny synthetic tensor,
   `‖delta_attack‖₂` matches `‖delta_honest‖₂` within Section 9's `1e-4`
   relative tolerance.
3. **Exact direction-change test:** for the same grid, measured
   `cos(delta_attack, delta_honest)` matches the configured
   `target_cosine` within Section 10's `1e-3` absolute tolerance.
4. **Orthogonality construction test:** `u_perp` (Section 4 step 2) is
   verified numerically orthogonal to `delta_hat` (dot product ≈ 0)
   before being used in the rotation, on a tiny synthetic tensor.
5. **Reversal test:** `target_cosine = -1.0` produces `direction_attack`
   exactly `-delta_hat` (the degenerate rotation case, `theta = 180°`).
6. **Determinism test:** identical `(delta, seed, round_id, client_id,
   target_cosine)` produces byte-identical `delta_attack` across two
   separate calls.
7. **Cross-round/client variation test:** a different `round_id` or
   `client_id` (all else equal) produces a different `u_perp` and hence a
   different `delta_attack` (for `target_cosine ≠ ±1`).
8. **Trainable-keys-only test:** non-trainable buffer keys (e.g.
   `bn.running_mean`, `bn.num_batches_tracked`) are copied through
   unchanged, mirroring `test_apply_gaussian_update_noise_perturbs_only_trainable_keys`.
9. **Selective-perturbation / honest-client-isolation tests:** mirroring
   `tests/test_runner.py`'s existing `update_transform_callback` test
   suite — only the designated malicious client's update is transformed;
   honest clients' own submitted updates are provably unaffected.
10. **`self_prev_round_cosine` correctness test:** on a tiny synthetic
    two-round sequence with known, hand-computable updates, the new
    diagnostic (Section 15) matches the expected cosine value.
11. **Degenerate-delta test:** behavior is explicitly defined (raise a
    clear error, mirroring `generate_gaussian_attack_noise`'s existing
    zero-norm handling) when `delta_honest` has zero L2 norm — a
    Gram-Schmidt/normalization step cannot proceed from a zero vector.

---

## 22. Reproducibility Requirements

- `target_cosine`, `malicious_client_id`, `attack_start_round`, and the
  attack-type identifier must be recorded in the experiment's config and
  echoed into `summary.json`'s `"attack"` block, exactly as `EXP_014`'s
  four attack fields already are.
- The rotation's random direction (`u_raw`) must use the same
  reproducible, explicitly-seeded mechanism as `gaussian_update_noise`
  (Section 5) — no reliance on uncontrolled global RNG state, so
  re-running an identical config reproduces an identical sequence of
  attacked updates.
- `run.log` must record attack activation at `round_id == attack_start_round`
  and on every subsequent active round, mirroring the existing
  `round_callback` convention.
- All new ground-truth/verification fields (Section 14) and the new
  diagnostic (Section 15) must be persisted in `client_round_trace.csv`
  alongside the unchanged `EXP_014` fields — never left implicit in log
  text.
- The test-set-touched-exactly-once discipline is preserved without
  exception, exactly as in every prior experiment.

---

## 23. PI Approval Checklist

Before any part of this plan is implemented or executed, confirm:

- [ ] This plan is reviewed and agreed as the correct next benchmark
      direction, given `analysis/EXP_014_Attack_Difficulty_Calibration.md`
      §§8, 10's conclusion that a norm-preserving directional attack is
      more scientifically relevant to the flagship objective than a
      lower-`rho` Gaussian recalibration.
- [ ] The rotation-based construction (Section 4) — the simplest
      mathematically interpretable norm-preserving transformation
      considered, over any learned/adaptive alternative — is agreed.
- [ ] `target_cosine` as the single severity parameter (Section 7), and
      `target_cosine = 0.0` as the first pilot value (Section 8), are
      agreed as reasonable, analytically-justified, explicitly
      non-optimal starting points.
- [ ] The injection point (Section 6, unchanged from `EXP_014`) is
      re-confirmed as architecturally safe, requiring no change to
      `FederatedRunner`, `FederatedClient`, or `FedAvg`.
- [ ] Sections 9–10's exact norm-preservation and direction-change
      invariants and tolerances are agreed before any code is written.
- [ ] The minimum additional ground-truth fields (Section 14) and the
      single new diagnostic (Section 15) are agreed as sufficient and
      not excessive.
- [ ] Section 16's requirement to re-evaluate both existing `EXP_014`
      naive baselines unmodified, and Section 16's stated (unconfirmed)
      prediction that the norm detector should lose its advantage, are
      agreed as the correct evaluation protocol.
- [ ] Success/failure criteria (Sections 17–18) and the decision tree
      (Section 19) are agreed in advance, before any GPU run.
- [ ] The explicit boundary (Section 20) — no FedTrace design in this
      phase — is agreed and will be respected regardless of how the
      pilot result looks.
- [ ] All required tests (Section 21) will be implemented and passing on
      CPU before any GPU run is attempted.
- [ ] No other file (code, config, test, prior experiment result, or
      Research Database entry) will be modified as part of implementing
      this plan until its own results are ready for documentation.
