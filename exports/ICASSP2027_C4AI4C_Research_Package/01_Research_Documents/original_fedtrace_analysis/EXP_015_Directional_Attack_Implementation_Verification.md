# EXP_015 Directional Attack Implementation Verification

**Audit date:** 2026-08-08
**Scope:** A complete implementation audit of the
`norm_preserving_directional` attack infrastructure added on top of the
Phase 5 / `EXP_014` pilot, against
`research_plans/EXP_015_Norm_Preserving_Directional_Attack_Plan.md`. This
document verifies implementation **correctness only** — no FedTrace
design, no new detector, no algorithmic improvements are proposed. No
code, test, config, README, or Research Database file was modified in
producing it.

**Files audited:**
- `research_plans/EXP_015_Norm_Preserving_Directional_Attack_Plan.md`
- `src/attacks/traceability.py`
- `src/experiments/run_federated_cifar100.py`
- `tests/test_attacks_traceability.py`
- `tests/test_federated_experiment.py`
- `README.md`

**Method:** every checklist item is verified against specific code
locations and, where one exists, the test that exercises the property.
No experiment was run to produce this audit — all evidence is static
(code reading) or drawn from the already-written test suite's own
assertions (not executed here, but their logic inspected directly).

---

## 1. Existing `attack_type=none` behavior remains inert

**PASS.** `AttackConfig`'s defaults are unchanged
(`attack_type="none"`, `malicious_client_id=None`,
`attack_start_round=None`, `attack_strength_rho=0.0`); the new
`directional_target_cosine=None` default does not alter `is_active()`,
which still short-circuits `False` whenever `attack_type != "none"` is
false (`traceability.py`, `is_active()` unchanged). Confirmed by
`test_attack_config_defaults_are_inert` and
`test_directional_target_cosine_is_inert_for_gaussian_and_none`.

## 2. Existing `gaussian_update_noise` implementation remains unchanged

**PASS.** `generate_gaussian_attack_noise`, `apply_gaussian_update_noise`,
and `_derive_attack_seed` are byte-for-byte identical to their
pre-`EXP_015` bodies — no line inside any of these three functions was
touched; all new code was added as new, separate functions
(`generate_directional_attack_perturbation`,
`apply_norm_preserving_directional_attack`, `_derive_directional_attack_seed`,
`compute_cosine_similarity`, `compute_norm_ratio`) placed adjacent to,
not inside, the existing ones. Every pre-existing Gaussian test in
`tests/test_attacks_traceability.py` is present, unmodified.

## 3. `norm_preserving_directional` is inactive before `attack_start_round`

**PASS.** `is_active()` is attack-type-agnostic (checks only
`attack_type != "none"`, `client_id == malicious_client_id`,
`round_id >= attack_start_round`) and was not modified to accommodate the
new type — it did not need to be. Confirmed by
`test_is_active_directional_attack_inactive_before_start_round`.

## 4. Only the designated malicious client is transformed

**PASS.** `update_transform_callback` in
`src/experiments/run_federated_cifar100.py` gates the entire attack
dispatch (both `gaussian_update_noise` and `norm_preserving_directional`
branches) behind a single `if attack_config.is_active(round_id, client_id):`
check, identical in structure to the pre-existing Gaussian-only version.
Confirmed end-to-end by
`test_one_tiny_end_to_end_federated_run_with_directional_attack`, which
asserts every honest client's directional-verification fields stay
`None` across all three rounds while only the designated malicious
client's fields are populated on its active round.

## 5. Honest local training completes before attack transformation

**PASS.** The injection point is unchanged from `EXP_014`
(`src/federated/runner.py`'s `update_transform_callback` hook, invoked
after `client.get_model_weights()`, before `ClientUpdate.create()`) — no
change was made to `runner.py` at all for this task. This property was
already independently verified for that exact mechanism in
`analysis/PHASE5_Implementation_Verification.md` item 5; since the
directional attack reuses the identical hook with no architectural
change, that verification transfers directly.

## 6. Server aggregation remains FedAvg and is not bypassed

**PASS.** `src/federated/aggregation/fedavg.py` was not touched;
`run_federated_cifar100.py` still constructs `strategy = FedAvg()`
unconditionally, regardless of `attack_type`. No new code path calls
`server.load_global_weights()` outside the runner's own existing,
unmodified aggregation step.

## 7. Directional attack modifies trainable parameters only

**PASS.** `apply_norm_preserving_directional_attack` computes `delta`
and writes `perturbed[key]` only for `key in keys` where
`keys = list(trainable_keys)`; every other key is carried through via
the initial `perturbed = {key: value.detach().clone() for key, value in honest_model_state.items()}`
and never subsequently touched. Confirmed by
`test_apply_directional_attack_perturbs_only_trainable_keys`.

## 8. Non-trainable buffers remain identical to the honest client state

**PASS.** Same test as item 7 asserts `torch.equal(perturbed["bn.running_mean"], honest_state["bn.running_mean"])`
and the equivalent for `bn.num_batches_tracked` — exact equality, not
approximate.

## 9. Delta is correctly defined as: honest local model − global model

**PASS.** `apply_norm_preserving_directional_attack`:
`delta = {key: honest_model_state[key] - global_state[key] for key in keys}`
— matches the task's definition exactly, over trainable keys only.

## 10. Norm preservation is mathematically correct

**PASS.** `direction_attack = c·delta_hat + s·u_perp` where `delta_hat`
and `u_perp` are constructed as an orthonormal pair (`u_perp` is
Gram-Schmidt-orthogonalized against `delta_hat`, then normalized to unit
length) and `s = sqrt(1 - c²)`. For any orthonormal pair,
`‖c·a + s·b‖² = c²‖a‖² + s²‖b‖² + 2cs⟨a,b⟩ = c² + s² = 1` (since
`⟨a,b⟩=0`, `‖a‖=‖b‖=1`), so `direction_attack` is an exact unit vector,
and `delta_attack = ‖delta‖·direction_attack` has `‖delta_attack‖ = ‖delta‖`
exactly, up to floating-point rounding. This is a closed-form identity,
not an empirical approximation.

## 11. `‖delta_attack‖ / ‖delta_honest‖ ≈ 1`

**PASS.** Verified numerically by
`test_directional_perturbation_preserves_norm_within_tolerance` (ratio
within `1e-4` absolute of `1.0`, across seven `target_cosine` values from
`-1.0` to `1.0`) and `test_apply_directional_attack_norm_preserved_end_to_end`
(within `1e-4` relative), plus the end-to-end test's
`delta_attack_norm_ratio == pytest.approx(1.0, abs=1e-3)` assertion using
the real, fully-wired attack path (not just the pure function in
isolation).

## 12. `cosine(delta_attack, delta_honest) ≈ configured target`

**PASS.** Verified by `test_directional_perturbation_realized_cosine_matches_target`
(measured cosine within `1e-3` absolute of the configured
`target_cosine`, across seven values) and the end-to-end test's
`realized_cosine_to_honest == pytest.approx(0.0, abs=1e-2)` assertion.

## 13. `target_cosine=1` behaves as the honest direction

**PASS.** Special-cased exactly:
`if target_cosine == 1.0: direction = delta_hat` — no RNG draw, no
approximation. Verified by
`test_directional_perturbation_target_cosine_one_reproduces_honest_update`
(`torch.allclose(..., atol=1e-5)`).

## 14. `target_cosine=0` yields an orthogonal direction

**PASS.** Goes through the general rotation path with `c=0`, so
`direction = 0·delta_hat + 1·u_perp = u_perp`, which is orthogonal to
`delta_hat` by construction (the explicit Gram-Schmidt projection-removal
step). Verified by `test_directional_perturbation_target_cosine_zero_is_orthogonal`
(realized cosine within `1e-3` of `0.0`).

## 15. `target_cosine=-1` yields the reversed direction

**PASS.** Special-cased exactly:
`elif target_cosine == -1.0: direction = {key: -tensor for key, tensor in delta_hat.items()}`.
Verified by `test_directional_perturbation_target_cosine_negative_one_is_exact_reversal`
(`torch.allclose(perturbation, -delta, atol=1e-5)`).

## 16. Deterministic seeding uses seed + round_id + client_id + attack type; does not alter the Gaussian attack's sequence

**PASS.** `_derive_directional_attack_seed(seed, round_id, client_id)`
hashes `f"{seed}-{round_id}-{client_id}-norm_preserving_directional"` —
all three required identifiers plus a literal attack-type discriminator
folded into the same key. This is a **separate function** from
`_derive_attack_seed` (the Gaussian attack's own, completely untouched
seed derivation) — not a shared function with an added parameter — so
the Gaussian attack's derived seeds, and therefore its noise draws (which
determined `EXP_014`'s already-recorded results), are provably unaffected
by construction, not merely by inspection. One implementation-detail
note: the attack-type identifier is a hardcoded literal string inside
`_derive_directional_attack_seed`, not an explicit function parameter —
functionally correct for the one attack type it serves, but see item 28.

## 17. Directional attack does not mutate global torch RNG state

**PASS.** `torch.Generator(device="cpu")` is a local, independent
generator instance, seeded via `.manual_seed(...)` on that instance only
— by PyTorch's own API contract, an explicit `generator=` argument to
`torch.randn(...)` never reads or advances the global default generator.
Verified directly (not just architecturally) by
`test_directional_perturbation_does_not_touch_global_rng_state`, which
checks both `torch.get_rng_state()` before/after equality and that a
subsequent `torch.randn(4)` draw is bit-identical whether or not a
directional-attack call intervened.

## 18. Zero-norm honest delta is handled safely

**PASS.** `if delta_norm == 0: return {key: torch.zeros_like(tensor) for key, tensor in delta.items()}`
— no division, no RNG draw, returns before any of the geometry code
executes. Verified by `test_directional_perturbation_handles_zero_delta_safely`.

## 19. Degenerate Gram-Schmidt behavior is fail-fast and clear

**PASS WITH LIMITATION.** By code inspection, the implementation is
correct: `if u_raw_norm == 0: raise RuntimeError("Cannot construct directional attack perturbation: ...")`
— an immediate, clearly-worded failure rather than silently proceeding
with an undefined or zero-length `u_perp`.

**Limitation:** no test directly forces this branch to execute (it is a
near-zero-probability event for continuous random draws, not
practically triggerable without contriving a specific seed/delta pair
known in advance to produce an exactly-parallel draw). This mirrors a
**pre-existing** gap already present in the Gaussian attack's own
analogous check: `generate_gaussian_attack_noise`'s
`if raw_noise_norm == 0: raise RuntimeError(...)` is likewise never
directly exercised by any existing test (the one Gaussian test named
similarly, `test_generate_gaussian_attack_noise_rejects_zero_delta`,
actually exercises the *other*, `delta_norm == 0` check). This is not a
regression introduced by this work — it is an inherited pattern.

## 20. `directional_target_cosine` validation and backward-compatible defaults

**PASS.** At the `AttackConfig` level: range-validated in `[-1, 1]`
whenever set (`test_attack_config_rejects_out_of_range_directional_target_cosine`),
required when `attack_type == "norm_preserving_directional"`
(`test_attack_config_requires_directional_target_cosine_for_directional_attack`),
inert (accepted, unused) for other types
(`test_directional_target_cosine_is_inert_for_gaussian_and_none`). At the
`FederatedExperimentConfig`/YAML level: defaults to `None` via
`raw.get("directional_target_cosine")`, so `EXP_004`'s (and every other
pre-existing) config loads unchanged
(`test_old_configs_load_with_backward_compatible_directional_target_cosine_default`),
the already-run `EXP_014` Gaussian config is confirmed unaffected
(`test_gaussian_config_directional_target_cosine_stays_none`), and the
same range/required validation is duplicated at this layer with matching
tests.

## 21. `delta_attack_norm_ratio` logging is correct

**PASS.** `build_trace_row` computes
`directional_norm_ratio = compute_norm_ratio(client_delta, honest_delta)`
only when `successful and is_directional_active_row and client_delta is not None and honest_delta is not None`
— `compute_norm_ratio` itself is `‖submitted‖/‖honest‖`, matching the
required definition exactly. Verified at the unit level
(`test_build_trace_row_directional_fields_only_populated_on_active_directional_rows`,
`test_compute_norm_ratio_matches_hand_computation`) and end-to-end
(`malicious_row["delta_attack_norm_ratio"] == pytest.approx(1.0, abs=1e-3)`
using the real attack function's output, not a hand-constructed pair).

## 22. `realized_cosine_to_honest` logging is correct

**PASS.** Same gating logic as item 21, using
`compute_cosine_similarity(client_delta, honest_delta, trainable_keys)`.
Verified unit-level and end-to-end, symmetric to item 21's evidence.

## 23. `self_prev_round_cosine` semantics

**PASS**, all five required sub-properties verified:

- **`None` on first observation:** `previous_round_deltas.get(client_id)`
  returns `None` until that client has a recorded prior successful round;
  verified both at the unit level
  (`test_build_trace_row_self_prev_round_cosine_none_without_history`)
  and end-to-end (round-0 assertion in the full federated-run test).
- **Uses only the same client's prior submitted update:** `self_prev_delta`
  is looked up strictly by `previous_round_deltas.get(client_id)` — no
  cross-client key is ever read at this line.
- **Updated only after the current round's value is computed:** in
  `round_callback`, `previous_round_deltas[client_id] = current_round_deltas[client_id]`
  executes *after* `trace_rows.append(build_trace_row(..., self_prev_delta=self_prev_delta))`
  for that same client, within the same loop iteration — the row is
  always built from the *old* value before it is overwritten.
- **Independent across clients:** `previous_round_deltas` is a
  `dict[str, ...]` keyed by `client_id`; no shared/global state couples
  one client's history to another's.
- **No future leakage:** `round_callback` fires once per round, strictly
  in the sequential order `FederatedRunner.run()` drives (`run_round(0)`,
  `run_round(1)`, ...); `previous_round_deltas` can therefore only ever
  contain data from rounds strictly before the one currently being
  processed.

## 24. Previous-round memory behavior when a client is not selected or fails

**PASS WITH LIMITATION.** By code inspection, the behavior is correct
and matches the documented design: `if client_id in current_round_deltas: previous_round_deltas[client_id] = current_round_deltas[client_id]`
— the memory entry is updated *only* on a genuinely successful round for
that client; if a client is not selected or fails, its entry is left
**unchanged** (stale, pointing to its last successful round), never reset
to `None` and never backfilled from a later round.

**Limitation:** this specific behavior is not exercised by any test. The
end-to-end test uses `clients_per_round=3, minimum_updates=3` (all three
clients always selected, no failure injected), so the carry-forward path
is implemented and documented but currently untested.

## 25. All existing `EXP_014` trace fields remain present and semantically unchanged

**PASS.** `TRACE_CSV_FIELDNAMES` retains all 14 original field names, in
their original order, with the four new fields appended at the end (not
interleaved). The computation of every original field
(`update_l2_norm`, `peer_mean_cosine_similarity`,
`attack_configured`/`attack_active`/`attack_type`/`attack_strength_rho`/
`malicious_client_id`/`attack_start_round`) inside `build_trace_row` is
untouched — the function's new parameters (`honest_delta`,
`self_prev_delta`) are additive, optional, and default to `None`, so
every pre-existing caller/test that omits them observes identical
behavior for the original fields.

## 26. No test-set data enters attack generation or trace diagnostics

**PASS.** Every new function (`generate_directional_attack_perturbation`,
`apply_norm_preserving_directional_attack`, `compute_cosine_similarity`,
`compute_norm_ratio`) takes only `Mapping[str, torch.Tensor]`/scalar
arguments — no dataset, dataloader, or test-set reference anywhere. The
existing architectural guard test,
`test_traceability_module_has_no_dataset_or_test_set_dependency` (source
inspection for `"cifar100"`, `"test_dataset"`, `"test_loader"`,
`"Dataset"`, `"DataLoader"`), still runs over the entire module including
all newly added code, and none of those forbidden substrings appear in
the new code (confirmed by direct reading).

## 27. Existing FedAvg, FedProx, and Gaussian experiments remain behaviorally compatible

**PASS.** `src/federated/aggregation/fedavg.py`, `src/federated/client.py`,
and `src/training/engine.py` (FedProx's home) are untouched. Every new
config field defaults to a value that is a no-op for `attack_type="none"`
or `"gaussian_update_noise"`. One structural note, not a compatibility
break: `current_round_honest_deltas[client_id] = {...}` is now computed
**unconditionally**, for every client, every round, regardless of
`attack_type` (added at the top of `update_transform_callback`, before
the attack-type dispatch). This is a small, fixed, side-effect-free
addition (one tensor subtraction per client per round, stored only in
per-round scratch state that is cleared every round) — it cannot alter
any submitted update, any aggregation result, or any existing output file
(`round_history.csv`, `summary.json`, checkpoints); it only feeds the new
trace-CSV columns and adds a small constant compute overhead to every
run, attack-configured or not.

## 28. Implementation limitations that are NOT correctness bugs

**PASS WITH LIMITATION** (the audit's own deliverable for this item —
several genuine, non-blocking limitations were found):

- **Misplaced section-header comment.** In
  `tests/test_attacks_traceability.py`, the
  `# Architectural guarantee: no dataset / test-set dependency anywhere`
  comment header now sits directly above the newly-inserted `EXP_015`
  test section rather than above
  `test_traceability_module_has_no_dataset_or_test_set_dependency` (which
  moved to the end of the file). Purely cosmetic — every test still
  executes and asserts correctly regardless of comment placement — but a
  genuine documentation/organization inconsistency worth a future cleanup
  pass.
- **Untested degenerate-orthogonality failure path** (item 19) and
  **untested previous-round carry-forward-on-failure path** (item 24) —
  both correctly implemented by inspection, neither directly covered by
  a test; the first mirrors a pre-existing gap in the Gaussian attack's
  own equivalent check, the second is new to this work.
- **`_derive_directional_attack_seed` hardcodes its attack-type
  discriminator as a literal string** rather than accepting it as a
  parameter. Correct and sufficient for exactly one directional attack
  type; a third attack type would need either another near-duplicate
  function or a refactor to a shared, parameterized seed-derivation
  helper — a forward-looking extensibility note, not a present defect.
- **No dedicated `tests/test_runner.py` case targets the directional
  attack specifically.** Coverage instead comes from pure-function unit
  tests (`tests/test_attacks_traceability.py`) plus one comprehensive
  end-to-end test (`tests/test_federated_experiment.py`) that exercises
  the real `FederatedRunner`/`update_transform_callback` integration.
  This is a reasonable scope choice (the generic hook mechanism itself
  was already fully tested for the Gaussian attack and needed no
  attack-specific re-verification at the runner level) rather than an
  oversight, but is worth naming explicitly.
- **The always-on `current_round_honest_deltas` computation** (item 27)
  is a small, permanent addition to every run's per-round cost,
  regardless of whether any attack is configured — harmless to results,
  but not gated behind `attack_configured` the way it plausibly could
  have been for a marginal efficiency saving on attack-free runs.

---

## Summary

| # | Item | Verdict |
|---|---|---|
| 1 | `attack_type=none` remains inert | PASS |
| 2 | `gaussian_update_noise` unchanged | PASS |
| 3 | Directional attack inactive before start round | PASS |
| 4 | Only malicious client transformed | PASS |
| 5 | Honest training completes before transformation | PASS |
| 6 | Aggregation remains FedAvg, not bypassed | PASS |
| 7 | Only trainable parameters modified | PASS |
| 8 | Non-trainable buffers unchanged | PASS |
| 9 | Delta correctly defined | PASS |
| 10 | Norm preservation mathematically correct | PASS |
| 11 | `‖delta_attack‖/‖delta_honest‖ ≈ 1` | PASS |
| 12 | Realized cosine ≈ configured target | PASS |
| 13 | `target_cosine=1` → honest direction | PASS |
| 14 | `target_cosine=0` → orthogonal | PASS |
| 15 | `target_cosine=-1` → reversed | PASS |
| 16 | Seeding uses seed+round+client+type; Gaussian sequence untouched | PASS |
| 17 | Global torch RNG state unmutated | PASS |
| 18 | Zero-norm delta handled safely | PASS |
| 19 | Degenerate Gram-Schmidt fail-fast/clear | PASS WITH LIMITATION |
| 20 | `directional_target_cosine` validation/defaults | PASS |
| 21 | `delta_attack_norm_ratio` logging correct | PASS |
| 22 | `realized_cosine_to_honest` logging correct | PASS |
| 23 | `self_prev_round_cosine` semantics (5 sub-properties) | PASS |
| 24 | Previous-round memory on failure/non-selection | PASS WITH LIMITATION |
| 25 | Existing `EXP_014` trace fields preserved | PASS |
| 26 | No test-set data in attack/diagnostics | PASS |
| 27 | FedAvg/FedProx/Gaussian behaviorally compatible | PASS |
| 28 | Non-bug implementation limitations identified | PASS WITH LIMITATION |

**No FAIL findings.** All three "PASS WITH LIMITATION" items reflect
correctly-implemented behavior with either an untested-but-inspected
edge case or a cosmetic/organizational note — none is a functional
defect, an incorrect computation, or a violation of the `EXP_015` plan's
stated requirements.
