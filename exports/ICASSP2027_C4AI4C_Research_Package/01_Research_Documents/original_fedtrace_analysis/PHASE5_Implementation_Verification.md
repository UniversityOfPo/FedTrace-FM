# Phase 5 Implementation Verification — Attack/Trace Pilot Infrastructure

**Audit date:** 2026-08-07
**Scope:** A complete implementation audit of the Phase 5 malicious-client
attack pilot infrastructure built against
`research_plans/PHASE5_Malicious_Client_Traceability_Plan.md`. This
document verifies implementation **correctness only** — no algorithmic
improvements are proposed, no `EXP_014` config was created, no experiment
was run, and no code was modified while producing it.

**Files audited:**
- `src/attacks/traceability.py`
- `src/federated/runner.py`
- `src/experiments/run_federated_cifar100.py`
- `tests/test_attacks_traceability.py`
- `tests/test_runner.py`
- `tests/test_federated_experiment.py`
- `research_plans/PHASE5_Malicious_Client_Traceability_Plan.md`

**Method:** Every checklist item below is verified against specific code
locations and, where one exists, the test that exercises the property.
Line numbers refer to the files as they exist on disk at audit time.

---

## 1. Is the attack completely inactive for all previous experiments?

**PASS.**

`FederatedExperimentConfig`'s four new fields default to
`attack_type="none"`, `malicious_client_id=None`, `attack_start_round=None`,
`attack_strength_rho=0.0` (`run_federated_cifar100.py:173-176`).
`AttackConfig.is_active()` (`traceability.py:101-114`) returns `False`
unconditionally whenever `attack_type != "none"` is false — the first term
of its `and`-chain. `update_transform_callback` (`run_federated_cifar100.py:741-777`)
checks `attack_config.is_active(round_id, client_id)` and falls through to
`submitted_model_state = honest_model_state` (line 772) — the identical
object, not a modified copy — whenever it is `False`.

Every existing config (`EXP_002`–`EXP_013`) predates these four fields and
therefore parses them via the `raw.get(..., default)` fallback paths at
`run_federated_cifar100.py:239-246`, landing on the fully-inert defaults.
`test_old_configs_load_with_backward_compatible_attack_defaults`
(`tests/test_federated_experiment.py`) confirms `EXP_004`'s config loads
with `attack_type="none"` and all three other fields at their inert
defaults.

## 2. Does attack_type=none recover identical behavior?

**PASS.**

Two independent code paths confirm this: (a) `AttackConfig.is_active()`
short-circuits `False` for `attack_type == "none"` regardless of any other
field's value (`traceability.py:109-113`); (b) even considered in
isolation, `apply_gaussian_update_noise`/`generate_gaussian_attack_noise`
are simply never called when `is_active()` is `False` — the `else`
branch at `run_federated_cifar100.py:772` returns `honest_model_state`
untouched, not a cloned/recomputed equivalent. Since the returned object
is the exact input object, `ClientUpdate.create()` receives bit-identical
tensors to the pre-Phase-5 code path.
`test_update_transform_callback_none_matches_pre_existing_behavior`
(`tests/test_runner.py`) and `test_attack_config_defaults_are_inert`
(`tests/test_attacks_traceability.py`) both exercise this directly.

## 3. Can honest clients ever be modified accidentally?

**PASS.**

Three independent safeguards were verified:
1. **Exact-match gating.** `is_active()` requires
   `client_id == self.malicious_client_id` (`traceability.py:111`) — a
   plain string equality, not a prefix/substring/fuzzy match.
2. **Fail-fast on misconfiguration.** `main()` defensively checks
   `attack_config.malicious_client_id not in clients` and raises
   `ConfigError` before any round runs (`run_federated_cifar100.py:697-701`)
   — a typo'd or non-existent `malicious_client_id` is caught at startup,
   not silently ignored or (worse) silently matched against the wrong
   client.
3. **Per-client call isolation.** `update_transform_callback` is invoked
   once per client inside the runner's per-client loop
   (`runner.py:467-471`), receiving only that client's own
   `model_state` — there is no code path by which one client's callback
   invocation can read or write another client's data.

`test_update_transform_callback_receives_only_that_clients_own_state`
(`tests/test_runner.py`) directly confirms each client's own live model
weights match exactly what the hook observed for that client and no
other.

## 4. Is only the malicious client's submitted update modified?

**PASS.**

`is_active()`'s `client_id == self.malicious_client_id` condition
(`traceability.py:111`) is the sole gate. Verified by
`test_is_active_only_true_for_malicious_client`,
`test_update_transform_callback_perturbs_only_the_designated_client`, and
`test_honest_client_update_reaches_aggregation_unperturbed`, the last of
which confirms the **aggregated global model** matches the FedAvg formula
computed from client_0's perturbed report and client_1's genuinely
unperturbed one — an end-to-end proof, not just a unit-level one.

## 5. Is the attack injected after honest local training?

**PASS.**

`runner.py:456-467`: `client.load_global_weights(...)` →
`client.train_one_round()` (this call is honest local SGD, completely
unmodified — `FederatedClient`/`src/training/engine.py` were not touched
by this implementation) → `submitted_model_state = client.get_model_weights()`
(captures the fully-trained state) → **only then** is
`self.update_transform_callback` invoked (`runner.py:468-471`). The
transform never runs before or during `train_one_round()`.

## 6. Is the server global model untouched (only changed via normal aggregation)?

**PASS.**

The only call to `self.server.load_global_weights(...)` anywhere in
`runner.py` is at line 572, inside `run_round()`'s "Step 8," using
`aggregation_result.aggregated_state` produced by
`self.aggregation_strategy.aggregate(pre_round_global_state, received_updates)`
(line 543-545) — the same unmodified `FedAvg.aggregate()` every prior
experiment uses. `run_federated_cifar100.py`'s `main()` never calls
`server.load_global_weights()` itself (only the read-only
`server.get_global_weights()` and `server.register_client()`). The attack
transform has no reference to `self.server` at all — it is a pure
function of `(client_id, round_id, model_state, global_state)`.

## 7. Is attack deterministic?

**PASS.**

`generate_gaussian_attack_noise` seeds a dedicated
`torch.Generator(device="cpu")` via `_derive_attack_seed(seed, round_id, client_id)`
(`traceability.py:234-236`), a pure `sha256`-based hash with no external
state dependency (`traceability.py:117-128`). `test_generate_gaussian_attack_noise_is_deterministic`
confirms identical inputs produce byte-identical noise across two
separate calls.

## 8. Is RNG isolated (does not consume or depend on global RNG state)?

**PASS WITH LIMITATION.**

The implementation uses `torch.Generator(device="cpu").manual_seed(...)`
(`traceability.py:234-236`) and passes it explicitly as
`generator=generator` to every `torch.randn(...)` call
(`traceability.py:244-246`) — by PyTorch's own API contract, a tensor
constructor called with an explicit `generator=` argument reads from
*only* that generator object and never touches (reads or advances) the
global default RNG. This is an architectural guarantee, not a
probabilistic one.

**Limitation:** no test in `tests/test_attacks_traceability.py` or
elsewhere explicitly asserts that calling
`generate_gaussian_attack_noise`/`apply_gaussian_update_noise` leaves
`torch.get_rng_state()` unchanged, or that a subsequent honest client's
`DataLoader` shuffling (which does consume the global RNG) is unaffected
by an attack call earlier in the same round. The property holds by
construction today, but there is no regression test that would catch a
future edit accidentally switching to `torch.manual_seed(...)` (which
*would* mutate global state) instead of a local `Generator`.

## 9. Is attack strength mathematically correct?

**PASS.**

`scale = (rho * delta_norm) / raw_noise_norm` (`traceability.py:256`),
applied uniformly to every element of the raw noise
(`traceability.py:257-259`). Since
`‖scale · raw_noise‖₂ = scale · ‖raw_noise‖₂ = rho · delta_norm` exactly
(scalar multiplication scales L2 norm linearly), the returned noise's own
norm equals `rho · ‖delta‖₂` up to floating-point rounding from the
division/multiplication and the final dtype cast.
`test_generate_gaussian_attack_noise_matches_target_norm` and
`test_apply_gaussian_update_noise_matches_target_norm_relative_to_delta`
verify this numerically (`rel=1e-5` tolerance) against hand-computable
cases.

## 10. Is rho=0 exactly equivalent to no attack?

**PASS.**

`generate_gaussian_attack_noise` short-circuits at `traceability.py:231-232`:
`if rho == 0: return {key: torch.zeros_like(tensor) ...}` — exact zero
tensors, no RNG draw at all. `honest_model_state[key] + 0.0 == honest_model_state[key]`
holds exactly under IEEE 754 for any finite value, so
`apply_gaussian_update_noise`'s output is bit-identical to the honest
input. `test_apply_gaussian_update_noise_rho_zero_is_byte_identical_to_honest`
confirms this via `torch.equal` (exact equality, not `allclose`).

Note: this code path is exercised only at the unit-test level — the real
config validation (`run_federated_cifar100.py:355-359`) rejects
`attack_strength_rho <= 0` when `attack_type="gaussian_update_noise"`, so
`rho=0` is unreachable through a real experiment config for that attack
type. The equivalent "no attack" state is reached, in a real run, via
`attack_type="none"` instead (item 2). Both paths were verified
independently.

## 11. Are BatchNorm buffers preserved?

**PASS.**

`trainable_keys` is built from `model_fn().named_parameters()`
(`run_federated_cifar100.py:702`) — `named_parameters()` yields only
`nn.Parameter` tensors (conv/linear weights and biases, and BatchNorm's
own affine `weight`/`bias`), never registered buffers such as
`running_mean`, `running_var`, or `num_batches_tracked`. Inside
`apply_gaussian_update_noise`, `perturbed` is first seeded as a full clone
of **every** key in `honest_model_state` (`traceability.py:287`), and only
the `trainable_keys` subset is subsequently overwritten
(`traceability.py:288-289`). `test_apply_gaussian_update_noise_perturbs_only_trainable_keys`
directly verifies `bn.running_mean` and `bn.num_batches_tracked` are
`torch.equal` before/after the transform while `layer.weight` changes.

## 12. Are trainable tensors only modified?

**PASS.**

The converse of item 11: `delta`, `noise`, and the final overwrite loop in
`apply_gaussian_update_noise` all iterate over exactly `keys = list(trainable_keys)`
(`traceability.py:281-289`) — no other key is ever written. Since
`trainable_keys` is derived from the same model's `named_parameters()`
that produced `get_model_weights()`'s `state_dict()`, key names align
exactly (no `KeyError` risk, no silently-skipped trainable key).
BatchNorm's own trainable affine parameters (`bn.weight`/`bn.bias`,
distinct from its buffers) are correctly included, matching the plan's
Section 5 scope.

## 13. Does logging leak future information?

**PASS WITH LIMITATION.**

The core question — does the **naive detector's threshold computation**
use future (post-attack-window) information — is **no**:
`apply_naive_detector` filters `pre_attack_values` strictly by
`row["round_id"] < attack_start_round` (`traceability.py:380-384`) before
computing `mean_pre`/`std_pre`, and the detector never reads the
ground-truth `attack_active`/`malicious_client_id` columns to decide what
to flag — it is blind to ground truth by construction, using only
`peer_mean_cosine_similarity`. `test_naive_detector_uses_only_pre_attack_window`
directly proves this (two row sets differing only in post-attack values
produce identical thresholds).

Diagnostic features themselves (`update_l2_norm`,
`peer_mean_cosine_similarity`) are computed from that round's own data
only — no forward-looking computation there either.

**Limitations found:**
- `attack_active` (`traceability.py:325`, per row) is a pure function of
  `(round_id, client_id)` and does **not** account for whether the client
  was actually `selected` that round. In a configuration where
  `clients_per_round < num_clients` (not the case for the currently
  planned `EXP_014` pilot, where all 5 clients are selected every round),
  a row could show `attack_active=True` while `selected=False` and no
  perturbation actually occurred that round. This is fully recoverable by
  cross-referencing the `selected` column, but the field name alone
  invites misreading as "an attack occurred this round" rather than "the
  attack policy is active this round."
- `first_flagged_round` is back-filled onto **every** row for a client,
  including rows from **before** `attack_start_round`
  (`traceability.py:406-407`) — i.e., a pre-attack row for a client can
  carry a `first_flagged_round` value referring to a round that, relative
  to that row, is in the future. This is intentional (a per-client
  summary constant, documented in the function's own docstring) but is a
  form of "future information" appearing in a row that otherwise looks
  like a per-round record. See item 16.

## 14. Does the naive detector use only pre-attack rounds?

**PASS.**

Confirmed directly in code (`traceability.py:380-384`,
`if attack_start_round is None` guard at `traceability.py:368-373`) and
by `test_naive_detector_uses_only_pre_attack_window`.

## 15. Is the threshold computed independently per client?

**PASS.**

`rows_by_client` groups rows by `client_id` before any statistics are
computed (`traceability.py:375-377`), and the `mean_pre`/`std_pre`/
`threshold` computation runs inside a loop over `rows_by_client.values()`
(`traceability.py:379-391`) — no shared state or cross-client aggregation
exists in this computation. `test_naive_detector_threshold_is_per_client`
confirms two clients with different pre-attack distributions receive
different, correctly-computed thresholds.

## 16. Is first_flagged_round computed correctly?

**PASS WITH LIMITATION.**

The earliest-round-flagged logic itself is correct: rows are processed in
ascending `round_id` order (`sorted(client_rows, key=lambda r: r["round_id"])`,
`traceability.py:394`), and `first_flagged_round` is set exactly once, on
the first row where `flagged` becomes `True`
(`traceability.py:403-404`). `test_naive_detector_first_flagged_round_is_earliest_flag`
verifies this against a hand-constructed case with a known expected
answer.

**Limitation:** as noted in item 13, the computed value is then written
onto **every** row for that client (`traceability.py:406-407`), including
rounds before `attack_start_round`. A consumer reading the CSV row by row
(rather than understanding it as a per-client summary constant) could
mistake this for a live, round-by-round "detected so far" signal, which
it is not — it is the same, whole-run-computed value repeated on every
row for that client, decided only once all rounds are known.

## 17. Is client_round_trace.csv sufficient for future FedTrace?

**PASS WITH LIMITATION.**

The artifact is complete and correct **for its stated, narrower purpose**:
evaluating the naive baseline detector against ground truth (every field
the naive detector consumes or produces is present:
`peer_mean_cosine_similarity`, `naive_threshold`, `naive_flag`,
`first_flagged_round`, plus the ground-truth labels needed to score it).

It is **explicitly not**, and was never intended to be, a complete
feature set for the eventual FedTrace algorithm — the plan's Section 21
boundary and Section 10's explicit "cheap baseline vs. future FedTrace
features" split both anticipate this. Specifically absent, by design: any
layer-wise breakdown, any temporal/windowed feature, any
validation-impact feature, and (per item 18 below) several data points
that were available in the pipeline but not threaded into this artifact.

## 18. What important information is still NOT being logged?

Descriptive finding, not a pass/fail defect — this documents gaps against
what a future analysis phase would plausibly want:

- **Per-client train loss/top-1/top-5.** `client_train_metrics` is
  computed inside `runner.py`'s per-client loop (lines 460-464) but is
  never passed to `update_transform_callback` or otherwise threaded into
  `client_round_trace.csv` — only the round-level, sample-weighted
  aggregate appears in `round_history.csv`. A malicious client's own
  local training loss (which, per FedAvg unmodified local training, is
  genuinely honest even under an attack that only perturbs the *reported*
  update) is not directly correlatable with its trace row today.
- **Raw per-client update tensors.** Only two scalar summaries
  (`update_l2_norm`, `peer_mean_cosine_similarity`) are persisted per
  client-round; the actual update tensors are not saved anywhere
  associated with the trace (checkpoints only persist the *aggregated*
  global model and full per-client optimizer state, not the individual
  per-round submitted deltas). Re-deriving a richer feature (e.g.
  layer-wise norms) after the fact would require re-running the
  experiment, not re-processing saved artifacts.
- **No provenance identifiers inside the CSV itself.** `client_round_trace.csv`
  carries no `experiment_id`/`seed`/`git_commit` column; that information
  exists only in the sibling `summary.json`. A copy of the trace CSV
  separated from its directory would be unlabeled.
- **No per-round wall-clock timing at the client-row level** (only
  round-level `round_seconds` in `round_history.csv`).

## 19. Are there architectural risks for future attack families?

**PASS WITH LIMITATION** — the current design is sound for the scope it
was built for, with identifiable constraints for scope beyond it:

- **`update_transform_callback`'s signature is single-client-scoped by
  design** (`traceability.py`'s `UpdateTransformCallback`,
  `runner.py:219-246`): each invocation sees only one client's own
  update. This is correct and sufficient for the plan's single-attacker
  threat model (Section 3), but a future **colluding multi-attacker**
  study, where attackers coordinate using knowledge of each other's
  updates within the same round, cannot be expressed through this hook as
  currently defined without extending its signature or introducing a
  round-level (not per-client) hook.
- **`AttackConfig` has singular, not plural, attack-target fields**
  (`malicious_client_id: str | None`, one attack type, one `rho`).
  Supporting multiple simultaneous malicious clients, or mixing attack
  types across clients in one run, would require restructuring
  `AttackConfig` (e.g. to a list of per-client attack specs), not just
  adding a config field.
- **The injection point is specific to model/update-poisoning attacks.**
  Because `update_transform_callback` fires strictly after
  `client.train_one_round()` completes, it structurally cannot express a
  genuine **data-poisoning** attack (e.g. label flipping, the alternative
  the plan's Section 4 explicitly considered and rejected for this first
  study) — such an attack would need to alter the client's *training
  data* before/during `train_one_round()`, an entirely different
  injection point (inside `FederatedClient`/its dataset), not this hook.
  This is a correct match to the plan's chosen scope, not a defect, but
  it means the current infrastructure does not generalize to that attack
  family without separate work.
- **Attack strength is a single global scalar (`rho`)**, applied
  uniformly across all trainable parameters via one global norm-scaling
  factor. A future attack family wanting layer-specific or
  parameter-group-specific strength would need `AttackConfig`/
  `generate_gaussian_attack_noise` extended to accept a structured
  strength specification, not a single float.
- **The deterministic-seed derivation (`_derive_attack_seed`) is
  attack-type-agnostic** and reusable as-is by future attack families
  needing per-`(seed, round, client)` randomness (e.g. selecting a
  backdoor trigger pattern) — this part of the design generalizes well.

## 20. Does the implementation exactly match the Phase-5 plan?

**PASS WITH LIMITATION.**

Matches confirmed against the plan document:
- §4 (attack family: `gaussian_update_noise`) — implemented exactly, and
  no other attack type exists in `_VALID_ATTACK_TYPES`.
- §5 (strength parameterization, `‖noise‖₂ = rho · ‖delta‖₂`, trainable
  parameters only) — implemented exactly (items 9, 11, 12 above).
- §8 (injection point: after honest local training, before server
  aggregation, no global-model or honest-client contamination) —
  implemented exactly (items 3, 5, 6 above).
- §9 (ground-truth fields: selected, malicious client id, attack
  active/type/strength/start round) — implemented as a superset (adds
  `attack_configured`) of the plan's required field list.
- §11 (naive detector: peer-agreement cosine feature, per-client
  pre-attack-only 3-sigma threshold, outputs suspicious client and first
  flagged round) — implemented exactly.
- §21 (explicit boundary: no FedTrace algorithm) — respected; no code in
  `src/attribution/` and no learned/temporal/layer-wise detection logic
  exists anywhere in this implementation.

**Deliberate, documented narrowing relative to the plan:** the plan's
Section 10 lists a broader *candidate* set of cheap diagnostic signals
("update L2 norm, cosine similarity to previous/global/reference
direction, distance from robust/peer center, layer-wise norms if
justified, validation impact if it does not leak test data") introduced
with "such as," not as a mandatory complete list, and explicitly
instructs separating cheap signals from future FedTrace work. The actual
implementation task that authorized this code narrowed that candidate set
to exactly two signals (`update_l2_norm`,
`peer_mean_cosine_similarity`). This is consistent with the plan's own
minimality intent and its explicit task-level scoping instruction, not an
unauthorized deviation — but it means the implementation covers a proper
subset of the plan's Section 10 candidate list, which is why this item is
not an unqualified exact match.

Not applicable to a code audit (these are experiment-design decisions
deferred to the not-yet-created `EXP_014` config, correctly *not* baked
into the infrastructure): the specific choice of `client_002` (§6), round
20 (§7), and the seed-42-pilot-first strategy (§16).

---

## Summary

| # | Item | Verdict |
|---|---|---|
| 1 | Attack inactive for all previous experiments | PASS |
| 2 | attack_type=none recovers identical behavior | PASS |
| 3 | Honest clients cannot be modified accidentally | PASS |
| 4 | Only the malicious client's update is modified | PASS |
| 5 | Attack injected after honest local training | PASS |
| 6 | Server global model untouched (only via aggregation) | PASS |
| 7 | Attack is deterministic | PASS |
| 8 | RNG is isolated | PASS WITH LIMITATION |
| 9 | Attack strength mathematically correct | PASS |
| 10 | rho=0 exactly equivalent to no attack | PASS |
| 11 | BatchNorm buffers preserved | PASS |
| 12 | Trainable tensors only modified | PASS |
| 13 | Logging does not leak future information | PASS WITH LIMITATION |
| 14 | Naive detector uses only pre-attack rounds | PASS |
| 15 | Threshold computed independently per client | PASS |
| 16 | first_flagged_round computed correctly | PASS WITH LIMITATION |
| 17 | client_round_trace.csv sufficient for future FedTrace | PASS WITH LIMITATION |
| 18 | Important information not yet logged | PASS WITH LIMITATION (descriptive) |
| 19 | Architectural risks for future attack families | PASS WITH LIMITATION |
| 20 | Implementation exactly matches the Phase-5 plan | PASS WITH LIMITATION |

**No FAIL findings.** Every limitation identified is a documented design
choice, an untested-but-architecturally-guaranteed property, or an
intentional, plan-consistent scope boundary — none reflects a code
defect, an incorrect computation, or a violation of the plan's stated
requirements.
