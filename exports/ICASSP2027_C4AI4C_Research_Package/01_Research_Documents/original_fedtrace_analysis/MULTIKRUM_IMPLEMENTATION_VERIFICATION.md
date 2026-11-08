# Multi-Krum Implementation Verification (Read-Only Audit)

**Status:** Read-only audit document. No code, test, config, README, FedTrace, attack, or Research Database file was created or modified in producing it (this document itself is the only file written). No experiment was run. No Multi-Krum experiment config was created. No pytest or other command was executed — the `47 passed in 2.87s` result cited below is the PI's own manually-run output, reproduced here as evidence, not re-derived.

---

## 1. Files Inspected

- `research_plans/MULTIKRUM_EXTERNAL_BASELINE_IMPLEMENTATION_PLAN.md` (the frozen plan being audited against)
- `research_plans/FEDTRACE_EXTERNAL_BASELINE_SELECTION_PLAN.md`
- `src/federated/aggregation/multikrum.py`
- `src/federated/aggregation/fedavg.py`
- `src/federated/aggregation/base.py`
- `src/experiments/run_federated_cifar100.py`
- `tests/test_multikrum_aggregation.py`
- `tests/test_aggregation.py`
- `configs/cifar100_fedavg_dirichlet_a01_seed123_50rounds.yaml` (EXP_008 — honest FedAvg, seed 123)
- `configs/cifar100_fedavg_dirichlet_a01_seed2026_50rounds.yaml` (EXP_009 — honest FedAvg, seed 2026)
- `configs/cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed123.yaml` (EXP_016 — inspected only as the established "strict single-variable replication of EXP_008" template convention, to confirm how a paired config is correctly derived)
- Directory listings: `configs/*.yaml`, `analysis/EXP_01*` (to confirm the current experiment-numbering convention and that no `Research_Database` file exists in this repository to inspect)

## 2. Category-by-Category Verdicts

### A. Krum score — **PASS**

- Squared Euclidean distance: `MultiKrum._pairwise_squared_distances` computes `((v_i - v_j) ** 2).sum()` directly (no sqrt-then-square round trip), matching `d(i,j) = ||u_i - u_j||_2^2` exactly (`multikrum.py:263-286`).
- Flattened trainable-parameter deltas only: `_flatten_trainable_delta` concatenates `model_state[k] - global_state[k]` for `k in sorted(self._trainable_keys)` (`multikrum.py:240-261`); `trainable_keys` is injected at construction and, in the experiment script, is exactly `frozenset(name for name, _ in model_fn().named_parameters())` (`run_federated_cifar100.py:766`) — the same set the attack code and FedTrace's own `z_m` already use.
- Score uses `n-f-2` nearest peers: `neighbors_used = n - f - 2` (`multikrum.py:166`); `_krum_scores` sums the `neighbors_used` smallest peer distances (`multikrum.py:288-297`).
- No BatchNorm/non-trainable buffers enter the distance: the vectors fed to `_pairwise_squared_distances` are built exclusively from `sorted_trainable_keys`; buffers are never included. Verified against a differentiating adversarial test case (`test_non_trainable_floating_buffer_excluded_from_distance`) constructed so that including the buffer *would* have flipped the selection outcome — it does not.

### B. n/f/m — **PASS**

- Runtime `n = len(updates)` (`multikrum.py:156`).
- `f=1` in experiment integration: `_MULTIKRUM_F: int = 1` (`run_federated_cifar100.py:91`), passed as `MultiKrum(trainable_keys=trainable_keys, f=_MULTIKRUM_F)` (`run_federated_cifar100.py:787`). Not a YAML field — confirmed `FederatedExperimentConfig` gained no `f`/`m` field.
- `m = n - f`, always derived, never independently settable (`multikrum.py:165`).
- `n >= 2f+3` validated; raises `ValueError` naming the exact bound when violated, with no fallback to `FedAvg` or any other strategy (`multikrum.py:158-164`).

### C. Selection — **PASS**

- The `m` lowest-scoring clients are selected: `ordered[:m]` after ascending sort (`multikrum.py:318-321`).
- Deterministic `(score, client_id)` tie-breaking: `sorted(scores.keys(), key=lambda cid: (scores[cid], cid))` (`multikrum.py:318`).
- No attacker identity or attack metadata used: `aggregate()` reads only `update.client_id`, `update.round_id` (via `validate_common_update_structure`), and `update.model_state`. `update.num_samples` and `update.metadata` (which could in principle carry attack-adjacent bookkeeping such as `local_epochs`) are never read by `MultiKrum` at all. Confirmed by direct code reading — no reference to `attack_config`, `malicious_client_id`, or any ground-truth field exists anywhere in `multikrum.py`.

### D. Aggregation — **PASS**

- Selected-client trainable parameters averaged unweighted: `_unweighted_average` sums and divides by `len(per_client_tensors)`, never touching `num_samples` (`multikrum.py:324-339`).
- Sample counts do not affect weighting: confirmed both by code inspection (no `num_samples` reference anywhere in the aggregation path) and by `test_sample_counts_do_not_influence_multikrum_output`, which uses a 1,000,000:1 sample-count skew and asserts an identical output to the equal-sample-count case.
- Non-trainable floating buffers averaged over the same selected set: the `reference_tensor.is_floating_point()` branch in `aggregate()` uses `selected_updates` for *every* floating key, trainable or not (`multikrum.py:191-203`) — not a second, independent selection.
- `.num_batches_tracked` and other non-floating buffers: `.num_batches_tracked`-suffixed keys use `FedAvg._max_across_clients` across **all** `n` participating updates (not just the `m` selected); every other non-floating key uses `FedAvg._require_identical`, also across all `n` updates (`multikrum.py:204-209`). This matches the frozen plan's Section 4 exactly ("applied across all participating clients", explicitly not restricted to the selected subset), and reuses `FedAvg`'s own methods directly rather than re-implementing them — no duplicated buffer-handling logic exists to drift out of sync.

### E. Diagnostics — **PASS**

- Per-client Krum score, rank, and selected/rejected status are present in `metadata["multikrum_diagnostics"][client_id]` as `krum_score`/`krum_rank`/`selected` (`multikrum.py:211-218`).
- Selected/rejected client-ID tuples: `metadata["selected_client_ids"]`/`metadata["rejected_client_ids"]` (`multikrum.py:226-227`).
- `n`/`f`/`m`/`neighbors_used`: all present at the top level of `metadata` (`multikrum.py:221-224`).
- Persisted separately to `multikrum_diagnostics.csv`: `round_callback` reads `result.metadata["aggregation_metadata"]["multikrum_diagnostics"]` (populated via the pre-existing, unmodified `AggregationResult.metadata` → `FederatedRunner` step-10 → `FederatedRoundResult.metadata["aggregation_metadata"]` pipeline — **no runner change was needed or made**), accumulates one row per `(round_id, client_id)`, and `save_multikrum_diagnostics_csv` writes them to `output_dir / "multikrum_diagnostics.csv"` only when non-empty (`run_federated_cifar100.py:924-948, 1045-1048`).
- `client_round_trace.csv`/FedTrace schema untouched: `src/attacks/traceability.py` was not modified in the implementation turn (confirmed no diff exists against it); `build_trace_row`, `apply_naive_detector`, and `save_client_round_trace_csv` are called with the exact same arguments as before Multi-Krum's addition. The two CSV artifacts (`client_round_trace.csv` and `multikrum_diagnostics.csv`) are written independently, by independent code paths, to independent files.
- **Diagnostics sufficiency for the pre-registered rejection-proxy analysis (plan §11):** sufficient. The plan's proxy requires, per client: first-anomalous-round (first `multikrum_rejected=True` row), persistent-rejection (`P_krum=3` consecutive rejected rounds), onset estimate (first round of that confirmed streak), and predicted client (earliest persistent rejection, tie-broken by rejection frequency then client_id). Every one of these is computable purely by filtering/grouping `multikrum_diagnostics.csv`'s per-`(round_id, client_id)` rows (`round_id`, `client_id`, `krum_score`, `krum_rank`, `multikrum_selected`, `multikrum_rejected`) — no additional field is missing, and the CSV contains a row for **every** participating client every round (selected and rejected alike), not only the rejected ones, which is what a consecutive-streak/tie-break computation needs.

### F. Backward compatibility — **PASS**

- `aggregation: fedavg` behavior unchanged: the dispatch branch instantiates `FedAvg()` identically to before (`run_federated_cifar100.py:784-785`); `fedavg.py` itself has zero diff.
- Existing configs remain valid: no new required or optional YAML field was added to `FederatedExperimentConfig` for Multi-Krum (`f`/`m` are deliberately not YAML fields, per the plan and per this task's explicit restriction); `_SUPPORTED_AGGREGATIONS` was only extended (`("fedavg",)` → `("fedavg", "multikrum")`), never narrowed or altered for the existing value. Every existing `aggregation: fedavg` config (EXP_002–EXP_019) continues to validate and dispatch exactly as before.
- Runner/server/base interfaces were not redesigned: `src/federated/runner.py`, `src/federated/server.py`, and `src/federated/aggregation/base.py` all have zero diff. Diagnostics reach `round_callback` entirely through the pre-existing, generic `AggregationResult.metadata` → `aggregation_metadata` pass-through that has existed since the runner was first built — no new parameter, hook, or interface method was added anywhere in the plugin architecture.

### G. Test evidence — **PASS, with one disclosed limitation**

- **PI-provided result, recorded verbatim:** `.venv\Scripts\python -m pytest tests/test_multikrum_aggregation.py tests/test_aggregation.py -v` → **47 passed in 2.87s, 0 failed, 0 errors.** This audit did not rerun this command; the result above is taken as given.
- **Material coverage of the frozen plan:** `tests/test_multikrum_aggregation.py`'s 27 tests, read in full, directly exercise every "choose exactly ONE" decision the plan made: the exact score formula and neighbor count, `n=5/f=1/m=4` derivation, feasibility boundary and violation, negative-`f`/empty-`trainable_keys` rejection, deterministic tie-breaking (using a constructed score-tie scenario), unweighted averaging and its independence from sample counts, buffer exclusion from distance (using a differentiating adversarial construction, not a coincidental pass), buffer averaging over the selected set, `.num_batches_tracked`-vs-other-buffer handling, structural-validation reuse (missing key, shape mismatch), determinism across repeated calls, and diagnostics-field completeness. `test_fedavg_behavior_unchanged_by_multikrum_addition` directly checks non-interference with `FedAvg`. This is a materially thorough unit-level audit of `MultiKrum.aggregate()` itself.
- **Disclosed limitation (non-blocking):** no automated test exercises the *integration* wiring added to `run_federated_cifar100.py` — specifically, the `config.aggregation == "multikrum"` dispatch branch, the `multikrum_diagnostics` extraction loop inside `round_callback`, and `save_multikrum_diagnostics_csv`. A repository-wide search confirms no existing test (before or after this addition) asserts on `result.metadata["aggregation_metadata"]` at all, for any strategy — this specific pass-through has never been directly unit-tested by name, only exercised as a side effect of running any round. This is a real, honest coverage gap, but a narrow and low-risk one: the *data* being extracted (`multikrum_diagnostics`'s contents) is independently verified correct at the unit level (above); only the CSV-writing glue code that reads it out of `FederatedRoundResult.metadata` and serializes it is unverified end-to-end. It is also a strict no-op for every `aggregation: fedavg` config (`multikrum_diagnostics is not None` guards the whole block), so it cannot regress any already-validated FedAvg run. **Recommendation, not an action taken here:** treat the first clean-control run itself as the integration check for this specific wiring — after it completes, visually confirm `multikrum_diagnostics.csv` exists, has one row per `(round, client)`, and its `krum_score`/`krum_rank`/`multikrum_selected` values are sane, before creating the second (seed-2026) clean-control config or any attacked config.

### H. Experiment-readiness — **A: clean-control config creation is authorized next**

No correctness or methodological blocker was found in A–F. The one item in G is a disclosed, non-blocking limitation whose natural verification point is precisely the first clean-control run the plan already requires next — it does not justify withholding that run. This matches the plan's own Section 7 requirement (clean Multi-Krum controls at seed 123 and seed 2026, run *before* any attacked comparison) and this task's expected sequence (seed 123 paired to EXP_008, then seed 2026 paired to EXP_009).

## 3. Implementation Discrepancies

**None found.** Every mechanism specified in `research_plans/MULTIKRUM_EXTERNAL_BASELINE_IMPLEMENTATION_PLAN.md` Sections 2–14 (score formula, `n`/`f`/`m`, update representation, weighting rule, buffer handling, diagnostics, architecture, test plan) is implemented exactly as specified, with no invented variant and no silent deviation.

## 4. Methodological Discrepancies

**None found relative to the frozen plan.** The one open item is the test-coverage limitation recorded under G above (an implementation-verification gap, not a methodological deviation from the plan's science) — it does not change what the plan committed to, only how thoroughly one glue-code path has been exercised so far.

## 5. Diagnostics Sufficiency for the Rejection-Proxy Analysis

**Sufficient**, as detailed under E above: `multikrum_diagnostics.csv`'s per-`(round_id, client_id)` rows contain everything the plan's Section 11 proxy definitions (first-anomalous-round, `P_krum=3` persistent rejection, onset estimate, predicted client with rejection-frequency/`client_id` tie-break) require, computable entirely offline from this one artifact, with no ground truth involved in producing it.

## 6. FedAvg Backward Compatibility

**Preserved**, as detailed under F above: `fedavg.py` has zero diff, the `aggregation: fedavg` dispatch path is behaviorally identical to before Multi-Krum's addition, no existing config's required/optional field set changed, and `test_fedavg_behavior_unchanged_by_multikrum_addition` directly confirms no cross-contamination from importing both strategies in the same process.

## 7. Test Result

**47 passed, 0 failed, 0 errors, 2.87s** (`tests/test_multikrum_aggregation.py` + `tests/test_aggregation.py`), as reported by the PI. Not rerun by this audit.

## 8. Clean-Control Config Creation: Authorized Next?

**Yes — Decision A.** The audit passes; creation of the seed-123 clean Multi-Krum control config is authorized as the next step, subject to the disclosed, non-blocking G limitation being kept in mind (verify `multikrum_diagnostics.csv` after that first run completes, before proceeding further).

## 9. Exact Next Recommended Config

A byte-for-byte replication of `configs/cifar100_fedavg_dirichlet_a01_seed123_50rounds.yaml` (**EXP_008**), following the exact "strict single-variable replication" convention already established for `EXP_008`→`EXP_016` (confirmed by inspecting `EXP_016`'s own header comment), with **only** these fields changed:

| Field | EXP_008 value | Recommended clean Multi-Krum control value |
|---|---|---|
| `experiment_id` | `EXP_008` | Next unused sequential ID — `EXP_020` by the repository's current numbering (highest existing is `EXP_019`); the PI should confirm this against whatever external experiment ledger is authoritative, since none was found inside this repository to check automatically |
| `experiment_name` | `cifar100_fedavg_dirichlet_a01_seed123_50rounds` | e.g. `cifar100_multikrum_dirichlet_a01_seed123_50rounds_clean` |
| `output_dir` / `checkpoint_dir` | EXP_008's paths | New, non-colliding paths matching the new experiment name |
| `aggregation` | `fedavg` | `multikrum` |

Every other field (`num_clients: 5`, `clients_per_round: 5`, `minimum_updates: 5`, `communication_rounds: 50`, `dirichlet_alpha: 0.1`, `seed: 123`, optimizer/scheduler/precision settings, and — critically — **no `attack_type` field at all**, i.e. a clean, honest-only run) stays byte-identical to EXP_008, exactly matching this task's required "first future run must be a CLEAN Multi-Krum control, not an attacked run" and the plan's own Section 7/8 pairing (seed 123 ↔ EXP_008, then seed 2026 ↔ EXP_009 second). **This config was not created in this task.**

## 10. Confirmation No Experiment Was Run

Confirmed. No call to `src/experiments/run_federated_cifar100.py` (as a script or otherwise) was made in this task.

## 11. Confirmation No Config Was Created or Modified

Confirmed. No file under `configs/` was created, edited, or written to. Only reads were performed on `EXP_008`, `EXP_009`, and `EXP_016`'s existing config files.

## 12. Confirmation No Code/Test/FedTrace/Attack/Research_Database File Was Changed

Confirmed. `src/federated/aggregation/multikrum.py`, `src/federated/aggregation/fedavg.py`, `src/federated/aggregation/base.py`, `src/experiments/run_federated_cifar100.py`, `tests/test_multikrum_aggregation.py`, and `tests/test_aggregation.py` were all read-only in this task. `src/analysis/fedtrace_detector.py` and `src/attacks/traceability.py` were not opened or touched. A repository search for a `Research_Database` file found none to inspect or change. The only file written by this task is this audit document itself, `analysis/MULTIKRUM_IMPLEMENTATION_VERIFICATION.md`.

## 13. Commands Executed

**None.** This audit was performed entirely via read-only file inspection (`Read`) and file-pattern/content search (`Glob`, `Grep`) tool calls. No shell, PowerShell, or Bash command was executed; no Python process (including `pytest` or a syntax check) was run at any point in this task.
