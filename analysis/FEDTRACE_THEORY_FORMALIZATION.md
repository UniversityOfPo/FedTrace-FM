# FedTrace v3 — Mathematical / Theoretical Formalization

**Status:** Theory/formalization document only. `src/analysis/fedtrace_detector.py` (frozen FedTrace v3), `src/attacks/traceability.py`, `src/federated/aggregation/multikrum.py`, and every existing config/test/result/checkpoint/README/Research-Database/workbook file were read but **not modified**. No experiment, training run, or FedTrace/Multi-Krum analysis was executed. No threshold, persistence rule, calibration convention, or combiner was changed. This document formalizes only what the frozen implementation and the project's own already-completed, human-reviewed analysis series actually established — it introduces no new algorithm and no unsupported statistical guarantee (see Part 14).

**Companion files:** `analysis/FEDTRACE_LATEX_EQUATION_LIBRARY.md` (Part 15 content), `analysis/FEDTRACE_THEOREM_LEMMA_INVENTORY.csv` (Part 16 content).

---

## Part 1 — Sources Read and Implementation-vs-Plan Check

Read in full or targeted re-verification in this task: `research_plans/FEDTRACE_METHOD_DESIGN_PLAN.md`, `research_plans/FEDTRACE_V2_TEMPORAL_CALIBRATION_PLAN.md`, `research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md`, `analysis/FEDTRACE_V3_RETROSPECTIVE_ANALYSIS.md`, `analysis/FEDTRACE_V2_RETROSPECTIVE_ANALYSIS.md`, `analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md`, `analysis/FEDTRACE_FINAL_GENERALIZATION_GATE_STATUS.md`, `analysis/MULTIKRUM_EXTERNAL_BASELINE_FINAL_SYNTHESIS.md`, `analysis/FEDTRACE_MASTER_RESEARCH_STATUS.md`, `src/analysis/fedtrace_detector.py`, `src/attacks/traceability.py`, `src/federated/aggregation/multikrum.py`.

**Implementation-vs-plan discrepancy check: none found beyond what was already documented.** Every formula in Parts 3–4 below matches `src/analysis/fedtrace_detector.py` exactly, term for term. The one previously-known documentation-completeness gap (`analysis/EXP_018_STAGE_B_DIRECTIONAL_SEED123_VALIDATION.md` §10's "no honest client ever exceeds 3" overclaim, already flagged in `analysis/FEDTRACE_MASTER_RESEARCH_STATUS.md` §3/§7) is not re-litigated here; Part 14/Part 15 use only the corrected wording ("zero honest persistent false positives"). One additional, newly-confirmed-in-this-task detail, material to Part 5: the Gaussian attack's noise is **not** raw `N(0, σ²I)` — `src/attacks/traceability.py::generate_gaussian_attack_noise` draws i.i.d. standard-normal noise and then **rescales the entire noise vector so its L2 norm equals exactly `ρ·‖Δ‖₂`** before adding it. This is a sharper, more favorable setup for Part 5's derivation than a generic "isotropic Gaussian" framing would give (see Part 5's remark).

## Part 2 — Formal Problem Definition

**Federated learning notation.** At communication round `r`, let `w_r` denote the global model. Client `k ∈ K_r` (the participating-client set that round, `n_r = |K_r|`) performs local training starting from `w_r` and produces a local model `w_{k,r}`. The **submitted update** is

```
Δ_{k,r} = w_{k,r} − w_r
```

restricted to the trainable-parameter keys (matching `flattened_l2_norm`/`compute_update_l2_norm` in `src/attacks/traceability.py` and the identical trainable-key restriction in `src/federated/aggregation/multikrum.py`). `Δ_{k,r} ∈ ℝ^d` after flattening, with `d = 11{,}220{,}132` for every experiment in this project's registry (`resnet18_cifar`, all parameters trainable).

**The forensic problem.** Given the logged trace `{Δ_{k,r} : k ∈ K_r, r = 0, …, R−1}` (equivalently, the cheap per-round diagnostic features derived from it — `m_{k,r}`, `t_{k,r}`, Part 3), jointly estimate:

```
k̂_adv    (predicted malicious client identity)
r̂_onset  (predicted attack onset round)
```

**Constraint, stated explicitly because it is the entire point of the exercise:** the estimator computing `k̂_adv` and `r̂_onset` must be a function of `{Δ_{k,s} : s ≤ r}` alone — it must never read `attack_start_round` or `malicious_client_id` as an input to its own decision. (Ground truth is used only *afterward*, by a separate evaluation function, to score `k̂_adv`/`r̂_onset` against it — `evaluate_detector_v3` in the implementation reads `attack_start_round`/`malicious_client_id` exclusively for this post-hoc scoring, never inside `apply_persistence_rule_v3`.)

**This is explicitly *not* the global-anomaly-detection problem** "is *some* client behaving anomalously this round?" — a question a per-round statistic alone can sometimes answer. It is the strictly harder, two-part **forensic** problem: **which** client, and **since when**, using only information available at analysis time from the accumulated trace. A method that only answers "yes/no, something is wrong this round" does not solve this problem even if its yes/no answers are correct.

## Part 3 — Formalization of the FedTrace v3 Signals

Notation matches `src/analysis/fedtrace_detector.py` exactly; `T_ref = 20`, `τ = 3.0`, `P = 3` throughout (`DEFAULT_T_REF`, `DEFAULT_THRESHOLD`, `DEFAULT_PERSISTENCE`).

**(A) Magnitude signal.**
```
m_{k,r} = ‖Δ_{k,r}‖₂
```
Per-client historical calibration, using only the frozen pre-reference window `r < T_ref`:
```
μ^m_k = mean{ m_{k,r} : r < T_ref },     s^m_k = stdev{ m_{k,r} : r < T_ref }
```
(`statistics.stdev`, i.e. **sample** standard deviation, `n−1` denominator; excluded — `None`, never fabricated — if fewer than 2 observations or `s^m_k = 0`.) The magnitude anomaly score is **two-sided**:
```
z^m_{k,r} = | m_{k,r} − μ^m_k | / s^m_k
```

**(B) Temporal self-consistency signal.**
```
t_{k,r} = cos(Δ_{k,r}, Δ_{k,r−1})
```
defined for `r ≥ 1` only (undefined at `r = 0` for every client, by construction — there is no prior round).

**(C) Common round reference.** For each round `r`, over every client present that round (**no leave-one-out** — the client later being scored, including a malicious one, is included in computing its own round's reference; re-justified, not merely inherited from v2, per `research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md` §2):
```
c_r = median_{j ∈ K_r} t_{j,r}
```
(Excluded — `None` — if fewer than 2 clients have a defined `t` value that round; no scale/MAD is computed from this round at all, unlike v2.)

**(D) Temporal residual.**
```
d_{k,r} = c_r − t_{k,r}
```

**(E) Historical residual calibration.** Per client, over the same frozen pre-reference window, with round 0 excluded (`t_{k,0}`, hence `d_{k,0}`, is undefined for every client):
```
D_k = { d_{k,r} : 1 ≤ r < T_ref }
μ^d_k = mean(D_k),     s^d_k = stdev(D_k)
```
(Sample standard deviation, `n−1`, matching `μ^m_k`/`s^m_k`'s convention exactly — a deliberate choice, since at `T_ref = 20` this window has `n ≈ 19` observations, large enough that the small-sample instability documented for v2's *cross-sectional* scale (Part 9) does not apply to this *longitudinal* one.)

**(F) One-sided temporal residual anomaly score.**
```
z^t_{k,r} = max( 0, (d_{k,r} − μ^d_k) / s^d_k )
```
One-sided by design: an unusually **high** temporal self-consistency this round relative to a client's own history is not evidence of attack under this project's threat model, and scores exactly `0`, never a large negative number. (This document uses `z^t_{k,r}`; the source code and prior analysis documents use `z_t_res` interchangeably — same quantity.)

**(G) Final FedTrace v3 score.**
```
S_{k,r} = max( z^m_{k,r}, z^t_{k,r} )
```
**Undefined-component exclusion, precisely:** if only one of `z^m_{k,r}`/`z^t_{k,r}` is defined, `S_{k,r}` equals that one value — the missing component is *excluded from the max*, never substituted with `0` (which would bias `S_{k,r}` downward and could suppress a genuine detection). If neither is defined, `S_{k,r}` is itself undefined for that row (`combined_score` returns `None`), and that row is treated as non-anomalous when applying the persistence rule (Part 3-H) — it cannot start or extend a streak, but it also does not falsely register as "the value 0."

**(H) Threshold, persistence, and the three distinct round-labels.**

- **Raw anomaly** at `(k, r)`: `S_{k,r}` is defined and `S_{k,r} > τ = 3.0`.
- **First anomalous round** for client `k`: the smallest `r` with a raw anomaly, *regardless of whether it is ever confirmed persistent*. Can be an isolated, never-repeated event (Part 8's round-0 artifact is exactly this).
- **Confirmed streak**: a maximal run of **contiguous** rounds (`r, r+1, …`, no gap) each a raw anomaly for the same client, of length `≥ P = 3`.
- **Confirmation round**: the round at which a streak first reaches length `P` — i.e. the `P`-th round of that contiguous run. This is when the detector "fires."
- **Onset estimate** `r̂_onset(k)`: the **first** round of that confirmed `P`-length streak — always `confirmation_round − (P−1)` for a contiguous run, and the quantity compared against ground-truth `attack_start_round` for onset-localization error. **This is a retroactive label**: the decision to *act* is delayed until persistence is confirmed, but the *labeled onset* points back to where the anomalous behavior actually began, not to when the detector became confident about it.
- **Attribution rule**: `k̂_adv` = the client with the **earliest** confirmation round among all clients ever confirmed; ties broken by **higher mean score during its confirmed run**, then by **client_id** (ascending, lexical) — exactly `_select_predicted_client`'s three-key sort. A client never confirmed cannot be selected, regardless of its raw scores.

**These three labels are never conflated in this document or in any prior analysis document in this series**: e.g. in every Stage B/generalization run, "first anomalous round" (which can be round 0, an artifact — Part 8) is reported separately from "first anomalous round of the confirmed run" (which equals the true onset in every PASS case) and from "confirmation round" (`onset + 2` in every PASS case, per Part 11's lemma).

## Part 4 — Algorithm Block

**Algorithm 1: FedTrace v3 — Offline Attribution and Onset Localization**

This is an explicitly **offline/retrospective** algorithm — it consumes an already-collected, complete trace and is re-run afresh over the whole trace, exactly as `src/analysis/fedtrace_detector.py` implements it (its own module docstring: *"operates purely on already-collected client_round_trace.csv rows; it trains nothing, runs no federated round"*). **No streaming/online variant exists or is claimed here** — although the persistence rule (Part 3-H) is causal in the sense that `apply_persistence_rule_v3` never looks at a future round to decide a past round's status, the *calibration* step (`μ^m_k`, `s^m_k`, `μ^d_k`, `s^d_k`) is computed once, up front, from the full `[0, T_ref)`/`[1, T_ref)` windows of the *entire loaded trace* — it is not incrementally updated as new rounds arrive. Presenting this as "online-capable" would overclaim; it is stated here as offline, matching exactly what was evaluated in every experiment in this project.

```
Algorithm 1: FedTrace-v3-Offline(rows, T_ref=20, τ=3.0, P=3)
Input:  rows      -- the complete client_round_trace (all rounds, all clients)
        T_ref, τ, P
Output: ranking    -- clients ordered by confirmation round (earliest first)
        k_hat      -- predicted malicious client (or NONE if nobody confirmed)
        r_hat      -- onset estimate for k_hat (or NONE)
        status[k]  -- per-client: {first_anomalous_round, confirmation_round,
                                    onset_estimate, mean_score_during_run}

1.  clients <- distinct client_id values in rows
2.  rounds  <- distinct round_id values in rows

    # --- Calibration pass (uses ONLY r < T_ref data; computed once, up front) ---
3.  for each k in clients:
4.      mu_m[k], s_m[k] <- mean/stdev{ m_{k,r} : r < T_ref }      (or NONE if <2 obs / s=0)
5.  for each r in rounds:
6.      c[r] <- median_{j in K_r} t_{j,r}                          (or NONE if <2 clients)
7.  for each k in clients:
8.      D_k <- { c[r] - t_{k,r} : 1 <= r < T_ref, both defined }
9.      mu_d[k], s_d[k] <- mean/stdev(D_k)                         (or NONE if <2 obs / s=0)

    # --- Scoring pass (every round, every client -- NEVER reads attack_start_round
    #     or malicious_client_id) ---
10. for each (k, r) in rows:
11.     z_m <- |m_{k,r} - mu_m[k]| / s_m[k]           if mu_m[k] defined and m_{k,r} defined, else NONE
12.     z_t <- max(0, (c[r] - t_{k,r} - mu_d[k]) / s_d[k])
                                                        if mu_d[k], c[r], t_{k,r} all defined, else NONE
13.     S[k,r] <- max({z_m, z_t} \ {NONE})              if at least one defined, else NONE

    # --- Persistence pass (per client, independently; causal -- no future round read) ---
14. for each k in clients:
15.     streak <- 0; streak_start <- NONE; confirmed <- NONE; first_anom <- NONE
16.     for r in sorted(rounds):
17.         anomalous <- (S[k,r] is defined) and (S[k,r] > tau)
18.         if anomalous and first_anom is NONE: first_anom <- r
19.         if anomalous:
20.             if r == previous_r + 1 and previous_was_anomalous: streak <- streak + 1
21.             else: streak <- 1; streak_start <- r
22.             if streak >= P and confirmed is NONE:
23.                 confirmed <- r;  status[k].onset_estimate <- streak_start
24.         else:
25.             streak <- 0; streak_start <- NONE
26.         previous_r <- r; previous_was_anomalous <- anomalous
27.     status[k].first_anomalous_round <- first_anom
28.     status[k].confirmation_round    <- confirmed

    # --- Attribution ---
29. confirmed_clients <- { k : status[k].confirmation_round is defined }
30. if confirmed_clients is empty: return (NONE ranking, NONE, NONE, status)
31. ranking <- sort confirmed_clients by (confirmation_round ASC,
                                           mean_score_during_run DESC,
                                           client_id ASC)
32. k_hat <- ranking[0];  r_hat <- status[k_hat].onset_estimate
33. return (ranking, k_hat, r_hat, status)
```

This is a direct, line-for-line restatement of `compute_all_scores_v3` (lines 1–13) + `apply_persistence_rule_v3` (lines 14–28) + `_select_predicted_client` (lines 29–33) in `src/analysis/fedtrace_detector.py` — verified against that source in this task, not written independently and hoped to match.

---

## Part 5 — Theoretical Result 1: Gaussian Norm Inflation

Let `Δ ∈ ℝ^d` be an honest update and `ε ∈ ℝ^d` the attack perturbation, `Δ' = Δ + ε`.

**Exact algebraic identity (no assumption needed):**
```
‖Δ'‖² = ‖Δ‖² + 2⟨Δ, ε⟩ + ‖ε‖²
```

**Implementation-accurate refinement of the noise model.** As confirmed against `src/attacks/traceability.py::generate_gaussian_attack_noise` in this task (Part 1), `ε` is **not** raw `N(0, σ²I)` noise. It is constructed as an i.i.d. standard-normal draw whose **direction** is (by the standard fact that a normalized isotropic Gaussian vector is uniformly distributed on the unit sphere) uniform on `S^{d−1}`, independent of `Δ`, then **deterministically rescaled** so that
```
‖ε‖ = ρ · ‖Δ‖     exactly, for the configured attack-strength ratio ρ (ρ = 1.0 in every Gaussian run in this project)
```

**Proposition P1 (Gaussian squared-norm inflation, exact in expectation).** Under this construction — `‖ε‖ = ρ‖Δ‖` fixed, direction of `ε` uniform on `S^{d−1}` independent of `Δ` —
```
E[⟨Δ, ε⟩] = 0     (exactly, by the reflection symmetry ε ↔ −ε of the uniform sphere distribution)
```
hence
```
E[‖Δ'‖²] = ‖Δ‖² + ρ²‖Δ‖² = (1 + ρ²)‖Δ‖²
```
**Proof.** Write `ε = ‖ε‖ · U` with `U` uniform on `S^{d−1}`, independent of `Δ`. `⟨Δ, ε⟩ = ‖ε‖⟨Δ, U⟩`. Since `U` and `−U` have the same distribution (the uniform distribution on the sphere is symmetric), `⟨Δ, U⟩` and `−⟨Δ, U⟩` have the same distribution for any fixed `Δ`, so `E[⟨Δ, U⟩] = 0`. Linearity of expectation over the deterministic scale `‖ε‖ = ρ‖Δ‖` gives the result. ∎

**Precision about expectation vs. per-realization (mandatory, per this task's own instruction):**
- This is an **expectation-level** statement about `⟨Δ, ε⟩`, not a per-draw guarantee. **A single realized draw can have `⟨Δ, ε⟩ < 0`** (the random direction happens to point substantially against `Δ`); in that case `‖Δ'‖²` can be *smaller* than `(1+ρ²)‖Δ‖²`, and — precisely — `‖Δ'‖ < ‖Δ‖` occurs exactly when `cos θ < −ρ/2` (solving `2ρ‖Δ‖²cosθ + ρ²‖Δ‖² < 0`), where `θ` is the angle between `Δ` and `ε`. **This document does not claim every Gaussian draw increases the norm.**
- **However**, for `d = 11{,}220{,}132` (this project's actual parameter count), the projection `⟨Δ, U⟩` of a uniform sphere direction onto any fixed vector **concentrates sharply around 0** (its variance scales as `O(1/d)`), so in this project's specific high-dimensional regime, `‖Δ'‖² ≈ (1+ρ²)‖Δ‖²` holds with overwhelming realized-draw probability, not merely in expectation — a substantially stronger, still honest, statement specific to this project's actual dimensionality, not a generic claim about Gaussian noise in low dimensions.

**Why this supports the empirical finding.** At `ρ = 1.0`, `E[‖Δ'‖²] ≈ 2‖Δ‖²`, i.e. `‖Δ'‖ ≈ √2 · ‖Δ‖` — a large, easily-calibration-detectable shift relative to `μ^m_k`/`s^m_k`'s honest-only spread. This is exactly consistent with `z^m` dominating detection in every Gaussian-family run (`EXP_014`, `EXP_016`, and, mixedly with `z^t`, `EXP_017`) and is **not** claimed to generalize to attack constructions that do not fix `‖ε‖` proportionally to `‖Δ‖`, nor to low-dimensional settings where the concentration argument above would not hold.

---

## Part 6 — Theoretical Result 2: Norm-Preserving Directional Attack

Formalizing `src/attacks/traceability.py::generate_directional_attack_perturbation`, confirmed to match exactly in this task.

Let `Δ ≠ 0`, `Δ̂ = Δ / ‖Δ‖`. Draw a random raw vector `r`, form `u_raw = r − ⟨r, Δ̂⟩Δ̂` (Gram–Schmidt removal of the `Δ̂` component) and, provided `u_raw ≠ 0`, `u_⊥ = u_raw / ‖u_raw‖`, so `⟨u_⊥, Δ̂⟩ = 0` and `‖u_⊥‖ = 1` by construction. For target cosine `c ∈ [−1, 1]`, define
```
a = c·Δ̂ + √(1 − c²)·u_⊥,          Δ_attack = ‖Δ‖ · a
```

**Proposition P2 (exact norm and cosine control).** For any `c ∈ [−1, 1]` and any unit `u_⊥ ⊥ Δ̂`:
```
‖Δ_attack‖ = ‖Δ‖          and          cos(Δ_attack, Δ) = c
```
**Proof.** Since `Δ̂` and `u_⊥` are orthonormal, `‖a‖² = c²‖Δ̂‖² + (1−c²)‖u_⊥‖² + 2c√(1−c²)⟨Δ̂,u_⊥⟩ = c² + (1−c²) + 0 = 1`, so `‖a‖ = 1` and `‖Δ_attack‖ = ‖Δ‖·‖a‖ = ‖Δ‖`. For the cosine: `⟨Δ_attack, Δ⟩ = ‖Δ‖·⟨a, Δ⟩ = ‖Δ‖·‖Δ‖·⟨a, Δ̂⟩ = ‖Δ‖²·(c·1 + √(1−c²)·0) = c‖Δ‖²`. Hence `cos(Δ_attack, Δ) = ⟨Δ_attack,Δ⟩ / (‖Δ_attack‖‖Δ‖) = c‖Δ‖²/‖Δ‖² = c`. ∎

**Special case `c = 0` (the value used in every directional-attack experiment in this project, `EXP_015`, `EXP_018`, `EXP_019`, `EXP_024`, `EXP_025`, `EXP_026`, `EXP_027`):** `a = u_⊥` exactly, so `Δ_attack = ‖Δ‖·u_⊥` is **exactly orthogonal** to `Δ` (`cos(Δ_attack, Δ) = 0`) while preserving `‖Δ‖` exactly — this is not approximate; it follows directly from the `c=0` substitution in the proof above.

**Edge cases, matching the implementation exactly (not merely asserted):**
- **`Δ = 0`:** `Δ̂` is undefined; the implementation returns `Δ_attack = 0` directly, with no random draw — trivially `‖Δ_attack‖ = ‖Δ‖ = 0`, and `cos(·,·)` is left undefined (as it is for any zero vector), consistent with `compute_cosine_similarity` returning `None` rather than a fabricated value.
- **`c = ±1`:** `√(1−c²) = 0`, so `u_⊥`'s coefficient vanishes and `a = ±Δ̂` exactly, **regardless of which `u_⊥` would have been drawn** — the implementation special-cases this to skip the random draw entirely (both because it is unnecessary and to avoid depending on the near-zero-probability degenerate case `u_raw = 0`, which can only arise when the drawn `r` is itself exactly parallel to `Δ̂`).

**Verified against the artifact record, not merely proved abstractly:** every directional-attack run's `client_round_trace.csv` carries `delta_attack_norm_ratio` and `realized_cosine_to_honest` fields computed independently by the pipeline itself; these were checked in `EXP_018`'s and `EXP_027`'s own analyses to be `1.0`/`0.0` to within `~10⁻¹⁰` at every active round — the floating-point residual expected from Proposition P2's algebra, not a discrepancy from it.

---

## Part 7 — Theoretical Result 3: Why Magnitude-Only Detection Can Fail

**Corollary C1 (norm-only indistinguishability under exact preservation).** Let `f: ℝ_{≥0} → {honest, attack}` be *any* deterministic decision rule that reads only the scalar `‖Δ_{k,r}‖` for a single round `r` (no other feature, no history). If `Δ_attack` is constructed as in Part 6 (Proposition P2), then for every `c`, `‖Δ_attack‖ = ‖Δ‖` exactly, so `f(‖Δ_attack‖) = f(‖Δ‖)` — **`f` outputs the identical decision on the attacked round as it would have on the honest counterfactual, because it is given the identical input.** No such `f` can distinguish the two on that round's norm alone, by construction, not by any property of `f` itself.

**Precision about scope, exactly as this task instructs:**
- This is a statement about **the information content of the single scalar `‖Δ_{k,r}‖` at one round**, given the attack's own exact-norm-preservation guarantee (Part 6). It is **not** a universal impossibility result for every magnitude-*history* method: a method using `‖Δ_{k,r}‖`'s trajectory across many rounds *could* still succeed if the attack indirectly perturbs later rounds' honest-counterfactual norms through the global model (exactly what empirically happens — Part 3 of `analysis/EXP_018_STAGE_B_DIRECTIONAL_SEED123_VALIDATION.md`, `analysis/EXP_027_ONSET_GENERALIZATION_ANALYSIS.md` — magnitude-only *did* eventually, weakly, cross threshold in `EXP_018`/`EXP_015`/`EXP_027`, just far too late to satisfy the pre-registered onset/delay criteria, and it *never* did so at all in `EXP_026`).
- This also does not claim `‖·‖`-based detection fails against **every** attack — Part 5 shows the opposite for the Gaussian family, precisely because that attack does *not* preserve norm.

---

## Part 8 — Theoretical Result 4: Common-Mode Drift Cancellation

Consider, for **honest** clients only, the working decomposition already used (informally) in `research_plans/FEDTRACE_V2_TEMPORAL_CALIBRATION_PLAN.md` and re-derived here explicitly:
```
t_{k,r} = g_r + b_k + η_{k,r}
```
`g_r` a round-wise common component (shared secular trend across all honest clients — this is what v1's static pre-attack calibration could not see coming), `b_k` a client-specific fixed offset, `η_{k,r}` client-specific round noise, mean-zero over `r`.

**Proposition P3 (common-mode drift cancellation, exact under the stated model).** If `c_r` in a given round is computed from a client subset whose `median_j (b_j + η_{j,r})` equals some value `β_r` (define `β_r` this way), then
```
c_r = g_r + β_r     and hence     d_{k,r} = c_r − t_{k,r} = (β_r − b_k) − η_{k,r}
```
— **the round-wise common term `g_r` cancels exactly**, leaving a residual whose distribution depends only on client-specific quantities (`b_k`, `η_{k,r}`) and the round's own realized `β_r`, not on the secular trend `g_r` that broke v1's static calibration.

**Robustness of `c_r` to the malicious client's own inclusion (no leave-one-out needed) — a defensible, not merely asserted, claim.** `c_r` is the sample **median**, whose breakdown point is `⌊(n_r−1)/2⌋` corrupted/arbitrary values before it can be moved to an arbitrary value. At `n_r = 5` (every experiment in this project), the median tolerates up to **2** arbitrary values among the 5 without becoming unbounded; a **single** attacker (this project's threat model throughout) is one such value, strictly below the breakdown point. This is why `c_r` including the attacker's own `t` value (no leave-one-out, per Part 3-C) does not, by itself, corrupt `c_r`'s estimate of `g_r + β_r` — though see the caveat below.

**Assumptions stated carefully, not glossed over:**
- This is an **exact** algebraic cancellation *given* the additive decomposition and *given* `median_j(b_j+η_{j,r})` is well-approximated by a fixed `β_r` each round — both are **modeling assumptions consistent with the observed data**, not derived from first principles of SGD training dynamics. The decomposition's validity is empirical/observational, exactly as already stated in `research_plans/FEDTRACE_V2_TEMPORAL_CALIBRATION_PLAN.md` and reiterated in `analysis/ICML_ICLR_EVIDENCE_GAP_AUDIT.md` Part 8.
- The median's breakdown-point argument establishes that a *single* attacker cannot make `c_r` unbounded; it does **not** by itself bound *how much* a single attacker's inclusion can perturb `c_r` within the breakdown point (a more precise perturbation bound would require additional distributional assumptions on `b_j+η_{j,r})` not established here — not claimed).

**Why this addresses v1's exact failure mode.** v1's `z_t` calibrated `t_{k,r}` against a **static** `μ_{pre}(k) = mean{t_{k,r} : r<T_ref}`, implicitly assuming `g_r` constant — false, per `analysis/FEDTRACE_STAGE_A_RETROSPECTIVE_ANALYSIS.md`'s finding that `t` declines secularly for every client. Because `d_{k,r}` (Proposition P3) has `g_r` removed at the *point of computation*, its own historical calibration (`μ^d_k`, `s^d_k`, Part 3-E) is calibrating a quantity that is *already* stationary with respect to `g_r` by construction — resolving exactly the mismatch that caused v1's late honest false positives, **without** needing v1's assumption that `g_r` is constant, and **without** re-introducing v2's own separate failure (Part 9), since the *scale* of `d_{k,r}` is estimated longitudinally per-client (Part 3-E), not cross-sectionally per-round.

---

## Part 9 — Theoretical Result 5: Why v2's Cross-Sectional MAD Is Unstable at n=5

**Failure-condition analysis (not a universal theorem about MAD).** v2 computed, per round, `scale_t(r) = 1.4826 · MAD_j(t_{j,r})` from only the `n_r = 5` clients present that round, then scored `z_t_rel(k,r) = (center_t(r) − t_{k,r}) / scale_t(r)`. **When several of the `n_r − 1` non-scored clients happen, by chance, to have nearly equal `t` values that round, `MAD` — itself a median-of-absolute-deviations, computed from only ~4–5 points — can become very small**, so an *ordinary*, non-malicious deviation for the scored client is divided by a near-zero denominator and inflated past threshold.

**Real, verified numerical illustration (not a constructed toy — the actual data from `analysis/FEDTRACE_V2_RETROSPECTIVE_ANALYSIS.md`, `EXP_015`, seed 42, honest `client_000`):**

| Round | `center_t(r)` | `scale_t(r) = 1.4826·MAD` | gap `= center_t(r) − t_{000,r}` | `z_t_rel(client_000)` |
|---|---|---|---|---|
| 8 | 0.3315 | 0.0075 | ≈0.0167 | 2.23 (not anomalous) |
| **9** | 0.3262 | **0.00706** | ≈0.0276 | **3.91** |
| **10** | 0.3164 | **0.00193** | ≈0.0239 | **12.35** |

At round 9, the four other clients' `t` values happened to cluster within a `0.005` band that round, driving `scale_t(9)` down to `0.00706`; `client_000`'s gap (`≈0.0276`) is unremarkable in absolute terms — *smaller* than a round-1 gap (`0.042`) that scored only `z=2.63` under a more typical round's larger scale — but the small denominator inflates it past threshold anyway. Round 10 is more extreme: an even tighter honest-peer cluster drives `scale_t(10)` to `0.00193`, turning a similarly ordinary `0.0239` gap into `z=12.35`. Three such consecutive rounds (9, 10, 11) satisfy `P=3` and **wrongly confirm honest `client_000` at round 11 — 9 rounds before the true attacker's own confirmation and 11 rounds before the true attack even begins.**

**Scope, stated conservatively:** this is a **small-sample failure-condition analysis of this specific estimator at this specific `n_r`**, illustrated with real, verified data from one development run — **not** a general theorem that median/MAD estimators are unsafe at every sample size, nor a claim about how often this specific clustering coincidence recurs across seeds/rounds (`analysis/FEDTRACE_V2_RETROSPECTIVE_ANALYSIS.md` §12 explicitly states this was not checked for recurrence, and this document does not check it either).

---

## Part 10 — Theoretical Result 6: Temporal-Residual Response to Directional Manipulation

**A conditional mechanistic remark, not a formula for `z^t_res` from `c` — stated with the exact scope this task requires.**

The directional attack (Part 6) enforces `cos(Δ_attack, Δ_r) = c` **against that same round's own honest counterfactual `Δ_r`** — it does **not** directly control, or even reference, `Δ_{r−1}` at all. Therefore **no exact formula for `cos(Δ_attack, Δ_{r-1})`, and hence no exact formula for the resulting `z^t_res`, follows from `c` alone** — this document does not claim one.

What can be stated, conditionally: write `Δ_{r-1} = α·Δ̂_r + β·v` for the (unique) decomposition of `Δ_{r-1}` into a component along `Δ̂_r` (`α = cos(Δ_{r-1}, Δ_r)·‖Δ_{r-1}‖`) and an orthogonal remainder `β·v` (`v ⊥ Δ̂_r`, unit). Then, using `Δ_attack = ‖Δ_r‖(c·Δ̂_r + √(1−c²)u_⊥)` from Part 6,
```
⟨Δ_attack, Δ_{r-1}⟩ = ‖Δ_r‖ [ c·α + √(1−c²)·β·⟨u_⊥, v⟩ ]
```
Because `u_⊥` is drawn independent of, and (for `d` this large) with overwhelming probability *not* aligned with, `v`, the term `⟨u_⊥,v⟩` is — by the same high-dimensional concentration reasoning as Part 5 — small in magnitude relative to `α` and `β` with high probability. **Under this condition** (`⟨u_⊥,v⟩` small), `⟨Δ_attack,Δ_{r-1}⟩ ≈ c·α·‖Δ_r‖`, so for `c` far from the honest `cos(Δ_r,Δ_{r-1})` — in particular for `c=0`, the value used throughout this project — the attacked cosine is driven toward `0`, i.e. **away from** whatever the honest self-consistency trajectory would have been, rather than toward it.

**This is a conditional, probabilistic mechanistic account, explicitly labeled as such — not a guarantee for every draw or every `(Δ_r, Δ_{r-1})` geometry.** It is offered as the *reason to expect* the empirically, repeatedly observed collapse (`self_prev_round_cosine` dropping to near-zero magnitude immediately at onset in every directional-attack trace inspected in this project — `EXP_015`, `EXP_018`, `EXP_019`, `EXP_024`–`027`), not as a derivation that reproduces the exact observed `z^t_res` values from `c` alone.

---

## Part 11 — Theoretical Result 7: Persistence-Delay Lemma

**Lemma L1 (minimum confirmation delay under `P`-round persistence).** If a client's score is a raw anomaly at every round in `[r*, r*+P-1]` (i.e. `P` consecutive anomalous rounds beginning exactly at `r*`), then:
```
confirmation_round = r* + P − 1          (exactly, under Algorithm 1's rule)
onset_estimate     = r*
detection_delay    = confirmation_round − r* = P − 1
```
and **no `P`-round-contiguous-persistence rule can confirm earlier than `r*+P-1`**, since by definition `P` contiguous anomalous rounds are required and the earliest round at which the `P`-th of them can have occurred, starting from `r*`, is `r*+(P-1)`.

**Proof.** Immediate from Algorithm 1 lines 19–23: `streak` increments by exactly 1 per contiguous anomalous round starting from `streak_start=r*`; `streak ≥ P` is first satisfied when `streak` has incremented `P` times, i.e. at round `r*+(P-1)`, at which point `confirmed ← r*+(P-1)` and `onset_estimate ← streak_start = r*`. Minimality follows because `streak` cannot exceed the count of contiguous anomalous rounds observed so far. ∎

**For `P = 3` (this project's frozen value): minimum confirmation delay is exactly 2 rounds.**

**Direct connection to the empirical record.** Every one of the 6 Stage B and generalization runs where FedTrace v3 correctly attributed the attacker (`EXP_016`–`019`, `026`, `027`) achieved `onset_error = 0` and `detection_delay = 2` **exactly** — the theoretical minimum from Lemma L1, not merely a good result. This is not a coincidence to be marveled at; it is the lemma's own necessary consequence of the attack signal becoming anomalous *immediately* at the true onset and remaining so for at least 3 consecutive rounds in every one of those runs (verified directly, per-run, in each analysis document) — Lemma L1 explains *why* `delay=2` is the best any run could have shown, and the empirical record shows every successful run achieved exactly that bound.

---

## Part 12 — Theoretical Result 8: Robust Aggregation vs. Forensic Attribution

**Structural Observation O1.** Define two classes of function over the trace:

**(A) A per-round robust-aggregation rule:**
```
Agg_r : { Δ_{k,r} : k ∈ K_r }  →  (selected/reweighted subset)  →  global update
```
— a function of **only round `r`'s own cross-section**. Multi-Krum (`src/federated/aggregation/multikrum.py`) is exactly this: `_krum_scores`/`_rank_and_select` read only `updates` (the current round's `ClientUpdate` objects) and `global_state`; no prior round's data is passed in or retained across the call.

**(B) A longitudinal forensic-attribution rule:**
```
Attr : { Δ_{k,s} : s ≤ r, k ∈ K_s }  →  (client identity, onset estimate)
```
— a function of the **entire accumulated history up to round `r`**. FedTrace v3 (Part 4, Algorithm 1) is exactly this: `μ^m_k`, `μ^d_k`, `s^d_k` are computed from the client's own multi-round window, and the persistence rule requires observing `P` rounds.

**Observation.** `Agg_r`'s domain contains no reference to any round `s < r` at all — it is *structurally incapable* of using a client's own history to decide round `r`'s exclusion, regardless of how well or poorly it is implemented. Consequently, **a sequence of correct per-round `Agg_r` decisions does not, by itself, constitute or imply a correct `Attr` output**: `Agg_r` has no mechanism to distinguish "this client is chronically at the geometric periphery of the honest population" (a heterogeneity fact, not evidence of that round's misbehavior) from "this client just became anomalous relative to *its own* history" (Part 6/12 of this project's own threat model) — both can produce an identical `Agg_r` exclusion decision at round `r`, for entirely different underlying reasons, and `Agg_r` has no state in which to record, or later consult, which reason applied.

**This is a structural/definitional observation about the two function signatures — not a theorem that any *specific* robust-aggregation rule always fails at attribution, and not a claim about robust aggregators in general.** The empirical Multi-Krum record (`EXP_020`–`025`) is used in this project as **one concrete instance** that exhibits exactly the predicted disconnect (a per-round-optimal-by-design exclusion rule whose full-run rejection-persistence proxy misattributes the honest largest-partition client in every attacked run, and never once corrects for the directional family) — it is evidence *for* the observation being materially, not merely hypothetically, true in this project's tested configuration, not a proof that it must hold for every robust aggregator.

---

## Part 13 — Multi-Krum Failure-Condition Analysis

Krum score, exactly as implemented (`_pairwise_squared_distances` + `_krum_scores`):
```
d(i,j) = ‖u_i − u_j‖₂²          score(i) = Σ over the (n−f−2) smallest {d(i,j) : j≠i}
```
`n=5`, `f=1` throughout this project, so `neighbors_used = n−f−2 = 2` and `m = n−f = 4` clients selected each round.

**Decomposition, exact:** `d(i,j) = ‖u_i‖² + ‖u_j‖² − 2⟨u_i,u_j⟩ = ‖u_i‖² + ‖u_j‖² − 2‖u_i‖‖u_j‖·cos θ_{ij}`.

**Why norm preservation alone does not imply evasion — precisely.** A norm-preserving attack (Part 6) leaves `‖u_attack‖ = ‖u_honest‖` (the honest counterfactual's own norm), so the *magnitude* terms `‖u_i‖²+‖u_j‖²` in every `d(attacker, j)` are, to first order, whatever they would have been honestly — this by itself says nothing about `cos θ_{ij}`, which the directional attack deliberately, substantially changes (`c=0` in this project ⇒ `cos θ_{attacker,Δ_honest} = 0`, an exact orthogonality per Proposition P2). **The distance to peers is still altered — through the `cosθ` term, not the norm term.**

**Failure condition — the precise, defensible statement (not a universal theorem).** Whether this altered `cosθ` term is *enough* to raise `score(attacker)` above the 4th-ranked honest client's score (triggering exclusion, since only the `m=4` lowest-scoring of `5` clients are selected) depends on **how large the honest-population's own pairwise-distance spread already is** at that heterogeneity level. Precisely: exclusion requires the attacker's score to exceed at least one honest client's score by enough to change the ranking; if the strong-Non-IID (`α=0.1`) honest population already has a large baseline pairwise-distance spread (directly established by this project's own `EXP_009` characterization — up to `~31%` client sample-count coefficient of variation, translating into materially different local-update geometries), the directional rotation's *added* distance can remain smaller than the *pre-existing* honest-to-honest gaps, leaving the ranking — and hence the selection — unchanged. **This is exactly the mechanism the empirical record is consistent with**: `EXP_024`/`EXP_025` show 0/30 post-attack rejection at both seeds, with the same largest-partition honest client persistently occupying the highest-score (most-excludable) rank both before and after the attack begins (`analysis/MULTIKRUM_DIRECTIONAL_CROSS_SEED_SYNTHESIS.md`).

**Contrast with the Gaussian family.** There, the attack inflates *both* the norm term (Part 5: `E[‖u_attack‖²] ≈ 2‖u_honest‖²` at `ρ=1`) and, generically, contributes an uncontrolled `cosθ` shift as well (no rotation is engineered to keep `cosθ` favorable) — so `d(attacker,j)` grows through the (dominant, since it scales with `‖u‖²` while `cosθ`'s contribution is bounded) magnitude term for *every* peer `j` simultaneously, a much larger and more uniform score increase than a single controlled rotation produces. This is consistent with, and offered as a mechanistic account of, the observed 30/30 rejection rate at both seeds.

**Explicitly not claimed:** a universal theorem that Multi-Krum (or Krum-family scoring in general) fails on all norm-preserving attacks, at all `f`/`n`/heterogeneity levels, or against a rotation smaller than `c=0`'s exact orthogonality. This is a **failure-condition / geometric-intuition statement**, tied to this project's own tested configuration and directly evidenced by it — not a general robustness-defeat proof about Krum-family aggregation.

---

## Part 14 — Limits of the Theory (Mandatory Section)

**Explicitly NOT proven or claimed anywhere in this document, its companion LaTeX library, or the theorem inventory:**

- **No statistical false-positive-rate guarantee.** Every "zero honest false positives" statement in this project (and in this document) is an **observed-zero** result over a specific, finite set of runs — never a bound on the probability of a future false positive under any stated noise model.
- **No asymptotic consistency theorem** for `z^m`/`z^t_res` as `T_ref → ∞`, sample size grows, or under any other limiting regime.
- **No arbitrary Byzantine-robustness guarantee** — this project makes no claim analogous to Krum's own `n ≥ 2f+3` resilience proof for FedTrace; FedTrace has no comparable formal guarantee of any kind.
- **No guarantee for multiple simultaneous attackers** — every experiment in this project (`EXP_014`–`027`) has exactly one malicious client; the median-breakdown-point argument in Part 8 explicitly assumes at most one arbitrary value among `n_r=5`, and does not extend to two or more colluding attackers without re-derivation.
- **No guarantee for partial client participation** — every experiment uses `clients_per_round = num_clients = 5` (full participation); the persistence rule's contiguous-round definition (Part 3-H) has not been re-examined for a regime where a client is not observed every round.
- **No guarantee for an attack onset before `T_ref = 20`** — explicitly flagged already in `analysis/FEDTRACE_MASTER_RESEARCH_STATUS.md` §4/§6 and restated here: an onset inside the calibration window would contaminate `μ^m_k`/`μ^d_k`/`s^d_k` with already-attacked data, a structurally different and untested regime this document does not analyze.
- **No guarantee across arbitrary datasets/models/client counts/heterogeneity levels** — every formula and every piece of empirical support cited in Parts 5–13 is tied to this project's specific configuration (CIFAR-100/ResNet-18, `d=11{,}220{,}132`, `n_r=5`, `α=0.1`, the two specific attack constructions actually implemented).
- **No universal theorem that Multi-Krum (or any Krum-family method) fails on norm-preserving attacks in general** — Part 13 is a failure-*condition* analysis tied to this project's own tested heterogeneity level and evidenced by, not proven independently of, the empirical record.
- **No theoretical claim resolves or supersedes any of the "Claims Not Supported" items already catalogued in `analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md` §17 or `analysis/FEDTRACE_MASTER_RESEARCH_STATUS.md` §4** — this document adds mathematical explanation for *why* the already-established empirical results look the way they do; it does not expand what is empirically claimed.

## Part 15 — See `analysis/FEDTRACE_LATEX_EQUATION_LIBRARY.md`

(Kept as a separate file per this task's explicit output-file instruction.)

## Part 16 — See `analysis/FEDTRACE_THEOREM_LEMMA_INVENTORY.csv`

(Kept as a separate file per this task's explicit output-file instruction; summarized in Part 17 below.)

## Part 17 — What Belongs in the Main Paper vs. Appendix vs. Development History

**MAIN PAPER (the strongest, most load-bearing 4 results — kept deliberately small):**
- **P2** (exact norm/cosine control of the directional attack) + **C1** (norm-only indistinguishability) — together, these are the precise mathematical justification for *why the directional attack family is a meaningful, non-trivial stress test* and *why a magnitude-only detector cannot be expected to catch it by construction*, which is the paper's central methodological motivation for needing a second signal at all.
- **P3** (common-mode drift cancellation) — the exact mathematical content of *why v3's calibration works and why v1's did not*, i.e. the paper's central methodological contribution's own justification.
- **L1** (persistence-delay lemma) — short, clean, directly explains the `delay=2` result reported in every successful validation/generalization run; cheap to include, high explanatory payoff.

**APPENDIX:**
- **P1** (Gaussian norm inflation) — genuinely useful, supports the Gaussian-family result, but secondary to the directional-attack-motivated design story that P2/C1/P3 carry in the main text.
- **O1** (robust-aggregation-vs-attribution structural observation) — important for the Multi-Krum section's precise wording (§5/6 of `FEDTRACE_MASTER_RESEARCH_STATUS.md`), but is a definitional observation, not a headline mathematical result; belongs with the full Multi-Krum comparison detail.
- **R3** (Multi-Krum geometric failure-condition analysis) — the mechanistic explanation for the empirical Multi-Krum dichotomy; valuable, appendix-appropriate depth alongside the full 2×2 results table.
- **R2** (conditional temporal-consistency disruption remark) — explicitly hedged/conditional by design (Part 10); useful supporting intuition, not main-paper-strength on its own.

**DEVELOPMENT HISTORY ONLY (not in the paper, main or appendix — matches `analysis/ICML_ICLR_EVIDENCE_GAP_AUDIT.md` Part 9's existing ablation-placement discipline):**
- **R1** (v2 MAD instability numerical illustration) — genuinely important for understanding *why* v2 was rejected, but is inherently about a superseded, frozen-out design iteration; belongs in the same "Development History / Design Iteration Case Study" appendix subsection `ICML_ICLR_EVIDENCE_GAP_AUDIT.md` Part 9 already recommends for the v1→v2→v3 ablation narrative, not as a standalone theory result.

**Explicitly avoided, per this task's overclaim warnings (not included as a numbered result anywhere):** any claim of asymptotic false-positive control, any Byzantine-robustness bound for FedTrace, any generalization of the Multi-Krum failure-condition analysis into a universal theorem, any exact `z^t_res`-from-`c` formula (Part 10 is deliberately conditional/qualitative only).
