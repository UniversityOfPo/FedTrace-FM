# Gaussian vs. Directional Multi-Krum, Seed 123: Synthesis

**Status:** Analysis/synthesis document only. No code, test, config, README, FedTrace, attack, or Research Database file was created or modified in producing it. No experiment was run. Multi-Krum's `f=1`/`m=4`, tie-breaking, weighting, buffer handling, and the pre-registered `P_krum=3` rejection proxy were not modified or tuned. FedTrace v3 remains frozen. Multi-Krum's exclusion pattern is referred to throughout as the **"Multi-Krum rejection proxy"**, never as an attribution detector.

This document elevates the cross-attack-family comparison from `analysis/EXP_024_MULTIKRUM_DIRECTIONAL_SEED123_ANALYSIS.md` §10 into a standalone synthesis, because — as this task's own framing anticipated — it is potentially central to the paper's argument and merits independent reference.

---

## 1. The Comparison

Same seed (123), same Dirichlet(α=0.1) partition, same aggregation strategy (Multi-Krum), same malicious client (`client_002`), same attack onset (round 20) — the *only* variable is which attack family is active:

| | `EXP_022` (Gaussian, `rho=1.0`) | `EXP_024` (Directional, `target_cosine=0.0`) |
|---|---|---|
| Magnitude change | Unconstrained — attack adds independent noise, inflating `update_l2_norm` substantially (5.44→7.64→9.26 over the post-attack window) | **None by design** — `delta_attack_norm_ratio≈1.0` at every round (deviation ~1e-10) |
| Direction change | A side effect of adding noise (partial `self_prev_round_cosine` degradation) | **The entire mechanism** — rotated to `target_cosine=0.0`, `self_prev_round_cosine` collapses to ~0 |
| `client_002` post-attack Krum rank | **5 (rejected), every round, 30/30** | **2–3 (selected), every round, 0/30 rejected** |
| Rejection-target switch at round 20? | Instantaneous, complete | **None — no switch occurs at all** |
| Full-run rejection-proxy attribution | Misattributes `client_004` (wrong), but `client_002` eventually confirmed (round 22) | Misattributes `client_004` (wrong), and `client_002` is **never** confirmed |
| Attacked vs. clean Multi-Krum (Top-1) | **+4.52 pp** (attacked outperforms clean) | **−5.55 pp** (attacked underperforms clean) |
| Attacked Multi-Krum vs. attacked FedAvg (Top-1) | −2.07 pp | **−8.74 pp** (much larger gap) |
| Attack effect under FedAvg alone (Top-1, from each attack's own Stage B analysis) | −0.33 pp ("essentially neutral") | **−3.73 pp** (real, substantial damage) |
| FedTrace v3 result (paired FedAvg run) | Correct, onset 20, delay 2, 0 honest FP | Correct, onset 20, delay 2, 0 honest FP |
| FedTrace's dominant signal | `z_m` (magnitude), 30/30 anomalous rounds | `z_t_res` (temporal residual), 30/30 anomalous rounds |

## 2. Is Norm-Preserving Manipulation Materially Harder for Multi-Krum? Yes — Confirmed, Not Merely Predicted

`EXP_022`'s own trace analysis (`analysis/EXP_022_MULTIKRUM_GAUSSIAN_SEED123_ANALYSIS.md` Part H) offered, as a candidate mechanism, that Multi-Krum's raw Euclidean-distance metric separates the Gaussian attacker primarily because that attack inflates magnitude, and that a magnitude-invariant signal (like the naive peer-cosine baseline) would not respond the same way. This document's evidence goes further than that hypothesis: it is a **direct experimental test**, not an inference. Holding the attacker, seed, partition, onset, and aggregation strategy fixed and varying only whether the attack preserves norm:

- **Under magnitude-revealing attack (Gaussian):** Multi-Krum's raw distance metric separates the attacker completely and immediately — 100% post-attack rejection, zero-lag regime switch.
- **Under norm-preserving attack (directional):** Multi-Krum's raw distance metric **never separates the attacker at all** — 0% post-attack rejection, no regime switch, no exclusion-based protection whatsoever.

This is the cleanest possible same-seed, same-attacker, single-variable comparison available in this program, and it settles the question the Gaussian-only evidence could only gesture at: **Multi-Krum's robustness, as implemented (raw squared Euclidean distance over flattened trainable-parameter deltas), is fundamentally a magnitude-sensitive mechanism.** A genuinely norm-preserving, direction-only attack is a real, demonstrated blind spot for it in this configuration — not a theoretical concern, an observed one.

## 3. Why FedAvg Also Suffers Real Damage Here, Unlike the Gaussian Case

A second, related contrast worth stating explicitly: the Gaussian attack was "essentially neutral" against plain FedAvg (`analysis/EXP_016_STAGE_B_GAUSSIAN_SEED123_VALIDATION.md` §9: −0.33 pp Top-1) because FedAvg's sample-count weighting dilutes one attacker's contribution regardless of what that contribution contains. The directional attack, by contrast, causes real, substantial FedAvg damage (`analysis/EXP_018_STAGE_B_DIRECTIONAL_SEED123_VALIDATION.md` §17: −3.73 pp Top-1) — **dilution by sample-count weight does not neutralize a direction-flipping attack the way it happens to blunt a magnitude-inflating one**, because a rotated gradient direction, even down-weighted to 16% influence, still pulls the aggregate away from the honest consensus direction every round, compounding over 30 rounds. This matters for interpreting Part 4 below: Multi-Krum's failure here is not "FedAvg was already safe and Multi-Krum added no value" — FedAvg itself takes real damage from this attack. Multi-Krum still does *worse*.

## 4. Why Multi-Krum Underperforms FedAvg *More* Under the Attack It Fails to Catch

This is a further, non-obvious finding, distinct from "Multi-Krum fails to exclude the attacker." Even given that failure, one might expect Multi-Krum's outcome to be *no worse* than FedAvg's, since both aggregation rules end up incorporating `client_002`'s poisoned update. Instead, Multi-Krum's directional-attack damage (−8.74 pp Top-1 relative to FedAvg under the same attack) substantially exceeds FedAvg's own damage relative to its honest baseline (−3.73 pp).

**Confirmed, mechanistic explanation (not speculation):** `client_002` holds 7,202 of 45,000 total samples (16.0%). Under FedAvg's sample-weighted average, its poisoned update — whenever it participates, every round — receives exactly that 16.0% weight. Under Multi-Krum's **unweighted** average of the `m=4` selected clients (confirmed directly from `src/federated/aggregation/multikrum.py`; `client_002` is selected in all 30 post-attack rounds, Part 4/§B of the companion analysis), its poisoned update instead receives a full **25% (1/4) weight** every one of those 30 rounds — objectively *more* influence than FedAvg would give it, entirely independent of the exclusion failure. Multi-Krum thus compounds two harmful effects here: it fails to exclude the attacker, *and* its own unweighted-averaging convention (a deliberate, literature-faithful design choice, not a bug — `research_plans/MULTIKRUM_EXTERNAL_BASELINE_IMPLEMENTATION_PLAN.md` §4) happens to over-weight this specific attacker relative to what FedAvg would do. This is reported as a **supported interpretation**, not a fully isolated causal decomposition (disentangling exactly how much of the −8.74 pp gap is attributable to the weighting difference versus the exclusion-failure-plus-honest-client-exclusion combination would require a controlled ablation not performed in this task).

## 5. What This Does and Does Not Establish

**Established:**
- In this exact configuration (5 clients, strong Non-IID `α=0.1`, `client_002` attacker, round-20 onset, seed 123), Multi-Krum's raw-Euclidean-distance robustness mechanism completely fails against a norm-preserving directional attack while completely succeeding against a magnitude-revealing Gaussian one — a direct, same-seed, single-variable-controlled demonstration, not an inference from separate runs.
- FedTrace v3, using its temporal-residual component specifically for the directional case (dominant in 100% of anomalous rounds, per `EXP_018`'s frozen analysis) and its magnitude component for the Gaussian case (dominant in 100% of anomalous rounds, per `EXP_016`'s frozen analysis), succeeds with materially identical timing (onset 20, delay 2, zero honest FP) against **both** attack families under the identical seed and partition — exactly the cross-attack coverage property its design was built to provide (`research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md`), now shown operating against a method (Multi-Krum) that has no comparable coverage across the same two attack families.
- Multi-Krum's own aggregation-robustness value (as opposed to its repurposed attribution proxy) is attack-geometry-dependent in this configuration: genuinely protective in one case (Gaussian, though still net-unhelpful relative to FedAvg per `EXP_022`'s own analysis) and actively harmful relative to FedAvg in the other (directional).

**Not established:**
- That this blind spot generalizes beyond `target_cosine=0.0`, this partition, this seed, or this client count. A weaker rotation (closer to the honest direction) would very plausibly be caught even less; whether a stronger effective distance perturbation (not tested, and not to be tuned per this task's restrictions) would eventually trigger exclusion is an open, untested question.
- That Multi-Krum's robust-aggregation function is defective in general — only that it has a demonstrated, attack-geometry-specific blind spot in this configuration, alongside a genuine (if net-unhelpful) exclusion capability against a different, magnitude-revealing attack family.
- That raw-Euclidean-distance robust aggregation is blind to *all* direction-only attacks in general, beyond this specific norm-preserving rotation.
- That the seed-2026 directional run (not yet performed) will show the same pattern — this is a single-seed finding.

## 6. Contribution to the Project's Central Argument

This comparison directly strengthens the project's traceability-vs-robust-aggregation distinction in a way the Gaussian-only or directional-only evidence individually could not:

- The Gaussian-family evidence (`EXP_022`/`EXP_023`) showed that even when Multi-Krum's raw scoring *does* separate the true attacker, its repurposed full-run attribution proxy still misattributes an honest client first — a limitation of using a memoryless, per-round exclusion signal for attribution.
- This directional-family evidence adds a second, distinct limitation: Multi-Krum's raw scoring can **fail to separate the attacker from the honest cluster at all**, depending on the attack's geometry — a limitation of the underlying distance metric itself, not merely of repurposing its output for attribution.
- FedTrace v3, by contrast, succeeds against both attack families with the same detection timing, because its design deliberately combines a magnitude-sensitive signal (`z_m`) with a temporal-self-consistency signal (`z_t_res`) that responds to direction changes a magnitude-only or raw-distance-only method cannot reliably catch.

**The paper's claim that robust aggregation and forensic attribution solve different problems is now supported by two independent, mechanistically distinct pieces of evidence** — a memoryless-scoring/attribution mismatch (Gaussian family) and an attack-geometry blind spot (directional family) — rather than one.

---

## Final Report Addendum (This Document)

- **File created:** `analysis/GAUSSIAN_VS_DIRECTIONAL_MULTIKRUM_SEED123_SYNTHESIS.md` (companion to `analysis/EXP_024_MULTIKRUM_DIRECTIONAL_SEED123_ANALYSIS.md`, whose Final Report covers both files jointly).
- No code, config, test, README, FedTrace, attack, or Research Database file was changed; no experiment was run; no command was executed.
