# Multi-Krum External-Baseline Program: Final Synthesis

**Status:** Analysis/synthesis document only. No code, test, config, README, FedTrace, attack, or Research Database file was created or modified in producing it. No experiment was run. Multi-Krum's `f=1`/`m=4`, tie-breaking, weighting, buffer handling, and the pre-registered `P_krum=3` rejection proxy were not modified or tuned. FedTrace v3 remains frozen. Multi-Krum's exclusion pattern is referred to throughout as the **"Multi-Krum rejection proxy"**, never as an attribution detector.

This document closes the Multi-Krum external-baseline branch opened by `research_plans/MULTIKRUM_EXTERNAL_BASELINE_IMPLEMENTATION_PLAN.md`. It synthesizes all six experiments in the pre-registered minimal matrix (`EXP_020`–`EXP_025`) and every prior analysis document produced along the way, and renders the branch-closure verdict (Part 14).

---

## The Complete Six-Run Matrix

| | Seed 123 | Seed 2026 |
|---|---|---|
| **Clean Multi-Krum** | `EXP_020` (↔ `EXP_008`) | `EXP_021` (↔ `EXP_009`) |
| **Gaussian-attacked Multi-Krum** | `EXP_022` (↔ `EXP_016`) | `EXP_023` (↔ `EXP_017`) |
| **Directional-attacked Multi-Krum** | `EXP_024` (↔ `EXP_018`) | `EXP_025` (↔ `EXP_019`) |

All six runs are complete, technically valid (diagnostics structurally intact in every case; attacks independently verified genuinely norm-preserving/magnitude-scaled as configured in every attacked run), and paired against their exact frozen FedAvg counterparts with no scientific difference other than `aggregation: fedavg → multikrum`. This is exactly the minimal matrix `research_plans/MULTIKRUM_EXTERNAL_BASELINE_IMPLEMENTATION_PLAN.md` §8/§18 pre-registered — two clean controls, four attacked runs, no more.

## Part 13 — Paper-Level Claim Matrix

### CONFIRMED ACROSS BOTH SEEDS

- **Multi-Krum's clean-performance cost.** ≈−6.9 to −9.1 pp across accuracy metrics relative to clean FedAvg, both seeds (`EXP_020`/`EXP_021` vs. `EXP_008`/`EXP_009`).
- **Clean rejection-proxy false positive.** Under `P_krum=3`, with zero attacker present, one honest client is persistently, falsely confirmed at both seeds — always that seed's single largest Dirichlet partition (`client_004` at seed 123, `client_001` at seed 2026).
- **Clean-phase sample-count/Krum-rank correspondence.** Ascending sample count matches ascending Krum rank exactly, both seeds.
- **Gaussian attack: complete, immediate geometric exclusion.** 100% (30/30) post-attack rejection, instantaneous zero-lag regime switch at round 20, both seeds (`EXP_022`/`EXP_023`).
- **Directional attack: complete geometric evasion.** 0% (0/30) post-attack rejection, no regime switch at any point, both seeds (`EXP_024`/`EXP_025`).
- **Attack geometry reproducibly determines Multi-Krum's raw-distance exclusion outcome.** Across all four attacked runs (2 seeds × 2 families), magnitude-revealing attacks are fully excluded and norm-preserving attacks are fully evaded — a perfectly consistent 2×2 result (Part 11 of `analysis/MULTIKRUM_DIRECTIONAL_CROSS_SEED_SYNTHESIS.md`).
- **Full-run rejection proxy never correctly attributes the true attacker.** In every one of the four attacked runs, applying `P_krum=3` honestly (no onset foreknowledge) names an honest, largest-partition client first — for the Gaussian runs the true attacker is eventually, separately confirmed later (18 rounds after the false positive, both seeds); for the directional runs the true attacker is **never** confirmed at all, at either seed.
- **FedTrace v3 succeeds where the repurposed proxy fails, at every seed and every attack family tested.** Zero honest false positives and correct, well-localized attribution (onset error 0, delay 2) across all four already-validated Stage B runs (`EXP_016`–`EXP_019`), via `z_m` (Gaussian) or `z_t_res` (directional) — automatically, with no attack-specific tuning.
- **Under the directional attack, Multi-Krum substantially underperforms FedAvg**, both seeds (−8.74 pp seed 123, −10.06 pp seed 2026, Top-1), despite (in fact partly *because of*, via its unweighted-averaging convention) never excluding the attacker.
- **Under the Gaussian attack, attacked Multi-Krum outperforms clean Multi-Krum** (both seeds, direction only) **but still underperforms attacked FedAvg** (both seeds) — Multi-Krum never "wins" against FedAvg in any of the four attacked comparisons, clean-relative gains under Gaussian notwithstanding.

### SUPPORTED BUT LIMITED

- **The Gaussian attacked-vs-clean-Multi-Krum trajectory *shape*** replicates in final-outcome direction at both seeds but not in exact shape (immediate gain at seed 123; a brief early dip before crossover at seed 2026) — a genuine, reported difference, not smoothed over.
- **The unweighted-25%-vs-sample-weighted-~16–20% over-weighting mechanism** (Multi-Krum gives the directional attacker's poisoned update more relative influence than FedAvg would, whenever it is selected) is a **confirmed fact** about the code and selection counts, and a well-grounded, evidence-consistent explanation for why Multi-Krum's directional-attack damage exceeds FedAvg's — but the exact quantitative decomposition of how much of the total gap this contributes, versus the exclusion failure itself, was not isolated via a controlled ablation.
- **The raw-Euclidean-distance-is-magnitude-sensitive explanation** for the Gaussian/directional dichotomy is strongly, consistently supported by every trace-level observation made across all four attacked runs, but is not an independently, formally derived proof for this specific neural-network parameter space.

### HYPOTHESIS

- Whether intermediate attack strengths (a `directional_target_cosine` between 0 and 1, or a Gaussian `rho` below 1.0) would show a continuum between the two observed extremes, rather than another sharp dichotomy — untested, and not to be tested by tuning per every task's explicit restriction in this branch.
- The unequal-local-SGD-steps-under-fixed-`local_epochs`-across-differently-sized-partitions mechanism connecting sample count to clean-phase Krum rank — plausible, consistent with every observation, never directly measured (e.g., no raw per-client update-norm-vs-sample-count regression was performed in any task in this branch).
- Generalization beyond `n=5` clients, `α=0.1`, ResNet-18/CIFAR-100, and the two tested seeds.

### UNSUPPORTED — Claims Not to Be Made

- That Multi-Krum's native robust-aggregation function is defective in general. It is not — it performs a complete, verified, immediate exclusion function against the Gaussian family; it simply also fails completely against the directional family, and both outcomes are now well-characterized, not merely observed once.
- That FedTrace v3 would succeed against every conceivable attack, partition, or heterogeneity configuration — only that it succeeded, correctly, with zero false positives, on every one of the four attack-family/seed combinations actually tested.
- That the Gaussian attack causally "improves" learning — the observed attacked-vs-clean Multi-Krum gain is attributed to a rejection-target side effect changing which client's signal reaches the aggregate, never to the attack's content being beneficial.
- That any finding in this branch generalizes beyond the exact tested configuration (5 clients, `α=0.1`, `f=1`, these two attack parameterizations, these two seeds).
- That the specific *identity* of the falsely-rejected honest client, or of any other client-level result, is fixed or predictable in general — only the *structural role* (largest partition) is consistent across seeds.

## Part 14 — External-Baseline Branch Verdict

**Decision: A — the external Multi-Krum experiment matrix is complete; proceed to the final external-baseline synthesis / evidence-gap audit for publication.**

Justification: the pre-registered minimal matrix (`research_plans/MULTIKRUM_EXTERNAL_BASELINE_IMPLEMENTATION_PLAN.md` §8: two clean controls, four attacked runs, no more) is now fully executed, and every one of that plan's pre-registered scientific questions (§1–§17) has a direct, evidence-grounded answer, replicated across two seeds for every finding that a second seed could test:

- §1 scientific role (robust aggregation vs. repurposed proxy) — empirically demonstrated, not merely asserted, across four attacked runs.
- §6 Non-IID fairness risk — confirmed at both clean-control seeds and shown to persist, unresolved, inside every attacked run too.
- §9/§10 robustness metrics and rejection diagnostics — collected and analyzed for all six runs.
- §11 repurposed rejection proxy — applied exactly as pre-registered (never reset at onset, never tuned after seeing results) across all four attacked runs.
- §12 Table A/Table B separation — maintained throughout every analysis document in this branch.
- §16/§17 success criteria and publication-interpretation outcomes — resolved: the actual outcome realized is closest to outcome **C** (Multi-Krum struggles under this project's strong Non-IID heterogeneity, both in its clean-control fairness behavior and, for the directional family, in its native robustness function) for the directional family, and closer to outcome **A** (mitigates damage under Gaussian, but the rejection proxy tracks attribution poorly) for the Gaussian family — both pre-registered as legitimate, reportable outcomes before any run occurred, and neither is treated as more or less desirable than the other.

**No further Multi-Krum GPU experiment is recommended.** Per this task's explicit instruction, no additional run is proposed unless a specific pre-registered gap requires it, and none does: the plan's own explicit "do not run" list (§18: no additional seeds, no new attack families, no `f`/`m` sweeps, no FedTrace modifications, no new datasets/models) remains correctly unexplored, and every scientific question the six-run matrix was designed to answer now has an answer, replicated where replication was the design's own bar for a claim.

## Strongest Paper-Ready Finding

**The attack-geometry dichotomy, confirmed across a fully-crossed 2-seed × 2-attack-family design:** Multi-Krum's raw-Euclidean-distance robust-aggregation mechanism achieves complete, immediate exclusion (100%, zero-lag) of a magnitude-revealing attacker and complete, total evasion (0%, no exclusion at any point) by a norm-preserving, direction-only attacker — with FedTrace v3 succeeding identically, correctly, and with zero honest false positives against **both** attack families at **both** seeds, via a design that switches its dominant signal (`z_m` vs. `z_t_res`) automatically, without knowing the attack type in advance. This is the single most quantified, most reproducible, and most directly paper-relevant result the branch produced — it converts the project's abstract "robust aggregation ≠ forensic attribution" claim into a concrete, four-run, cross-seed-replicated demonstration with a clear, verified causal mechanism (raw distance responds to magnitude; a genuinely norm-preserving attack removes exactly the signal that mechanism depends on).

## Important Unsupported Claims (Repeated for Emphasis)

Per Part 13's UNSUPPORTED section: this branch does **not** show Multi-Krum is broadly defective, does **not** show FedTrace is universally robust, does **not** generalize beyond the tested seeds/attacks/configuration, and does **not** support any causal "the attack helped" reading of the Gaussian family's counter-intuitive performance result. Every claim in this document and its component analyses is scoped to exactly what was measured.

---

## Final Report (Covers All Three Files Created in This Task)

1. **Files created:** `analysis/EXP_025_MULTIKRUM_DIRECTIONAL_SEED2026_ANALYSIS.md`, `analysis/MULTIKRUM_DIRECTIONAL_CROSS_SEED_SYNTHESIS.md`, `analysis/MULTIKRUM_EXTERNAL_BASELINE_FINAL_SYNTHESIS.md`.
2. **EXP_025 pairing verdict:** valid — differs from EXP_019 only by `aggregation: fedavg→multikrum`; identical seed-2026 partition to EXP_021.
3. **Attack correctness:** confirmed genuinely norm-preserving (`delta_attack_norm_ratio≈1.0`, max deviation ≈2.1×10⁻¹⁰) and correctly targeted (`realized_cosine_to_honest≈0.0`, max deviation ≈7.5×10⁻¹⁰), all 30 active rounds.
4. **Diagnostics integrity:** pass — 250/250 rows, fully structurally valid.
5. **Pre/post rejection counts:** `client_001`=19 pre/**30 post**; `client_004`=1 pre/0 post; **`client_002`=0 pre/0 post**; `client_000`/`client_003`=0/0.
6. **`client_002` post-attack rejection/selection result:** **0/30 rejected (0%), 30/30 selected (100%)** — complete evasion, replicating `EXP_024`.
7. **Attacker rank behavior:** constant rank 3, zero variation across the entire 50-round run.
8. **Rejection-switch result:** none — `client_001` remains rejected every round, uninterrupted by the attack.
9. **Full-run `P_krum` attribution:** misattributes `client_001` (confirmed round 4, identical timing to `EXP_024`'s honest FP); `client_002` never confirmed at either seed.
10. **EXP_025 vs. EXP_021 metric differences:** −6.79 pp Top-1, −7.36 pp Top-5 (attacked worse than clean).
11. **EXP_025 vs. EXP_019 metric differences:** −10.06 pp Top-1, −11.61 pp Top-5 (Multi-Krum much worse than FedAvg).
12. **Seed-2026 2×2:** clean-agg −6.97 pp; attack-under-FedAvg −3.70 pp; attack-under-MK −6.79 pp; aggregation-under-attack −10.06 pp (Top-1) — all four effects match seed 123 in sign.
13. **EXP_024/025 directional replication verdict:** all six tested seed-123 findings (A–F) **fully replicate** at seed 2026 — the cleanest, most uniform two-seed confirmation in the whole branch.
14. **Gaussian-vs-directional cross-seed result:** a perfectly consistent 2×2 dichotomy — 100% attacker exclusion under Gaussian, 0% under directional, at both seeds, with no exception.
15. **FedTrace-vs-Multi-Krum interpretation:** FedTrace succeeds identically, correctly, with zero honest FP at all four attack-family/seed combinations; the Multi-Krum rejection proxy misattributes an honest client at all four, and never confirms the true attacker at all in the directional pair — the strongest evidentiary basis in this program for the robust-aggregation-≠-forensic-attribution claim.
16. **Fairness/heterogeneity finding:** the falsely-rejected honest client is always that seed's single largest partition, across all six runs (clean, Gaussian, and directional alike) — reported as a replicated association, not a proven causal claim.
17. **External-baseline final verdict:** **A** — the six-run matrix is complete; proceed to final synthesis for publication; no further Multi-Krum GPU experiment is recommended.
18. **Strongest paper-ready finding:** the attack-geometry dichotomy (100% vs. 0% attacker exclusion, both seeds, both attack families) paired with FedTrace's consistent, unaffected success across the same four conditions.
19. **Important unsupported claims:** Multi-Krum is not shown to be broadly defective; FedTrace is not shown to be universally robust; no finding generalizes beyond the tested seeds/attacks/configuration; the Gaussian family's counter-intuitive performance gain is not a causal "attack helps" effect.
20. **Limitations:** two seeds only; one attack strength per family (`rho=1.0`, `target_cosine=0.0`); no controlled ablation isolating the unweighted-averaging-vs-exclusion-failure contributions; the partition-size/Krum-rank mechanism remains unmeasured; no statistical significance claimed from n=2.
21. **Next gate:** **A** — branch complete; no further Multi-Krum GPU experiment recommended unless a specific new pre-registered gap arises.
22. **Confirmation no implementation/config/test/result changed:** confirmed — `src/federated/aggregation/multikrum.py`, `src/analysis/fedtrace_detector.py`, `src/attacks/traceability.py`, all configs, all tests, `README.md`, Research_Database, and every existing analysis/result file were untouched; only pre-existing result artifacts and prior analysis/plan documents were read.
23. **Confirmation no experiment was run:** confirmed — no training script was executed.
24. **Confirmation whether commands were executed:** none — this task used only read-only file inspection (`Read`) and file-pattern search (`Glob`) tool calls; no shell, PowerShell, Bash, or Python command was executed.
