# Multi-Krum Clean-Control Cross-Seed Synthesis (EXP_020 + EXP_021)

**Status:** Analysis document only. No code, test, config, README, FedTrace, attack, or Research Database file was created or modified in producing it. No experiment was run. Multi-Krum's `f=1`/`m=4`, tie-breaking, weighting, buffer handling, and the pre-registered `P_krum=3` rejection proxy were not modified or tuned regardless of what these two runs showed. FedTrace v3 remains frozen. Multi-Krum's exclusion pattern is referred to throughout as the **"Multi-Krum rejection proxy"**, never as an attribution detector — it is an external, literature-faithful Byzantine-robust *aggregation* baseline whose selection/rejection behavior is being repurposed, secondarily, as a traceability proxy (`research_plans/MULTIKRUM_EXTERNAL_BASELINE_IMPLEMENTATION_PLAN.md` §1, §11).

---

## Comparison Table

| | Seed 123 | Seed 2026 |
|---|---|---|
| FedAvg baseline | EXP_008 | EXP_009 |
| Multi-Krum experiment | EXP_020 | EXP_021 |
| FedAvg test Top-1 | 56.83% | 58.26% |
| Multi-Krum test Top-1 | 49.91% | 51.29% |
| **Top-1 clean cost** | **−6.92 pp** | **−6.97 pp** |
| FedAvg test Top-5 | 87.79% | 88.56% |
| Multi-Krum test Top-5 | 80.06% | 79.62% |
| **Top-5 clean cost** | **−7.73 pp** | **−8.94 pp** |
| FedAvg test loss | 1.4654 | 1.4466 |
| Multi-Krum test loss | 2.1901 | 2.0016 |
| Test-loss delta | +0.7247 | +0.5550 |
| Best validation Top-1 clean cost | −6.82 pp | −6.82 pp |
| Persistently rejected client | `client_004` | `client_001` |
| Rejection count (of 50 rounds) | 49 (98%) | 49 (98%) |
| Longest consecutive rejection streak | 48 rounds (R2–R49) | 48 rounds (R2–R49) |
| Clean persistent false positive (`P_krum=3`) | **Yes** (onset R2, confirmed R4) | **Yes** (onset R2, confirmed R4) |
| Rejected client's sample-count rank (of 5) | 5th (largest — 11,785) | 5th (largest — 12,318) |
| Most-trusted client's (rank-1) sample-count rank | 1st (smallest — 6,024) | 1st (smallest — 5,233) |

## Replication of the Candidate Observation

**Candidate observation:** *"Under strong Dirichlet alpha=0.1 Non-IID heterogeneity, clean Multi-Krum systematically rejects one honest client for nearly the entire training trajectory."*

**Verdict: supported across both tested seeds.** In both EXP_020 (seed 123) and EXP_021 (seed 2026), one specific honest client is rejected in 49 of 50 rounds (98%), including an identical-length unbroken 48-round streak (rounds 2–49) in both runs, with the single exception in both cases occurring at round 1 (an early-training instability common to both seeds, discussed below). The magnitude, shape, and even the exact streak boundaries of this behavior are essentially identical across the two seeds.

**This is evidence of replication across these two specific tested seeds — it is *not* evidence of universal behavior across all seeds, partitions, or configurations.** With `n=2` seeds, no claim is made or supportable about what fraction of possible Dirichlet(α=0.1, 5-client) partition realizations would show this pattern, how it would behave at a different `α`, a different number of clients, or a different model/dataset. The observation is reported as "supported (n=2)," not generalized further.

## Does the Rejected Client's Identity Change Across Seeds?

**Yes — the identity changes** (`client_004` at seed 123 vs. `client_001` at seed 2026). Taken alone, this might look like it weakens the "systematic rejection" finding (different client each time). It does the opposite once the partition data is examined:

- At seed 123, the rejected client (`client_004`) holds the **largest** sample count of that seed's five clients (11,785 of 45,000 total).
- At seed 2026, the rejected client (`client_001`) also holds the **largest** sample count of that seed's five clients (12,318 of 45,000 total).
- In **both** seeds, the client Multi-Krum trusts most consistently (Krum-rank 1 in every single round) holds the **smallest** sample count of that seed (`client_000`: 6,024 at seed 123, 5,233 at seed 2026).
- In **both** seeds, the full Krum-rank ordering (1→5, most-trusted to most-rejected) matches the ascending sample-count ordering of that seed's five clients **exactly** — a perfect rank correspondence, independently, in each of the two runs.

**Why this matters scientifically:** the rejected client's *identity* is not fixed — it is whichever client happens to receive the largest Dirichlet(α=0.1) partition share under that particular seed's random draw. This strongly suggests the Multi-Krum rejection proxy, in this project's exact experimental configuration, is tracking a **structural artifact of unequal partition size** rather than any client-specific behavioral signature. A plausible (not directly measured in either analysis task, and not asserted as proven) mechanism: every client trains for a fixed `local_epochs=1` regardless of partition size, so a client with more local samples performs more mini-batch SGD steps within that one epoch than a smaller-partition client does, mechanically producing a larger-magnitude parameter update in Euclidean space — exactly the quantity Krum's raw distance metric penalizes — with no relationship to malicious intent. Confirming this mechanism directly (e.g., by measuring per-client raw update norms, or by re-running with a data-volume-normalized local-training schedule) is outside the scope of this read-only analysis task and is not claimed here as demonstrated; it is offered only as the most parsimonious explanation consistent with the data actually observed in both runs.

## Part 6 — External-Baseline Fairness Interpretation

Per `research_plans/FEDTRACE_EXTERNAL_BASELINE_SELECTION_PLAN.md` and `research_plans/MULTIKRUM_EXTERNAL_BASELINE_IMPLEMENTATION_PLAN.md` §1/§5:

- **FedTrace v3** is a purpose-built, frozen attribution/anomaly-detection method. It requires no assumption about how many clients (if any) are malicious, and its calibration is computed purely from each client's own historical/cross-sectional signal statistics.
- **Multi-Krum** is an external, literature-faithful Byzantine-robust *aggregation* method. It is given a known/assumed Byzantine bound (`f=1`) that FedTrace is never given and does not need (§5 of the implementation plan — an important, already-disclosed fairness asymmetry). Its rejection pattern is being repurposed, secondarily, as a traceability proxy — not a native capability of the method.

**Does the clean-control evidence reveal an important limitation of the Multi-Krum rejection proxy under strong Non-IID heterogeneity?**

**Yes, within this experimental setting specifically.** Across both tested seeds, with zero attacker present, the pre-registered `P_krum=3` rejection proxy produces a confident, sustained, unanimous "prediction" — and in both cases that prediction is simply the client that happened to receive the most training data that seed, not a client exhibiting any actual anomalous behavior (there is none to exhibit, by construction, in a clean run). This is precisely the fairness risk `research_plans/MULTIKRUM_EXTERNAL_BASELINE_IMPLEMENTATION_PLAN.md` §6 pre-registered before any run occurred ("honest clients' updates are themselves naturally far apart in Euclidean update-space... Multi-Krum may reject a genuinely honest client instead of, or in addition to, the true attacker"), now empirically confirmed, consistently, at both tested seeds.

**What this does *not* show:**
- It does **not** show that Multi-Krum's native robust-aggregation function is defective — that is a separate question (Table B of the plan's §12 comparison boundary; the clean-performance-cost findings above bear on it, but robustness/mitigation *under attack* has not yet been evaluated at all, since no attacked Multi-Krum run has been created).
- It does **not** show that FedTrace v3 is broadly or generally superior to Multi-Krum — only that, in the exact experimental setting tested here (5 clients, strong Non-IID Dirichlet α=0.1, fixed `local_epochs=1` across unequal partition sizes, no attacker), the *repurposed* rejection proxy specifically is dominated by a partition-size artifact rather than tracking anything resembling misbehavior. FedTrace v3 was purpose-built and independently validated for exactly this task (`analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md`); Multi-Krum's rejection pattern was never designed to answer it, and this finding is fully consistent with — not a surprising contradiction of — that framing.
- It does **not** generalize beyond this configuration (5 clients, this α, this model, `local_epochs=1`, these two seeds).

**Practical implication for the eventual attacked-run comparison (not performed in this task):** because the rejection proxy already, reliably, targets the largest-partition honest client under clean conditions, an attacked Multi-Krum run in which the malicious client (`client_002`, per the frozen attack configs) is *not* the largest-partition client risks the rejection proxy continuing to flag the largest honest client instead of, or in addition to, the true attacker — a confound that must be checked explicitly once attacked Multi-Krum results exist, not assumed away in either direction.

## Part 7 — Gate Decision

**Decision: A — the two-seed clean-control gate is complete; proceed to attacked Multi-Krum experiments.**

Justification: both required clean controls (seed 123 ↔ EXP_008, seed 2026 ↔ EXP_009) are complete, technically valid (diagnostics structurally intact and internally consistent in both runs — `analysis/EXP_020_CLEAN_MULTIKRUM_SEED123_ANALYSIS.md` §2/8-D and this document's Part 2 above), and scientifically interpretable. The findings are consistent, substantial, and negative for both Multi-Krum's clean-performance cost and its rejection-proxy fairness — but per this task's explicit instruction and the plan's own Section 15 ("scientific failure vs. implementation failure"), poor clean performance and a confirmed clean-run rejection-proxy false positive are retained as scientific results, not treated as blockers.

**Exact next experiment recommended (not created in this task):** the Multi-Krum analog of **EXP_016** — Gaussian update-noise attack (`client_002`, round-20 onset, `rho=1.0`), Dirichlet(α=0.1), **seed 123**, paired against both `EXP_008` (honest FedAvg) and `EXP_016` itself (attacked FedAvg), with `aggregation: multikrum` in place of `fedavg` and every other field identical to `EXP_016`/`EXP_008`. This is the first entry of the frozen plan's minimal attacked-run matrix (`research_plans/MULTIKRUM_EXTERNAL_BASELINE_IMPLEMENTATION_PLAN.md` §8: Gaussian×123, Gaussian×2026, directional×123, directional×2026, in that order), matching the seed-123-first convention already used for both the original Stage B sequence (`EXP_016` before `EXP_017`) and these two clean controls (`EXP_020` before `EXP_021`). No new seed, attack family, or `f`/`m` sweep is introduced; FedTrace v3, the attack code, and the dataset/model remain unchanged, per the plan's explicit scope-control (§18).

---

## Final Report

1. **Files created:** `analysis/EXP_021_CLEAN_MULTIKRUM_SEED2026_ANALYSIS.md`, `analysis/MULTIKRUM_CLEAN_CONTROL_CROSS_SEED_SYNTHESIS.md`.
2. **Files modified:** none.
3. **EXP_021 pairing verdict:** valid — no discrepancy found against EXP_009 on any required field; sole scientific difference is `aggregation: fedavg → multikrum`.
4. **EXP_021 diagnostics-integrity verdict:** pass — 250/250 expected rows, 50 rounds × 5 clients, ranks 1–5 exactly once per round, exactly 4 selected/1 rejected every round, no missing or malformed rows.
5. **Rejection count per client (EXP_021, of 50):** `client_000`=0, `client_002`=0, `client_003`=0, `client_004`=1, `client_001`=49.
6. **Longest rejection streak per client (EXP_021):** 0, 0, 0, 1, `client_001`=48 consecutive rounds (R2–R49).
7. **EXP_021 clean persistent-FP verdict:** Yes.
8. **Predicted suspicious client under rejection proxy (EXP_021):** `client_001`.
9. **First rejection / confirmation / onset (EXP_021):** first rejected round 0; onset round 2; confirmed round 4; persists continuously through round 49.
10. **EXP_021 vs. EXP_009 test Top-1 difference:** −6.97 pp.
11. **Top-5 difference:** −8.94 pp.
12. **Best-validation difference:** −6.82 pp (51.26%@r47 vs. 58.08%@r49).
13. **Other metric differences (EXP_021):** test loss +0.5550, macro precision −8.51 pp, macro F1 −8.89 pp, balanced accuracy −6.97 pp.
14. **EXP_021 clean-performance-cost verdict:** substantial.
15. **Seed-2026 partition/sample-count interpretation:** the persistently rejected client (`client_001`) has the largest sample count (12,318) of the five; the most-trusted (rank-1) client (`client_000`) has the smallest (5,233) — reported as an association with a plausible (unconfirmed) mechanism (unequal local SGD steps under fixed `local_epochs=1`), not a proven causal claim.
16. **EXP_020 + EXP_021 cross-seed rejection result:** replicates closely — 49/50 rejections and an identical 48-round streak (R2–R49) at both seeds; rejected-client *identity* differs (`client_004` vs. `client_001`) but in both cases is exactly that seed's largest-sample-count client, and the rank-1 (most-trusted) client is exactly that seed's smallest-sample-count client in both runs, with a perfect Krum-rank/sample-count-rank correspondence at both seeds.
17. **Cross-seed clean-performance-cost result:** replicates — Top-1 cost −6.92 pp (seed 123) vs. −6.97 pp (seed 2026); Top-5 cost −7.73 pp vs. −8.94 pp; best-validation cost −6.82 pp at both seeds.
18. **Whether the candidate observation is supported:** supported across both tested seeds (n=2) — explicitly not claimed as evidence of universal behavior across all seeds/configurations.
19. **Main fairness implication for the Multi-Krum rejection proxy:** in this exact experimental setting, the proxy's clean-run "prediction" tracks partition size (largest-sample client), not client behavior — a real, empirically-confirmed limitation of using it as an attribution proxy here, not a claim that Multi-Krum's aggregation function is defective in general or that FedTrace is broadly superior.
20. **Clean-control gate decision:** **A** — proceed to attacked Multi-Krum experiments.
21. **Exact next experiment recommended:** the Multi-Krum analog of EXP_016 — Gaussian update-noise attack, seed 123, paired with EXP_008/EXP_016, `aggregation: multikrum`, all else identical. **Not created in this task.**
22. **Confirmation Multi-Krum unchanged:** confirmed — `src/federated/aggregation/multikrum.py` was not opened or touched; `f=1`/`m=4` untouched.
23. **Confirmation rejection proxy/`P_krum` unchanged:** confirmed — `P_krum=3` applied exactly as pre-registered in both EXP_020's and this analysis, no alternate threshold considered or used.
24. **Confirmation FedTrace v3 unchanged:** confirmed — `src/analysis/fedtrace_detector.py` was not opened or touched.
25. **Confirmation attack code unchanged:** confirmed — `src/attacks/traceability.py` was not opened or touched.
26. **Confirmation no experiment/config/code/test/database changed:** confirmed — only pre-existing result artifacts (`summary.json`, `final_test_metrics.json`, `round_history.csv`, `multikrum_diagnostics.csv`, `client_partition_summary.csv` for EXP_021/EXP_009) and prior analysis/plan documents were read; no config, code, test, README, or Research Database file was created, edited, or run.
27. **Confirmation whether any commands were executed:** none — this task used only read-only file inspection (`Read`) and file-pattern search (`Glob`) tool calls; no shell, PowerShell, Bash, or Python command was executed.
