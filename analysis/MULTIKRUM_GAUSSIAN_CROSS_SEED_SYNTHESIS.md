# Multi-Krum Gaussian Attack Cross-Seed Synthesis (EXP_022 + EXP_023)

**Status:** Analysis/synthesis document only. No code, test, config, README, FedTrace, attack, or Research Database file was created or modified in producing it. No experiment was run. Multi-Krum's `f=1`/`m=4`, tie-breaking, weighting, buffer handling, and the pre-registered `P_krum=3` rejection proxy were not modified or tuned. FedTrace v3 remains frozen. Multi-Krum's exclusion pattern is referred to throughout as the **"Multi-Krum rejection proxy"**, never as an attribution detector.

---

## Part 8 — Cross-Seed Comparison Table

| | Seed 123 | Seed 2026 |
|---|---|---|
| Clean Multi-Krum control | EXP_020 | EXP_021 |
| Attacked Multi-Krum experiment | EXP_022 | EXP_023 |
| Attacked FedAvg baseline | EXP_016 | EXP_017 |
| Pre-attack persistently rejected honest client | `client_004` | `client_001` |
| Pre-attack rejection count (of 20) | 19 (95%) | 19 (95%) |
| Malicious `client_002` post-attack rejection count (of 30) | **30 (100%)** | **30 (100%)** |
| Rejection-switch round | **20** (attack onset, lag 0) | **20** (attack onset, lag 0) |
| Malicious client's Krum rank post-attack | 5 (rejected), every round | 5 (rejected), every round |
| Full-run rejection-proxy predicted client | `client_004` | `client_001` |
| Proxy attribution correct? | **No** (honest client) | **No** (honest client) |
| Proxy honest-FP confirmation round | 4 | 4 |
| True attacker's own confirmation round | 22 | 22 |
| Attacked Multi-Krum test Top-1 | 54.43% | 54.78% |
| Attacked FedAvg test Top-1 | 56.50% | 58.08% |
| Attacked − clean Multi-Krum (Top-1) | **+4.52 pp** | **+3.49 pp** |
| Attacked Multi-Krum − attacked FedAvg (Top-1) | **−2.07 pp** | **−3.30 pp** |

## Replication Classification

**A. One honest client dominates pre-attack rejection.** Seed 123: `client_004`, 19/20 (95%). Seed 2026: `client_001`, 19/20 (95%). **Replicated across both tested seeds** — identical rejection rate, different client identity (see Part 10 for why the identity difference is itself informative, not a failure to replicate).

**B. Attacker takes over rejection exactly at round 20.** Both seeds: zero-lag switch, verified round-by-round in each run's own Part 3 analysis. **Replicated across both tested seeds.**

**C. `client_002` rejected 30/30 post-attack rounds.** Both seeds: exactly 30/30, no exception, in either run. **Replicated across both tested seeds.**

**D. Full-run rejection proxy still misattributes due to an earlier honest false positive.** Both seeds: the honest largest-partition client is confirmed at round 4; `client_002` is confirmed at round 22; the honest false positive precedes the true attacker's confirmation by 18 rounds, in both runs, with the confirmation rounds themselves numerically identical. **Replicated across both tested seeds** — and, per each run's own analysis (`EXP_022`/`EXP_023` §4/Part 4), this exactness follows from shared structural facts (a round-1 crossover artifact in the pre-attack rejection streak, and the zero-lag switch at round 20) present at both seeds, not coincidence.

**E. Attacked Multi-Krum exceeds clean Multi-Krum.** Both seeds: positive (+4.52 pp seed 123, +3.49 pp seed 2026 on Top-1; +2.53 pp / +4.65 pp on Top-5). **The direction replicates across both tested seeds.** The *mechanism by which the gap emerges* does not replicate identically: at seed 123 the attacked run is ahead of the clean control from round 20 onward with no dip; at seed 2026 the attacked run briefly *trails* the clean control for several rounds around onset before crossing over near round 30 (`EXP_023` §5). **Classified as: replicated in final-outcome direction; partially replicated in trajectory shape.**

**F. Attacked Multi-Krum remains below attacked FedAvg.** Both seeds: negative (−2.07 pp seed 123, −3.30 pp seed 2026 on Top-1). **Replicated across both tested seeds**, with a somewhat larger gap at seed 2026.

**No universal claim is made from n=2 seeds.** Every "replicated" label above means "replicated across these two specific tested seeds, under this exact attack family, strength, partition scheme, and client count" — not "true in general."

## Part 9 — FedTrace vs. Multi-Krum, Both Seeds

Using the already-frozen, already-validated FedTrace v3 results for the paired FedAvg runs (not re-run on the Multi-Krum traces in either analysis task):

| | FedTrace v3, `EXP_016` (seed 123) | FedTrace v3, `EXP_017` (seed 2026) | Multi-Krum rejection proxy, full run, both seeds |
|---|---|---|---|
| Predicted client | `client_002` | `client_002` | Honest largest-partition client (`client_004` / `client_001`) — **wrong at both seeds** |
| Onset estimate | 20 (error 0) | 20 (error 0) | 2 (the honest false positive's own onset) |
| Confirmation round | 22 | 22 | 4 (the honest false positive) — 18 rounds before `client_002`'s own confirmation (round 22) at both seeds |
| Detection delay | 2 | 2 | Not applicable — the reported client is wrong |
| Honest persistent false positives | **0** | **0** | **1 at each seed** (the largest-partition client) |
| Requires known/assumed `f`? | No | No | Yes (`f=1`, an input this project's own single-attacker threat model happens to match — a disclosed asymmetry, not a general property of attribution methods) |
| Clean-run behavior | Silent — no client ever crosses threshold | Silent — no client ever crosses threshold | Confidently, persistently flags one honest client |

**Does the two-seed Gaussian Multi-Krum evidence strengthen the claim that robust exclusion is not equivalent to forensic attribution?**

**Yes.** A single run (`EXP_022`) showing this misattribution could plausibly have been dismissed as a property of one particular partition draw. **Two independent seeds now show not just the same qualitative failure (misattributing the largest-partition honest client ahead of the true attacker) but the same quantitative mechanism** — an identical pre-attack confirmation round (4) and an identical true-attacker confirmation round (22) at both seeds, arising from the same two structural facts (a round-1 rejection-pattern artifact and a zero-lag round-20 regime switch) independently verified at each seed. This is meaningfully stronger evidence than either run alone: it shows the failure mode is tied to a structural property of how Multi-Krum's per-round, memoryless scoring interacts with this project's fixed `local_epochs=1`/unequal-partition-size heterogeneity, not an artifact specific to one seed's random draw. **This conclusion is scoped exactly as stated — two seeds, one attack family, one attack strength, one partition scheme — and is not extended further.**

## Part 10 — Fairness / Heterogeneity, Both Seeds

**Sample-count orderings (ascending), from each run's own `client_partition_summary.csv`:**

- **Seed 123:** `client_000` (6,024) < `client_002` (7,202) < `client_001` (9,628) < `client_003` (10,361) < `client_004` (11,785, largest).
- **Seed 2026:** `client_000` (5,233) < `client_003` (6,556) < `client_002` (9,171) < `client_004` (11,722) < `client_001` (12,318, largest).

**Is the pre-attack rejected honest client again the largest-sample client?** **Yes, at both seeds** — `client_004` (seed 123) and `client_001` (seed 2026) are each their own seed's single largest partition. The *identity* of the affected client changes (as it must, since it is determined by that seed's random partition draw, not by any fixed client label) — what replicates is the **structural role**: whichever client happens to hold the largest partition that seed is the one falsely, persistently rejected.

**Does the clean-phase sample-count/Krum-rank correspondence repeat?** **Yes, exactly, at both seeds** — the ascending Krum-rank ordering during rounds 0–19 matches the ascending sample-count ordering perfectly in both `EXP_022` (`client_000<client_002<client_001<client_003<client_004`) and `EXP_023` (`client_000<client_003<client_002<client_004<client_001`). **This is reported as a replicated association across both tested seeds — not a causal claim.** The candidate mechanism offered in both individual analyses (unequal local SGD steps per round under fixed `local_epochs=1` across differently-sized partitions) remains a hypothesis, not directly measured in either task.

**One additional, honestly-reported detail:** the malicious client's own sample-count rank is *not* consistent across seeds — `client_002` is the 2nd-smallest partition at seed 123 (7,202) but the median (3rd of 5) partition at seed 2026 (9,171). The attack's dominance of the post-round-20 ranking (Part 3/Regime Switch, both runs) does not depend on the attacker's own partition size — a magnitude-scaled (`rho=1.0`) perturbation overrides the clean-phase, partition-size-driven ordering regardless of where the attacker started in that ordering.

## Part 11 — Paper-Level Interpretation

**CONFIRMED (directly supported by both seeds):**
- One honest client — specifically, that seed's largest-partition client — dominates pre-attack rejection under clean, strong Non-IID heterogeneity (19/20 rounds, 95%, at both seeds).
- The rejection target switches to the true attacker with zero lag, exactly at attack onset, at both seeds.
- The malicious client is rejected in 100% of post-attack rounds at both seeds.
- The full-run Multi-Krum rejection proxy, applied honestly (without foreknowledge of the true onset — the only fair way to apply it), misattributes the honest largest-partition client ahead of the true attacker at both seeds, with an identical 18-round margin.
- FedTrace v3 correctly attributes `client_002` with zero honest false positives at both seeds (already independently Stage-B-validated; cited here, not re-derived).
- Attacked Multi-Krum underperforms attacked FedAvg on every metric at both seeds, despite perfect malicious-client exclusion.
- The clean-phase sample-count/Krum-rank correspondence (exact ascending match) holds at both seeds.

**SINGLE-SEED (supported in only one seed, or not identically replicated):**
- The exact trajectory shape of the attacked-vs-clean Multi-Krum improvement (immediate at seed 123; delayed with an early dip at seed 2026) — the *direction* is confirmed at both seeds, but this specific shape is not.
- The exact magnitude of every metric delta (all deltas replicate in direction and rough scale, not to the percentage point).

**HYPOTHESIS (mechanistic explanation, not demonstrated):**
- Unequal local SGD steps under fixed `local_epochs=1` across differently-sized partitions as the cause of the sample-count/Krum-rank association.
- The exclusion-target switch (from the largest honest client to the attacker) changing the effective training mixture as the cause of the attacked-vs-clean Multi-Krum performance gain.
- Raw Euclidean distance's sensitivity to magnitude (vs. normalized cosine similarity's insensitivity to it) as the reason Multi-Krum separates this magnitude-revealing attacker geometrically where the naive peer-cosine baseline detector does not (established with concrete trace numbers in `EXP_022`'s Part H; not independently re-derived for `EXP_023` in this task).

**UNSUPPORTED (claims that must not be made from this evidence):**
- That Multi-Krum's robust-aggregation function is defective in general — only shown here to underperform FedAvg in a setting where the attack itself causes little damage and Multi-Krum's own structural cost is large; a stronger or more damaging attack could change this comparison entirely, and was not tested.
- That any of these findings generalize beyond Gaussian `rho=1.0`, Dirichlet `α=0.1`, 5 clients, `local_epochs=1`, one non-colluding attacker, or beyond the two tested seeds.
- That FedTrace v3 would succeed under every possible attack, heterogeneity, or partition configuration.
- That the Gaussian attack causally "improves" learning — the observed improvement is attributed to a rejection-target side effect (hypothesis, above), never to the attack's content being beneficial.
- That the *identity* of the affected client is fixed or predictable in general — only the structural role (largest partition) replicates, not a specific client label.

**Does this two-seed Gaussian evidence strengthen FedTrace's paper contribution? Yes, specifically as follows:** the project's central claim — that attribution (which client, since when) is a distinct problem from robust aggregation (which updates to exclude this round), and that a repurposed exclusion pattern from the latter does not substitute for a purpose-built solution to the former — now rests on two independent, seed-diverse demonstrations of the *identical* failure mechanism (an 18-round misattribution margin, structurally reproduced) rather than one. This is a meaningfully stronger evidentiary basis for that claim than the single-seed result alone provided, without overselling it: it remains scoped to the Gaussian attack family, `rho=1.0`, and the two tested seeds, and it says nothing about Multi-Krum's native robust-aggregation task in general or about the still-untested directional attack family.

## Part 12 — Next Gate

**Decision: A — the Gaussian Multi-Krum branch is validated (both seeds show consistent, well-understood, mechanistically-explained results); proceed to the directional Multi-Krum experiment at seed 123.**

Justification: both Gaussian-family Multi-Krum runs are technically valid (diagnostics fully intact, pairing confirmed, no implementation or methodological problem found in either), and the six specific replication questions (A–F) all resolve with clear, consistent, two-seed answers — either fully replicated or explicitly and honestly flagged as partially replicated (E's trajectory shape). There is no open technical question that a third Gaussian-seed run would resolve that isn't already answered; the next source of information gain is a different attack family, exactly as the frozen plan's own minimal matrix specifies next (Gaussian×123, Gaussian×2026 — both now complete — then directional×123, directional×2026).

**Exact next experiment recommended: the Multi-Krum analog of `EXP_018`** — norm-preserving directional attack, `client_002`, round-20 onset, seed 123, Dirichlet `α=0.1`, paired against `EXP_008` (clean FedAvg), `EXP_018` (attacked FedAvg, directional, seed 123), and `EXP_020` (clean Multi-Krum, seed 123). This is the highest-information-gain next step: the directional attack is norm-*preserving* rather than magnitude-revealing, and `EXP_022`'s own trace analysis (Part H) offered a specific, falsifiable hypothesis — that Multi-Krum's raw Euclidean-distance metric separates the Gaussian attacker primarily because that attack inflates magnitude — which a norm-preserving attack directly tests: if Multi-Krum's rejection proxy fails to isolate the attacker once magnitude is held constant, that would be a materially different (and highly informative) finding from anything the Gaussian-family runs could show. **Its config is not created in this task.**

---

## Final Report (Covers Both Files in This Task)

1. **Files created:** `analysis/EXP_023_MULTIKRUM_GAUSSIAN_SEED2026_ANALYSIS.md`, `analysis/MULTIKRUM_GAUSSIAN_CROSS_SEED_SYNTHESIS.md`.
2. **EXP_023 pairing verdict:** valid — differs from EXP_017 only by `aggregation: fedavg→multikrum`; identical partition to EXP_021.
3. **Diagnostics integrity:** pass — 250/250 rows, fully structurally valid.
4. **Pre/post rejection counts per client:** `client_001`=19 pre/0 post; `client_004`=1 pre/0 post; `client_002`=0 pre/**30 post**; `client_000`/`client_003`=0/0.
5. **Regime-switch result:** instantaneous, zero-lag, round 20 exactly — replicates `EXP_022`.
6. **Full-run rejection-proxy result:** misattributes `client_001` (round 4), 18 rounds before `client_002`'s own confirmation (round 22) — replicates `EXP_022` exactly, including the confirmation rounds.
7. **EXP_023 vs. EXP_021 metric differences:** +3.49 pp Top-1, +4.65 pp Top-5 (attacked > clean) — replicates direction; trajectory shape differs.
8. **EXP_023 vs. EXP_017 metric differences:** −3.30 pp Top-1, −3.80 pp Top-5 (Multi-Krum < FedAvg under attack) — replicates `EXP_022`.
9. **Seed-2026 2×2 result:** clean aggregation cost −6.97 pp Top-1; attack-under-FedAvg −0.18 pp; attack-under-Multi-Krum +3.49 pp; aggregation-under-attack −3.30 pp — same directions as seed 123, broadly similar magnitudes.
10. **EXP_022+023 cross-seed replication verdict:** A/B/C/D/F fully replicated across both seeds; E replicated in direction only (trajectory shape differs) — see Part 8 table and classification.
11. **FedTrace vs. Multi-Krum interpretation:** FedTrace correctly attributes `client_002` with zero honest FP at both seeds; the Multi-Krum rejection proxy misattributes the honest largest-partition client at both seeds, with an identical structural mechanism and 18-round margin — a materially stronger, two-seed basis for the "robust exclusion ≠ forensic attribution" claim than either run alone.
12. **Fairness/heterogeneity result:** the pre-attack falsely-rejected client is the largest-partition client at both seeds (different identities, same structural role); the sample-count/Krum-rank correspondence is exact at both seeds — reported as a replicated association, not a causal claim.
13. **Paper-level contribution verdict:** strengthened, specifically and narrowly — two independent seeds now show the identical misattribution mechanism, not one; see Part 11's CONFIRMED/SINGLE-SEED/HYPOTHESIS/UNSUPPORTED breakdown for exact scope.
14. **Limitations:** two seeds only; one attack family (Gaussian) and one attack strength (`rho=1.0`); FedTrace v3 was not re-run on either Multi-Krum trace (the paired FedAvg runs' already-frozen results were used); all mechanistic explanations remain hypotheses; no statistical significance claimed from n=2.
15. **Next-gate decision:** **A** — Gaussian branch validated; proceed to directional Multi-Krum, seed 123.
16. **Exact next experiment recommended:** the Multi-Krum analog of `EXP_018` (norm-preserving directional attack, `client_002`, round-20 onset, seed 123, paired with `EXP_008`/`EXP_018`/`EXP_020`). **Not created in this task.**
17. **Confirmation no implementation/config/test/result was changed:** confirmed — `src/federated/aggregation/multikrum.py`, `src/analysis/fedtrace_detector.py`, `src/attacks/traceability.py`, all configs, all tests, `README.md`, Research_Database, and every existing analysis file were untouched; only pre-existing result artifacts and prior analysis/plan documents were read.
18. **Confirmation no experiment was run:** confirmed — no training script was executed.
19. **Confirmation whether commands were executed:** none — this task used only read-only file inspection (`Read`) and file-pattern search (`Glob`) tool calls; no shell, PowerShell, Bash, or Python command was executed.
