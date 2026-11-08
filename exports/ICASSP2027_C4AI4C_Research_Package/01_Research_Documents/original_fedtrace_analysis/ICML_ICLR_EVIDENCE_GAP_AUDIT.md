# FedTrace-FM: ICML/ICLR Evidence-Gap Audit

**Status:** High-level synthesis and gap-audit document only. No code, config, test, README, FedTrace, attack, Multi-Krum, Research Database, or experiment result file was created or modified in producing it. No experiment was run. No shell/PowerShell/Bash command was executed. This document was produced entirely by reading `research_plans/*.md`, `analysis/*.md`, `Research_Database/*`, and `src/analysis/fedtrace_detector.py` directly (not solely from prior conversation context), and by synthesizing across them.

---

## 1. The Full Research Story, Reconstructed From Source Documents

**Phase 1–3 (baseline, not re-audited here in detail):** centralized ResNet-18/CIFAR-100 (`EXP_001`), IID FedAvg (`EXP_002`–`EXP_004`), cosine scheduling (`EXP_005`).

**Phase 4 (heterogeneity characterization, `EXP_006`–`EXP_013`):** Dirichlet Non-IID partitioning introduced at `α=0.5` then `α=0.1` (`EXP_006`/`EXP_007`); a three-seed (42/123/2026) `α=0.1` FedAvg baseline (`EXP_007`–`EXP_009`) established that strong Non-IID heterogeneity degrades test Top-1 by a mean 10.26pp relative to IID, with a **2.85pp purely-honest seed-to-seed noise floor** and a client sample-count coefficient of variation up to ≈31% — the single most load-bearing quantitative fact motivating everything after it. FedProx (`μ=0.01`) was tried across the same three paired seeds (`EXP_010`–`EXP_013`) and did **not** clear its own pre-registered bar (mean Top-1 delta −0.11pp, direction disagreeing on 5/8 metrics); FedAvg was retained as the sole aggregation strategy for every subsequent phase, deliberately decoupling the optimization question from the traceability question that follows.

**Phase 5 (threat model, attack, naive baseline — `research_plans/PHASE5_Malicious_Client_Traceability_Plan.md`):** formalized the exact research question — *can malicious model updates be distinguished from honest update variability already observed under this project's own established α=0.1 noise floor* — a single-attacker, local-knowledge-only, honest-server threat model, a first attack family (additive Gaussian update noise, closed-form, norm-relative strength `ρ`), a fixed malicious client (`client_002`, chosen for the smallest cross-seed sample-count spread) and fixed onset (round 20, chosen from this project's own already-established fast/calm training-phase boundary), and a naive peer-cosine 3-sigma baseline detector, pre-registered to be expected to fail under honest Non-IID heterogeneity — not merely "detectors are imperfect" in the abstract, but for two specific, evidence-grounded reasons (non-stationary pre-attack reference windows; genuine honest peer-direction disagreement from Dirichlet skew).

**`EXP_014`/`EXP_015` (development pilots, seed 42):** `EXP_014` (Gaussian, `ρ=1.0`) showed the naive peer-cosine baseline never flags anyone (defeated by a secular decline shared by every client), while a trivial norm z-score trivially "solves" the attack (`z>9` every round) — leaving no room for a more sophisticated method to add value. `EXP_015` (norm-preserving directional, `target_cosine=0.0`, `research_plans/EXP_015_Norm_Preserving_Directional_Attack_Plan.md`) removed the magnitude shortcut almost entirely (norm z-score delayed 29 rounds) but exposed a new, not-yet-formalized signal — `self_prev_round_cosine`, a client's own round-to-round temporal self-consistency — collapsing to near-zero immediately at onset. **The explicit lesson driving FedTrace's whole design:** each attack lit up exactly one signal and defeated the other two; no single scalar generalizes.

**FedTrace v1 (`research_plans/FEDTRACE_METHOD_DESIGN_PLAN.md`):** three complementary signals (magnitude `m`, peer-direction `p`, temporal self-consistency `t`), per-signal honest-reference calibration, an unweighted `max()` combiner (chosen explicitly over a weighted sum or learned combiner to avoid attack-specific tuning), a `>3` threshold, `P=3` persistence, retroactive onset-labeling. **Stage A (`analysis/FEDTRACE_STAGE_A_RETROSPECTIVE_ANALYSIS.md`)** — pure offline re-analysis of `EXP_014`/`EXP_015`'s already-collected traces — found `v1` attributed correctly on both runs, but `z_t`'s calibration (a static, pre-attack-only per-client mean) could not track `t`'s own secular decline across training, producing **3 of 4 honest clients confirmed persistent late in `EXP_015`** (rounds 43–48) — attribution survived only because `client_002` was still confirmed first. **Decision B: methodological weakness found, resolve before Stage B.**

**FedTrace v2 (`research_plans/FEDTRACE_V2_TEMPORAL_CALIBRATION_PLAN.md`):** replaced `z_t`'s calibration with a same-round, cross-sectional, robust (median/MAD) reference (`z_t_rel`) — the same class of fix already used for `p`. **The v2 retrospective (`analysis/FEDTRACE_V2_RETROSPECTIVE_ANALYSIS.md`)** found this genuinely eliminated the 3 late false positives, but introduced a *new*, worse failure: at `n=5` clients per round, the MAD scale estimator is unstable, and on specific rounds where 4 honest clients happened to cluster tightly by chance, an ordinary honest client (`client_000`) was confirmed persistent **at round 11 — before the attack even begins** — flipping attribution to the wrong client entirely. **Decision B again: v2 is a regression on the metric that matters most (which client), despite improving the raw false-positive count.**

**FedTrace v3 (`research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md`):** kept v2's same-round median reference (which correctly cancels shared drift) but replaced its unstable *scale* estimate with each client's own **historical residual calibration** — the same longitudinal mean/stdev convention already proven stable for `z_m`, applied instead to the *residual* `d(k,r)=c(r)-t(k,r)`. **The v3 retrospective (`analysis/FEDTRACE_V3_RETROSPECTIVE_ANALYSIS.md`)** found this resolved *both* prior failure modes simultaneously on the same development data that exposed them: `client_000`'s three previously-breaking rounds dropped from `z≈3.9/12.4/4.5` to `z≈0.37/0.28/0.52`; `client_002` remained correctly, immediately detected (`z` in the 20s–30s); **zero honest persistent false positives anywhere, at any round, for any client** — the first version in the series to score perfectly on every tracked metric. Frozen, unmodified, as the method evaluated in Stage B.

**Stage B (`research_plans/FEDTRACE_STAGE_B_INDEPENDENT_VALIDATION_PLAN.md`, `analysis/EXP_016`–`EXP_019`, `analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md`):** FedTrace v3, frozen before any of these four traces existed, applied without modification to two unseen seeds (123, 2026) per attack family. **4/4 PASS**: correct attribution, onset error 0, delay 2 (the theoretical minimum under `P=3`), zero honest persistent false positives, in every run. Cross-attack signal complementarity reconfirmed on independent data: `z_m` dominant for Gaussian (100% in `EXP_016`, mixed 10/30 vs. 20/30 in `EXP_017`); `z_t_res` dominant (30/30, both seeds) for directional. **Explicitly, repeatedly self-disclosed limitation: the combiner never once outperformed whichever single component already dominated a given run — its demonstrated value is attack-agnostic coverage, not synergistic fusion.** Stage B's own §21 identified the single highest-priority remaining gap: **no external baseline had been evaluated against this data.**

**Multi-Krum external-baseline branch (`research_plans/FEDTRACE_EXTERNAL_BASELINE_SELECTION_PLAN.md`, `research_plans/MULTIKRUM_EXTERNAL_BASELINE_IMPLEMENTATION_PLAN.md`, `EXP_020`–`EXP_025`, five synthesis documents):** Krum/Multi-Krum (Blanchard et al., NeurIPS 2017) selected, with FoolsGold explicitly and correctly rejected on threat-model-mismatch grounds (its mechanism requires multiple colluding attackers; this project studies exactly one). Six runs — two clean controls, two Gaussian-attacked, two directional-attacked, both seeds each — closed the branch. **Headline, fully cross-seed-replicated findings:** clean Multi-Krum persistently, falsely rejects one honest client at both seeds (always that seed's largest partition); the Gaussian attacker is rejected 30/30 post-attack rounds with an instantaneous regime switch at both seeds; the directional attacker is rejected **0/30** post-attack rounds with **no** regime switch at either seed; the full-run rejection proxy misattributes an honest client at all four attacked runs, and for the directional family never confirms the true attacker at all; Multi-Krum substantially underperforms attacked FedAvg under the directional attack at both seeds (−8.7 to −10.1pp Top-1), driven by a confirmed, code-verified mechanism (unweighted 25% averaging over-weighting the attacker relative to FedAvg's sample-weighted ≈16–20%).

This is the complete evidentiary chain this audit evaluates.

## 2. Central Paper Thesis

**One-sentence thesis:** Under realistic strong non-IID federated heterogeneity, correctly attributing which client is malicious and when its behavior began is a fundamentally different problem from robustly aggregating updates to limit damage, and a detector that calibrates each client's behavior against its own longitudinal history — rather than only against its peers in a single round — is necessary to solve the former, because per-round, distance-based robust aggregation (exemplified by Multi-Krum) neither reliably tracks *who* is responsible nor generalizes across attack geometries.

**150-word abstract-level thesis:** Federated learning under strong client heterogeneity (Dirichlet `α=0.1`) produces honest clients whose updates already diverge substantially from one another, defeating naive peer-similarity anomaly detection and, as this work directly demonstrates, defeating Byzantine-robust aggregation's usefulness as an attribution proxy: Multi-Krum's per-round geometric exclusion rejects the single largest, most heterogeneous honest client under clean conditions, exhibits attack-geometry-dependent behavior (perfect exclusion of a magnitude-revealing attacker, complete evasion by a norm-preserving one) rather than tracking any consistent notion of misbehavior, and — because it has no memory of any client's own history — cannot distinguish a chronic outlier from a newly malicious one. FedTrace, a frozen, multi-signal detector combining a magnitude z-score with a novel historically-calibrated temporal-residual score, resolves both problems: independently validated across two structurally distinct attack families and two unseen seeds, it achieves exact onset localization and zero honest false positives, while Multi-Krum's repurposed rejection pattern cannot.

**Formal scientific question:** Given a federated round's per-client submitted model updates only (never raw data), under Dirichlet-`α=0.1` heterogeneity: (A) which client, if any, is malicious this run; (B) at which round did its behavior begin; both (C) without confusing honest heterogeneity for malicious behavior — and, separately: does a standard Byzantine-robust *aggregation* rule's per-round exclusion decision answer (A)/(B) as a byproduct, or are aggregation-robustness and forensic attribution provably, empirically distinct objectives under this exact configuration?

## 3. Contribution Audit

**Contribution 1 — FedTrace v3: a validated, multi-signal, per-client-longitudinally-calibrated attribution and onset-localization method under strong Non-IID heterogeneity.**
- *Exact claim:* a frozen detector combining a magnitude z-score and a historically-calibrated temporal-residual z-score correctly attributes the malicious client with zero-round onset error and zero honest persistent false positives across two attack families and two unseen seeds.
- *Supporting experiments:* `EXP_016`–`EXP_019` (Stage B).
- *Supporting analysis:* `analysis/EXP_016`–`EXP_019` (individual validations), `analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md`.
- *Independently validated?* Yes — formulas frozen (`research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md`) before any of the four validation traces existed; the development/validation boundary is explicit and enforced throughout.
- *Limitation:* one dataset/architecture/client-count/heterogeneity level; n=2 seeds per family; one fixed attacker identity and onset round across every run in the entire project.
- *Likely reviewer objection:* "n=2 seeds is not a statistical sample; how do we know this isn't specific to `client_002` at round 20?"
- *Evidence strength:* **STRONG**, within its declared scope.

**Contribution 2 — Concrete, non-hypothetical motivation: honest Non-IID heterogeneity alone defeats naive peer-similarity detection.**
- *Exact claim:* under this project's own already-characterized α=0.1 heterogeneity, a peer-cosine 3-sigma baseline never flags the true attacker in any of 6 runs it was evaluated on (2 development + 4 validation), because honest clients' peer-agreement already declines in a shared, attack-independent way.
- *Supporting experiments:* `EXP_009` (heterogeneity characterization), `EXP_014`–`EXP_019`.
- *Supporting analysis:* `analysis/EXP_009_Three_Seed_Baseline_Analysis.md`, `analysis/EXP_014_Gaussian_Attack_Traceability_Analysis.md`, `analysis/EXP_015_Directional_Attack_Traceability_Analysis.md`, all four Stage B documents.
- *Independently validated?* Yes, the naive-baseline failure reconfirms at all 4 unseen-seed validation runs, not only the 2 development runs it was diagnosed on.
- *Limitation:* only one naive baseline construction was tried (peer-cosine); a differently-designed naive baseline was not tested.
- *Likely reviewer objection:* "Is peer-cosine a strawman naive baseline, chosen to be easy to beat?"
- *Evidence strength:* **STRONG** for the specific baseline tested; **MODERATE** as a general claim about "naive detectors" (only one was built).

**Contribution 3 — An evidence-driven, non-curve-fitted method-development case study (v1→v2→v3).**
- *Exact claim:* each of two successive methodological flaws (v1's non-stationary calibration; v2's small-sample scale instability) was diagnosed from direct inspection of real development-run data, fixed by a targeted, re-justified (not defaulted) correction, and re-checked exhaustively before being frozen — with an explicit, enforced rule that development-run evidence can motivate a fix but can never itself validate it.
- *Supporting experiments:* `EXP_014`, `EXP_015` (used identically across all three design iterations).
- *Supporting analysis:* `research_plans/FEDTRACE_METHOD_DESIGN_PLAN.md`, `.../FEDTRACE_V2_TEMPORAL_CALIBRATION_PLAN.md`, `.../FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md`, `analysis/FEDTRACE_STAGE_A_RETROSPECTIVE_ANALYSIS.md`, `analysis/FEDTRACE_V2_RETROSPECTIVE_ANALYSIS.md`, `analysis/FEDTRACE_V3_RETROSPECTIVE_ANALYSIS.md`.
- *Independently validated?* The methodology itself (not a specific numeric result) is the contribution; its value is demonstrated by the fact that v3, not v1 or v2, is what survived Stage B.
- *Limitation:* this is a process contribution, harder to state as a single falsifiable claim than Contributions 1/2/4; reviewers unfamiliar with the development history will only see the frozen v3 formula unless the paper explicitly narrates this history (Part 9 addresses where).
- *Likely reviewer objection:* "This reads as engineering iteration, not a scientific contribution — what's the generalizable lesson?"
- *Evidence strength:* **MODERATE** as a standalone contribution; **STRONG** as supporting evidence for Contribution 1's credibility (it directly rebuts "how do we know v3 isn't itself overfit to `EXP_014`/`EXP_015`?").

**Contribution 4 — Empirical demonstration that robust aggregation and forensic attribution are distinct objectives, via a rigorous external baseline (Multi-Krum).**
- *Exact claim:* Multi-Krum's per-round geometric exclusion is attack-geometry-dependent (100% vs. 0% attacker-rejection rate depending on whether the attack is magnitude-revealing or norm-preserving, reproduced at both seeds) and, applied as a repurposed full-run attribution proxy with no onset foreknowledge, misattributes an honest client in all four attacked runs — never correctly identifying the attacker for the directional family at either seed — while FedTrace succeeds identically at all four.
- *Supporting experiments:* `EXP_020`–`EXP_025` (six runs), paired against `EXP_008`/`EXP_009`/`EXP_016`–`EXP_019`.
- *Supporting analysis:* `analysis/MULTIKRUM_IMPLEMENTATION_VERIFICATION.md`, `EXP_020`–`EXP_025` individual analyses, `MULTIKRUM_CLEAN_CONTROL_CROSS_SEED_SYNTHESIS.md`, `MULTIKRUM_GAUSSIAN_CROSS_SEED_SYNTHESIS.md`, `MULTIKRUM_DIRECTIONAL_CROSS_SEED_SYNTHESIS.md`, `ROBUST_AGGREGATION_VS_TRACEABILITY_SYNTHESIS.md`, `GAUSSIAN_VS_DIRECTIONAL_MULTIKRUM_SEED123_SYNTHESIS.md`, `MULTIKRUM_EXTERNAL_BASELINE_FINAL_SYNTHESIS.md`.
- *Independently validated?* Yes — every finding replicated at both seeds; the implementation itself was independently, code-level audited before any GPU run (`MULTIKRUM_IMPLEMENTATION_VERIFICATION.md`).
- *Limitation:* one external baseline (Multi-Krum only; coordinate-median/trimmed-mean explicitly deferred, not implemented); one `f` value (`f=1`, matching this project's own threat model, an acknowledged asymmetry Multi-Krum is given that FedTrace is not).
- *Likely reviewer objection:* "Multi-Krum was never designed to do attribution — is this a fair comparison, or a strawman?"
- *Evidence strength:* **STRONG** for the specific claim ("this repurposed proxy fails, under this configuration"), provided the paper states — as every underlying document already does — that this is not a claim Multi-Krum's native aggregation-robustness function is defective in general.

**Contribution 5 (secondary) — An honest, non-overclaimed characterization of what multi-signal combination actually buys ("coverage, not fusion").**
- *Exact claim:* across all six development+validation FedTrace runs, the `max()` combiner never once produced a detection outcome better than whichever single component already dominated that run; its demonstrated value is not knowing in advance which signal a future attack will manipulate.
- *Supporting experiments/analysis:* all of `EXP_014`–`EXP_019`'s detector-comparison tables.
- *Independently validated?* Yes, reconfirmed at all 4 validation runs, not merely asserted at development stage.
- *Limitation:* this is a negative/honesty finding, not a capability claim — it constrains, rather than strengthens, what can be claimed for the combiner.
- *Likely reviewer objection:* "If combination never helps, why not just run two detectors and report both?" — this is directly, pre-emptively answered in the existing documents (deployment-time attack-type-agnosticism), but must be stated explicitly in the paper, not left implicit.
- *Evidence strength:* **STRONG** and unusually rare to see stated this candidly — this can be framed as a strength (honesty/rigor) rather than a weakness if the paper does not try to claim more than the evidence shows.

## 4. Claim → Evidence Matrix

| Claim | Evidence required | Existing experiments | Existing result | Independent validation? | External baseline? | Missing evidence | Paper section | Main/Appendix |
|---|---|---|---|---|---|---|---|---|
| Honest Non-IID heterogeneity produces a real, quantified noise floor | Multi-seed honest baseline | `EXP_007`–`EXP_009` | 10.26pp mean IID-vs-α0.1 degradation; 2.85pp seed noise floor | Yes (3 seeds) | N/A | — | Problem Formulation | Main |
| Naive peer-cosine detector fails under this heterogeneity | Attack + naive-detector trace | `EXP_014`–`EXP_019` | 0/6 runs ever flag anyone | Yes (4/4 validation) | N/A | — | Motivation/Related Work | Main |
| FedTrace v3 correctly attributes the attacker, exact onset, zero honest FP | Frozen-method Stage B validation | `EXP_016`–`EXP_019` | 4/4 PASS, onset err 0, delay 2, 0 honest FP | Yes | N/A | Generalization beyond `client_002`/round 20 (Part 6/7) | Main Results | Main |
| Magnitude and temporal-residual signals are complementary across attack geometry | Detector-component dominance analysis | `EXP_016`–`EXP_019` | `z_m` dominant (Gaussian), `z_t_res` dominant (directional), both seeds | Yes | N/A | — | Ablation | Main |
| Max-combiner provides coverage, not fusion | Same as above, combiner vs. best-single-component | `EXP_014`–`EXP_019` | Combiner never beats best component in any of 6 runs | Yes | N/A | — | Ablation | Main |
| Clean Multi-Krum rejects one honest client persistently, no attacker present | Clean-control Multi-Krum run | `EXP_020`, `EXP_021` | 95–98% rejection rate at both seeds, largest-partition client | Yes (2 seeds) | Yes | — | Robust Agg. vs. Traceability | Main |
| Multi-Krum exclusion outcome is attack-geometry-dependent | Attacked Multi-Krum, both families, both seeds | `EXP_022`–`EXP_025` | 100% (Gaussian) vs. 0% (directional) rejection, both seeds | Yes (2×2) | Yes | Behavior at intermediate attack severities | Robust Agg. vs. Traceability | Main |
| Multi-Krum's full-run rejection proxy fails as an attribution method | Same as above, `P_krum=3` applied honestly | `EXP_022`–`EXP_025` | Misattributes at all 4; never confirms attacker in either directional run | Yes | Yes | — | Robust Agg. vs. Traceability | Main |
| Multi-Krum substantially underperforms FedAvg under directional attack | Same 2×2 factorial | `EXP_008/009/018/019/020/021/024/025` | −8.7 to −10.1pp Top-1, both seeds | Yes (2 seeds) | Yes | — | Robust Agg. vs. Traceability | Main |
| A second robust-aggregation/attribution external baseline agrees or disagrees with Multi-Krum | Coordinate-median/trimmed-mean or FoolsGold-class run | none | Not run (explicitly deferred, `FEDTRACE_EXTERNAL_BASELINE_SELECTION_PLAN.md` §4) | No | No | **A full second external baseline** | Robust Agg. vs. Traceability | Appendix (if added) |
| Findings generalize beyond `client_002`/round 20 | Different attacker identity/onset run | none | Not run | No | N/A | **Essential gap — see Part 6** | Limitations / (new) Generalization | Main claim scope note |
| Findings generalize beyond `α=0.1`/CIFAR-100/ResNet-18/5 clients | Second heterogeneity level/dataset/architecture/client-count | none | Not run | No | N/A | High-value, not essential — see Part 6/7 | Limitations | Main claim scope note |

## 5. ICML/ICLR-Level Gap Audit

| # | Dimension | Status | Why |
|---|---|---|---|
| A | Novelty | **PARTIALLY READY** | The individual formulas (z-scores, `max` combiner, persistence rule) are deliberately simple, evidence-driven engineering choices — the project's own documents explicitly disclaim algorithmic novelty at that level. Novelty rests on the whole validated framework plus the empirical robust-aggregation-vs-attribution contrast (Contribution 4), which is genuinely under-explored empirically. A paper leading with "we propose a new z-score" would be under-novel; one leading with "attribution and robust aggregation are different problems, demonstrated" is defensible. |
| B | Methodological depth | **PARTIALLY READY** | The iterative, diagnosed-not-guessed v1→v2→v3 process and the strict development/validation boundary are unusually disciplined. Depth in the sense of adaptive-adversary analysis, formal guarantees, or multi-attacker extension is absent. |
| C | Mathematical formalization | **PARTIALLY READY** | Every formula is fully, unambiguously specified (Part 8 restates them). They are stated as calibration/scoring *rules*, not as a formalized statistical model with assumptions declared as such. Achievable, scoped formalization is laid out in Part 8; it does not yet exist as written paper content. |
| D | Theoretical justification | **NOT READY** | Qualitative, evidence-consistent arguments exist (the additive-drift-cancellation reasoning in the v2/v3 plans); no formal proposition, bound, or guarantee has been written down as such. Part 8 identifies exactly what is achievable without inventing unsupported theory. |
| E | Experimental rigor | **READY** (within scope) | Frozen-before-validation discipline, paired single-variable-change designs, exhaustive (not sampled) per-round/per-client checks in every retrospective, and an explicit no-post-hoc-tuning rule enforced across every document in the series — this is a genuine strength, not merely adequate. |
| F | Dataset breadth | **NOT READY** | CIFAR-100 only. |
| G | Model breadth | **NOT READY** | ResNet-18 only. |
| H | Attack breadth | **PARTIALLY READY** | Two attack families chosen deliberately to stress opposite signal dimensions (magnitude vs. norm-preserving direction) — a real, purposeful contrast, not an arbitrary pair. No label-flipping/backdoor/sign-flip/scaling attack, no severity sweep, no adaptive adversary. |
| I | Heterogeneity breadth | **PARTIALLY READY** | Only `α=0.1` (deliberately "strong") is used for every attack/attribution/Multi-Krum result; `α=0.5` exists only in the earlier, attack-free honest baseline (`EXP_006`) and was never carried into the attack/attribution/Multi-Krum work. |
| J | Client-scale breadth | **NOT READY** | `n=5` clients, all participating every round, in every single run across the entire project. No partial-participation or larger-`n` regime tested. |
| K | Seed/statistical rigor | **PARTIALLY READY** | n=2 independent, never-tuned-against seeds per attack family is disciplined and honestly, repeatedly self-labeled as "not a statistical-significance sample" throughout the source documents — this honesty is itself a strength, but the underlying breadth is still thin by top-tier standards. |
| L | External baselines | **PARTIALLY READY** | One baseline (Multi-Krum) implemented with real rigor (6 runs, 2×2 factorial, code-audited, cross-seed replicated) — a genuine, unusually thorough execution. Only one of the ~3 originally-scoped candidate categories was pursued; coordinate-median/trimmed-mean remains documented-but-not-implemented. |
| M | Ablation completeness | **READY** | Magnitude-only/peer-only/temporal-only/all pairwise/full-combiner ablations exist for every development and validation run; the v1/v2/v3 evolution is itself an extensive additional ablation history. |
| N | Computational/reproducibility reporting | **PARTIALLY READY** | Per-experiment reproducibility (config, seed, git commit, runtime, checkpoint paths) is excellent and consistent across all 25+ experiments. Cross-experiment reproducibility bookkeeping (the master database, Part 14) is severely stale, and no third-party/external reproduction has been attempted. |
| O | Limitations/threats to validity | **READY** | Every single document in the series carries an explicit, extensive, non-perfunctory "Claims Not Supported"/Limitations section — one of the strongest aspects of the whole program, and directly reusable as paper content. |
| P | Visualization quality | **NOT READY** | Only default per-experiment matplotlib plots (validation curves, confusion matrices) exist; no publication-quality, cross-run comparison figure has been produced. Part 11 specifies exactly what is needed. |
| Q | Relation to prior work | **PARTIALLY READY** | Multi-Krum, coordinate-median/trimmed-mean, and FoolsGold are each verified against primary sources with correct citations (`FEDTRACE_EXTERNAL_BASELINE_SELECTION_PLAN.md` §5) — a real, if narrow, start. A full related-work section (broader FL anomaly-detection/provenance literature, Byzantine-robust aggregation survey) has not been assembled. |

## 6. Essential vs. Optional Experiments

**ESSENTIAL BEFORE TOP-TIER SUBMISSION (exactly one item):**

**Malicious-client-identity / attack-onset generalization check.**
- *Reviewer question answered:* "Is `client_002` special, or is round 20 special? Does FedTrace only work because it was tuned to this exact scenario?" — this is the single most obvious, hardest-to-deflect objection given that **every attack-bearing run in the entire project (`EXP_014`–`EXP_019`, `EXP_022`–`EXP_025`, ten runs total) uses the identical malicious client (`client_002`) at the identical onset round (20)**.
- *Scientific variable changed:* malicious client identity (e.g. `client_004`, the largest-partition client, already independently interesting given its role in the Multi-Krum clean-control findings) and/or attack onset round (e.g. round 30), holding attack family, seed, and everything else fixed.
- *Expected information gain:* high — directly tests whether FedTrace's calibration procedure (per-client, longitudinal) is client-identity-invariant, and whether its onset-localization mechanism generalizes to a different true onset, using zero new infrastructure.
- *Minimum number of runs:* 1–2 (e.g., one Gaussian-attack run with a different malicious client at one already-validated seed; optionally one run with a shifted onset round).
- *Scope-creep risk:* low — reuses the exact frozen attack code, frozen FedTrace v3 formulas, and existing config-generation convention; introduces no new attack family, dataset, model, or aggregation strategy.
- *Exact paper claim strengthened:* Contribution 1 (FedTrace generalizes within its threat model, not merely to one hand-picked scenario) and directly closes one of Stage B Final Synthesis's own explicitly self-disclosed limitations ("robustness to a different attacker identity... arbitrary attack onset times" — both currently untested).

**HIGH-VALUE IF BUDGET ALLOWS:**

| Candidate | Reviewer question | Variable changed | Info gain | Min. runs | Scope-creep risk | Claim strengthened |
|---|---|---|---|---|---|---|
| Second heterogeneity level (e.g. `α=0.5`, reusing `EXP_006`'s already-characterized honest baseline) | "Does FedTrace need *this much* heterogeneity to be necessary?" | `dirichlet_alpha` | Moderate–high | 2 (one attack family, both seeds, or one seed two families) | Low (infra exists) | Scope of the heterogeneity claim (Problem Formulation) |
| Second external baseline (coordinate-median/trimmed-mean, already literature-verified and scoped) | "Is Multi-Krum representative of robust aggregation generally, or an unlucky pick?" | Aggregation strategy | Moderate | 6 (mirroring the Multi-Krum matrix) | Moderate (new implementation) | Contribution 4's generality |
| Larger client count | "Does the `n=5`, small-cross-section regime drive both FedTrace's success and Multi-Krum's failure modes?" | `num_clients`/`clients_per_round` | Moderate | 2–4 | Moderate–high (partition/infra changes) | External validity of both main results |
| Second architecture (e.g. a smaller/larger CNN on the same dataset) | "Is this ResNet-18-specific?" | Model | Low–moderate | 2 | Moderate | Dataset/model breadth (G) |

**NOT WORTH RUNNING:**

- A third or later seed per already-tested configuration — two unseen seeds already independently validated every pre-registered criterion in both branches; a third would not close any identified gap (already explicitly declined in `FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md` §22).
- Any threshold/`P`/`T_ref`/`f`/`m`/attack-severity sweep with no attached claim — every document in this series explicitly, repeatedly prohibits post-hoc tuning; a sweep now would undermine, not strengthen, the credibility already built.
- A third attack family without a specific, pre-registered hypothesis about what new signal dimension it would stress.
- FoolsGold or any other multi-colluding-attacker-oriented method — already correctly, explicitly rejected as a threat-model mismatch.
- Second dataset/domain extension (e.g. a foundation-model or medical-imaging variant) — out of scope for this evidence package; would introduce new, unvalidated confounds without closing any currently-identified gap.
- Reopening the FedProx/optimization track — already a closed, separate question.

## 7. Generalization Audit

Ranked by scientific information gain (highest first), **not** all recommended:

1. **Malicious-client identity / onset round** (Part 6, essential) — highest gain, lowest cost, directly closes the most concentrated, most obvious confound in the entire evidence base (every one of 10 attack runs uses the same client at the same round).
2. **Second heterogeneity level (`α`)** — moderate-high gain; the entire project's premise is heterogeneity severity, so showing the method's necessity/success scales with it (rather than being an `α=0.1`-only artifact) is a natural, cheap-relative-to-payoff extension (an honest baseline at `α=0.5` already exists).
3. **Second external baseline (coordinate-median/trimmed-mean)** — moderate gain for Contribution 4's generality; more implementation cost than #1/#2.
4. **Client-scale (`n`)** — moderate gain, but changes partition mechanics enough to risk being a second, harder-to-isolate variable change.
5. **Second architecture** — low-moderate gain relative to cost; ResNet-18/CIFAR-100 is a standard, defensible FL benchmark configuration on its own.
6. **Second dataset** — similar to #5; lower priority than heterogeneity/client-identity axes given this project's specific thesis is about heterogeneity and attribution, not dataset generality.
7. **Attack-severity sweep** — explicitly, repeatedly rejected by every prior document in this series as scope creep without a specific claim; would also reopen the "no post-hoc tuning" discipline the whole program is built on if run after seeing results.
8. **More seeds at already-tested configurations** — lowest marginal gain; two independently-validated seeds already satisfy the pre-registered cross-seed criterion in both branches.

**No combination beyond #1 (essential) and, at most, #2–#3 (high-value) is recommended.**

## 8. Theory / Mathematical Contribution Audit

**Formalization of the existing method**, restated precisely (matching `research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md` §16 and `analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md` §3 exactly):

Let client `k`'s round-`r` submitted update be `Δ_{k,r} = w_{k,r}^{submitted} − w_r^{global}` (trainable parameters only).

- **Magnitude signal:** `m(k,r) = ‖Δ_{k,r}‖₂`.
- **Temporal self-consistency:** `t(k,r) = cos(Δ_{k,r}, Δ_{k,r−1})`.
- **Round reference:** `c(r) = median_j t(j,r)` (same-round, cross-sectional, no leave-one-out).
- **Residual:** `d(k,r) = c(r) − t(k,r)`.
- **Historical calibration:** `D_k = {d(k,r) : r ∈ [1, T_ref)}`, `T_ref = 20`; `center_k = mean(D_k)`; `scale_k = stdev(D_k)` (`n−1`).
- **One-sided temporal-residual score:** `z_t_res(k,r) = max(0, [d(k,r) − center_k] / scale_k)`.
- **Magnitude score:** `z_m(k,r)` — per-client longitudinal `|m(k,r) − mean_pre_m(k)| / std_pre_m(k)` over `r < T_ref`.
- **FedTrace score:** `FedTraceV3Score(k,r) = max(z_m(k,r), z_t_res(k,r))`.
- **Persistence/attribution/onset:** detected at `r*` iff score `> 3` for every `r ∈ [r*, r*+P−1]`, `P=3`; predicted client = earliest-confirmed (ties: higher mean score, then `client_id`); onset estimate `= r*` (not the confirmation round).

**What is genuinely achievable as light, honestly-scoped theory — without inventing unsupported claims:**

**Proposition 1 (drift cancellation, model-conditional).** Under the additive decomposition `t(k,r) ≈ trend(r) + offset(k) + noise(k,r)` (a stated *modeling assumption*, not derived from first principles about SGD), if `c(r) ≈ trend(r) + median_j offset(j)`, then `E[d(k,r)] = offset(k) − median_j offset(j)`, independent of `r` — the shared secular trend cancels by construction. This is already the qualitative argument in `research_plans/FEDTRACE_V2_TEMPORAL_CALIBRATION_PLAN.md` §4; it can be written as a one-paragraph, clearly-labeled proposition, explicitly flagged as conditional on the stated decomposition, not proven from raw training dynamics.

**Lemma 1 (minimum detection delay, exact/combinatorial).** Under the frozen persistence rule, if a client's score exceeds threshold at every round from its true onset `r_0` onward, confirmation occurs at exactly `r_0 + P − 1` and no `P`-round persistence rule can do better. This is a trivial but exact, statable fact about the decision rule — already empirically realized as delay `=2` in all four Stage B runs — and is legitimate to include as a short lemma.

**Proposition 2 (Krum's failure condition under norm-preserving attacks, derivable).** Krum-family scoring uses `‖Δ_i − Δ_j‖² = ‖Δ_i‖² + ‖Δ_j‖² − 2⟨Δ_i,Δ_j⟩`. For a norm-preserving attacker (`‖Δ_attacker‖ ≈ ‖Δ_honest_counterfactual‖` by construction), the distance to any peer is governed by the cross term (direction alignment) rather than a magnitude term. **If the attacker-induced directional distance increase is smaller than the pre-existing heterogeneity-driven distance gap between the most distant honest client and the rest, Krum's ranking is unchanged and the attacker is not excluded.** This is a clean, derivable if-then statement (standard vector-norm algebra plus the already-established empirical fact that one honest client's baseline distance already dominates the ranking under `α=0.1`) and is directly, exactly consistent with the observed 0/30 rejection rate in both directional Multi-Krum runs — genuinely achievable as a formal failure-condition proposition, not merely an empirical description.

**Observation 1 (structural distinction between exclusion and attribution, definitional but precise).** Multi-Krum's selection function at round `r` depends only on `{Δ_{k,r}}_{k=1}^n` — it is, by construction, memoryless in `k`'s own history. `z_t_res(k,r)`, by contrast, is an explicit function of `k`'s own historical residual set `D_k`. This is a structural fact about the two decision rules' domains (not requiring proof beyond restating the definitions), and it is the precise, formal version of the empirically-repeated finding that Multi-Krum's rejection pattern cannot distinguish a chronic outlier from a newly-anomalous client.

**On the five specific questions asked:**
1. **Why Gaussian noise inflates norm:** for independent zero-mean noise `n` added to `Δ_honest`, `E[‖Δ_honest+n‖²] = ‖Δ_honest‖² + E[‖n‖²]` (cross term vanishes in expectation) — a standard, one-line algebraic fact, directly consistent with the attack's own `ρ`-scaled construction. Achievable as a two-line remark, not a new result.
2. **Why norm-preserving rotation defeats magnitude detection:** by construction — `delta_attack_norm_ratio ≈ 1.0` to `~1e-10` precision is verified directly in every attacked trace this project has produced. This is close to definitional (the attack is explicitly built to be orthogonal to the `m` signal) and should be stated as such, not dressed up as a deeper result.
3. **Why temporal residual detects directional manipulation:** the attack replaces `Δ_{k,r}`'s direction with an independently-drawn rotation to `target_cosine=0` relative to the honest direction; unless that rotation happens to align with `Δ_{k,r−1}` (which it is not constructed to do), `t(k,r) = cos(Δ_{k,r},Δ_{k,r−1})` collapses toward the rotation's own low similarity to the prior honest trajectory. This is directly, repeatedly, empirically confirmed (`self_prev_round_cosine` collapsing to magnitudes `<10⁻³` immediately post-onset in every directional-attack trace inspected across this entire project) and can be stated as a mechanistic explanation grounded in the attack's own literal construction, not a new theorem.
4. **Why Euclidean-distance robust aggregation can fail under norm-preserving directional changes:** Proposition 2, above.
5. **Distinction between per-round exclusion and longitudinal attribution:** Observation 1, above.

**What is genuinely NOT achievable without inventing unsupported theory, and should not be claimed:** any formal statistical guarantee — a bound on false-positive probability under a stated noise model for `t(k,r)`, a consistency proof for `z_t_res` as `T_ref → ∞`, or a Byzantine-resilience-style bound analogous to Krum's own `n ≥ 2f+3` guarantee. No such result currently exists anywhere in this project's evidence, and manufacturing one now (e.g., assuming a specific noise distribution for `t(k,r)` purely to derive a rate) would be exactly the kind of invented theory this audit is instructed to avoid. **The paper's "Theoretical Analysis" section should be scoped to Propositions 1–2/Lemma 1/Observation 1 above plus the mechanistic remarks for items 1–3, explicitly framed as *formalization and failure-condition analysis*, not as a statistical-guarantee contribution.**

## 9. Ablation Audit

| Ablation | Recommended placement | Rationale |
|---|---|---|
| Full FedTrace v3 vs. magnitude-only vs. temporal-residual-only vs. peer-only, per Stage B run | **MAIN PAPER** | Direct evidence for the multi-signal design and the "coverage, not fusion" finding — central to the paper's argument. |
| Naive peer-cosine baseline vs. FedTrace | **MAIN PAPER** | The paper's core motivation figure/table. |
| Multi-Krum rejection proxy vs. FedTrace (Table A) and attacked Multi-Krum vs. attacked FedAvg (Table B) | **MAIN PAPER** | Central to Contribution 4; must remain two separate tables, never merged (per this project's own established discipline). |
| Pairwise combinations never informative in practice (magnitude+peer, peer+temporal) and the peer signal's own demotion rationale | **APPENDIX** (one compact summary line/table in main text: "peer excluded — 0% dominance across all 6 development+validation runs") | Legitimate supporting evidence but not central to the argument; full detail belongs in an appendix table. |
| v1→v2→v3 calibration evolution (client_000's round-9-11 false positive, the MAD-instability mechanism, the exact hand-verified spot-check tables) | **APPENDIX** ("Development History / Design Iteration") | Valuable for credibility and reproducibility, but would slow the main narrative; frame as a case study in avoiding curve-fitting, not as required reading for the main result. |
| The original, superseded 6-/8-round hand-verified spot-check tables from the early v2/v3 design documents (since fully superseded by the exhaustive Stage-A-v2/v3 retrospective checks) | **DEVELOPMENT HISTORY ONLY — not in the paper at all, not even the appendix** | Redundant once the exhaustive check exists; belongs only in `research_plans/`/`analysis/` project history. |

## 10. Multi-Krum Paper Positioning

**Avoid entirely:** "FedTrace beats Multi-Krum as a detector" — Multi-Krum was never designed to be a detector, and framing the comparison this way invites (and deserves) a strawman objection.

**Supported, reviewer-safe framing:** *robust aggregation and forensic attribution answer different questions, and this difference has real, measurable, attack-geometry-dependent consequences when one is repurposed to answer the other's question.* Precise language:

- **Known-`f` asymmetry:** Multi-Krum requires an assumed upper bound on the number of Byzantine clients (`f=1` here, matching this project's own single-attacker threat model exactly); FedTrace requires no such input. This is a real advantage given to Multi-Krum, not to FedTrace, and must be stated every time a comparison is made — not as a point in FedTrace's favor, but as a fairness caveat working *against* the paper's own thesis, making the observed Multi-Krum failures more, not less, notable.
- **Clean Non-IID rejection behavior:** with zero attacker present, Multi-Krum persistently, falsely excludes one honest client at both tested seeds — always that seed's single largest Dirichlet partition. This is reported as a property of distance-based exclusion under heterogeneity, not a claim about Multi-Krum's aggregation quality in general.
- **Gaussian success:** Multi-Krum's raw distance metric completely and immediately excludes a magnitude-revealing attacker (100% rejection, zero-lag switch, both seeds) — a genuine, verified capability, not diminished by the paper's other findings.
- **Directional failure:** the same mechanism completely fails against a norm-preserving attacker (0% rejection, no switch, both seeds) — a real, attack-geometry-specific blind spot, mechanistically explained (Proposition 2, Part 8), not merely observed.
- **Utility cost:** Multi-Krum imposes a substantial clean-performance cost (~7pp) and, under the directional attack specifically, performs *worse* than doing nothing (attacked FedAvg) by 8.7–10.1pp — driven partly by its literature-faithful unweighted-averaging convention giving the (undetected) attacker's update more relative influence than FedAvg's sample-weighting would.
- **Why the rejection proxy is not native attribution:** Multi-Krum's selection decision is a memoryless function of one round's cross-section (Observation 1, Part 8); the pre-registered `P_krum=3` full-run proxy, applied honestly (no onset foreknowledge — the only fair way to apply it), names the wrong (honest, largest-partition) client first in all four attacked runs, and for the directional family never names the true attacker at all.

**Explicitly avoid overselling:** the paper must not claim Multi-Krum's native robust-aggregation function is defective in general (it succeeds completely against the Gaussian family), and must not claim FedTrace would succeed against every conceivable attack.

## 11. Paper Figure Plan

| # | Message | X-axis | Y-axis | Curves/bars | Experiments | Placement | Caption (expected) |
|---|---|---|---|---|---|---|---|
| 1 | FedTrace correctly and promptly localizes onset for both attack families | Round | `FedTraceV3Score` | Malicious client + honest clients (2 panels: Gaussian, directional) | `EXP_016`, `EXP_018` (or all 4 as small multiples) | **Main** | "FedTraceV3Score for the malicious client and all honest clients; the malicious client crosses threshold within 2 rounds of the true onset (dashed line) in both attack families, while honest clients remain below it throughout." |
| 2 | Magnitude and temporal-residual signals are complementary, not redundant | Round | z-score | `z_m`, `z_t_res` (2 panels: Gaussian, directional) | Same as Fig. 1 | **Main** | "Component z-scores for the malicious client: magnitude dominates under the Gaussian attack, temporal residual dominates under the norm-preserving directional attack — the same fixed detector relies on whichever signal is informative." |
| 3 | Multi-Krum's exclusion outcome is attack-geometry-dependent, not attacker-behavior-dependent | Round | Client | Selected/rejected heatmap, 5 clients × 50 rounds (2 panels, same seed: Gaussian vs. directional) | `EXP_022`, `EXP_024` | **Main** | "Per-round Multi-Krum selection status at seed 123: the same attacker, at the same onset, is rejected in 100% of post-attack rounds under a Gaussian attack (left) and 0% under a norm-preserving directional attack (right)." |
| 4 | Robust aggregation's rejection proxy and forensic attribution disagree on which client — and when | Round | Annotated timeline | True onset, FedTrace confirmation, Multi-Krum proxy's (wrong) confirmation | `EXP_018`/`EXP_024` (or the seed-2026 pair) | **Main** | "Timeline comparison: FedTrace correctly confirms the true attacker two rounds after true onset; the Multi-Krum rejection proxy confirms an honest client sixteen rounds before the attack begins, and never confirms the true attacker at all." |
| 5 | Utility cost/benefit differs sharply by attack family and is never favorable to Multi-Krum relative to FedAvg | Condition (clean/attacked) × aggregation | Top-1 / Top-5 (%) | Grouped bars, 4 conditions × 2 aggregations, 2 attack families | `EXP_008/009/016-025` | **Main** | "Test accuracy across the full 2×2 factorial (clean/attacked × FedAvg/Multi-Krum), both attack families; Multi-Krum never exceeds attacked FedAvg." |
| 6 | Clean Multi-Krum's rejection targets the largest-partition honest client, not a random one | Client (sorted by sample count) | Rejection rate / count | Bars, annotated with sample count, both seeds | `EXP_020`, `EXP_021` | **Main** | "Clean-run rejection rate by client, both seeds; the most-rejected client is, in both cases, the client holding the largest Dirichlet partition." |
| 7 | Development history: each design iteration's diagnosed flaw and fix | Version (v1→v2→v3) | Honest false-positive count / attribution correctness | Bars or table-as-figure | `EXP_014`, `EXP_015` (development) | **Appendix** | "Honest persistent false positives and attribution correctness across three successive FedTrace temporal-calibration designs on the same development data." |

## 12. Paper Table Plan

- **Table 1 — Experimental setup:** columns `Component, Value` — Dataset, Model, Clients, Partition/`α`, Aggregation strategies compared, Attack families, Malicious client, Attack onset, Seeds, Communication rounds.
- **Table 2 — FedTrace independent validation:** columns `Run, Attack family, Seed, Predicted client, Onset error, Delay, Honest FP, Verdict` — directly `analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md` §5.
- **Table 3 — Ablation/detector comparison:** columns `Detector, Components, [per-run: Attribution / Onset error / Delay / Honest FP]` for `EXP_016`–`EXP_019`.
- **Table 4 — External baseline (Multi-Krum):** columns `Condition (Clean/Gaussian/Directional) × Seed, FedAvg metric, Multi-Krum metric, Δ, Attacker rejection rate, Regime switch (Y/N), Proxy attribution (correct/wrong)`.
- **Table 5 — Generalization (future, pending Part 6's essential experiment):** columns `Attacker identity, Onset round, [Attribution / Onset error / Delay / Honest FP]` — placeholder structure only, not populated by this audit.

## 13. Overleaf/LaTeX Structure

| Section | Fed primarily by |
|---|---|
| 1. Abstract | Part 2 of this audit; `FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md` §19; `MULTIKRUM_EXTERNAL_BASELINE_FINAL_SYNTHESIS.md` |
| 2. Introduction | `PHASE5_Malicious_Client_Traceability_Plan.md` §1–2; `FEDTRACE_METHOD_DESIGN_PLAN.md` §1–2; `EXP_009`/`EXP_013` motivation |
| 3. Related Work | `FEDTRACE_EXTERNAL_BASELINE_SELECTION_PLAN.md` §5 (verified citations); **not yet written as a full literature review — a gap** |
| 4. Problem Formulation | `PHASE5_Malicious_Client_Traceability_Plan.md` §2–3; `FEDTRACE_METHOD_DESIGN_PLAN.md` §1/§3 |
| 5. FedTrace Method | `FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md` §16; `FEDTRACE_METHOD_DESIGN_PLAN.md` §4–9 |
| 6. Theoretical Analysis | **New — Part 8 of this audit; does not yet exist as written content anywhere** |
| 7. Experimental Setup | `EXP_009` baseline docs; `PHASE5...Plan.md` attack config; `FEDTRACE_STAGE_B_INDEPENDENT_VALIDATION_PLAN.md` |
| 8. Main Results | `FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md` §5–13; `MULTIKRUM_EXTERNAL_BASELINE_FINAL_SYNTHESIS.md` |
| 9. Ablation | `FEDTRACE_STAGE_A/V2/V3_RETROSPECTIVE_ANALYSIS.md`; Stage B per-run detector tables |
| 10. Robust Aggregation vs. Traceability | Entire Multi-Krum branch (`EXP_020`–`EXP_025` + five synthesis documents) |
| 11. Limitations | `FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md` §17–18; `MULTIKRUM_EXTERNAL_BASELINE_FINAL_SYNTHESIS.md`'s UNSUPPORTED section, consolidated |
| 12. Conclusion | Synthesis of all of the above |
| Appendices | v1/v2/v3 development-history documents; full per-run detector tables; `EXP_014`/`EXP_015` development analyses; Research_Database provenance references |

**No full paper text is written in this task.**

## 14. Master Documentation Audit

- **`Research_Database/FedTrace_Master_Results.xlsx`**: **exists**, at `Research_Database/FedTrace_Master_Results.xlsx`. Per its own specification document (`Research_Database/FedTrace_Master_Results.md`), it was added directly (not generated by any script) and mirrors eight CSV source tables under `Research_Database/FedTrace_Master_Results/`.
- **`sync_master_results.py`**: **does not exist anywhere in this repository** — confirmed by an exhaustive filename search; `Research_Database/README.md` itself states explicitly: *"No automation yet... there is no script that auto-populates the CSV source tables from experiment outputs, and no script that regenerates/synchronizes `FedTrace_Master_Results.xlsx`."*
- **What result artifacts a future sync script would need to read:** each experiment's `summary.json`, `round_history.csv`, `final_test_metrics.json`, `client_round_trace.csv` (attack runs), `multikrum_diagnostics.csv` (Multi-Krum runs), and `run.log` — the same artifacts every analysis document in this project already reads directly.
- **Which sheets it would update:** `01_Experiment_Summary`, `02_Hyperparameters`, `03_Hardware` (every experiment); `04_Federated_Results` (federated only); `05_Attack_Results` (attack runs); `06_Attribution_Results` (attribution/FedTrace runs); `07_Paper_Tables`/`08_Notes` (manual/as-needed).
- **Whether it supports `EXP_014`–`EXP_025` fields:** **no** — `Experiment_Index.csv` currently contains exactly four rows (`EXP_001`–`EXP_004`); the workbook specification's own "Current Row Counts" table reports only two populated experiment rows (`EXP_001`, `EXP_002`) in its CSV source tables as of its last update. **Every one of the 21 experiments from `EXP_005` through `EXP_025` — the entire heterogeneity-characterization, FedProx, attack, FedTrace-validation, and Multi-Krum-external-baseline bodies of work — is completely unregistered in the master database.**
- **Whether Attack Results / Attribution Results / Paper Tables are supported:** the *schema* for `05_Attack_Results.csv` and `06_Attribution_Results.csv` already exists and is well-designed (columns anticipate exactly this kind of data), but both are **empty, header-only**, as documented in the spec's own "Current Row Counts" table — not one attack or attribution experiment has ever been entered. `07_Paper_Tables`/`Table_Index.csv` are likewise empty.
- **Whether Multi-Krum diagnostics are supported:** **no** — no column anywhere in the current eight-sheet schema anticipates Multi-Krum-specific fields (`krum_score`, `krum_rank`, `multikrum_selected`/`rejected`, `f`/`m`/`neighbors_used`); the schema predates the Multi-Krum branch entirely and would need extension, not merely population, to represent it.
- **Whether the current workbook is stale:** **severely** — it reflects the project's state as of roughly `EXP_002`, i.e., before Non-IID partitioning, FedProx, any attack, FedTrace itself, and the entire Multi-Krum branch existed.

**Recommendation (not performed in this task): a separate, dedicated follow-up synchronization task should (a) extend the CSV schema to cover attack/attribution/Multi-Krum-specific fields not currently represented, and (b) backfill all 21 missing experiment rows from their already-existing, already-verified `summary.json`/analysis-document sources — a bookkeeping task, not a research task, and one this audit does not attempt.**

## 15. Research Continuity Package

**Could a new chat reconstruct this project from existing files alone?** Yes, but only by reading a large, currently-unindexed set of `research_plans/*.md` and `analysis/*.md` files in the correct dependency order — there is no single document that lists them in narrative sequence. `Research_Database/`, which should serve exactly this "what exists and in what order" function, cannot currently serve it (Part 14): it is stale by roughly 21 experiments and carries no attack/attribution/Multi-Krum content at all.

**Minimum document set a new session would need, in reading order:**
1. `research_plans/PHASE5_Malicious_Client_Traceability_Plan.md` (problem formulation, threat model)
2. `research_plans/FEDTRACE_METHOD_DESIGN_PLAN.md` → `FEDTRACE_V2_TEMPORAL_CALIBRATION_PLAN.md` → `FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md` (method evolution)
3. `analysis/FEDTRACE_STAGE_A_RETROSPECTIVE_ANALYSIS.md` → `FEDTRACE_V2_RETROSPECTIVE_ANALYSIS.md` → `FEDTRACE_V3_RETROSPECTIVE_ANALYSIS.md` (why each version failed/succeeded)
4. `research_plans/FEDTRACE_STAGE_B_INDEPENDENT_VALIDATION_PLAN.md` + `analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md` (frozen-method validation)
5. `research_plans/FEDTRACE_EXTERNAL_BASELINE_SELECTION_PLAN.md` → `MULTIKRUM_EXTERNAL_BASELINE_IMPLEMENTATION_PLAN.md` → `analysis/MULTIKRUM_IMPLEMENTATION_VERIFICATION.md` → the six `EXP_020`–`EXP_025` analyses and five cross-seed/final syntheses (Multi-Krum branch)
6. This document (`analysis/ICML_ICLR_EVIDENCE_GAP_AUDIT.md`) as the current evidence-state snapshot.

**Proposed (not created in this task):** `research_plans/FEDTRACE_MASTER_RESEARCH_STATUS.md`, exactly as this task's own template suggests — a single, continuously-updated index containing the research objective, the frozen method (v3 formula), a chronological experiment registry (superseding the stale `Research_Database` for narrative purposes, without replacing its tabular role), current evidence status, external-baseline status, paper claims, known limitations, and the next gate. **This document is deliberately not created here** — the task explicitly permits proposing it without requiring its creation, and creating it now would itself be a scope decision better made once Part 17's gate decision (and any resulting Part 6 experiment) is acted on, so it can be written once, correctly, rather than immediately superseded.

## 16. Current Completion Estimate

*(Project-state estimates, not acceptance probabilities.)*

| Dimension | Estimate |
|---|---|
| Method development | **~95%** — v1→v2→v3 fully iterated, each failure diagnosed and fixed from real data, frozen; only Part 8's light formalization remains unwritten. |
| Experimental core | **~85%** — FedTrace Stage A/B and the full 6-run Multi-Krum matrix are complete and rigorously analyzed; the one essential generalization check (Part 6) is the main gap. |
| External baseline | **~90%** for the one baseline pursued (Multi-Krum, unusually thorough); **~30–35%** if judged against the originally-scoped plural "external baselines" (coordinate-median/trimmed-mean remains undone). |
| Reproducibility | **~60%** — excellent per-experiment reproducibility (config/seed/git-commit/summary.json for every run); dragged down by a severely stale cross-experiment master database (Part 14) and zero third-party reproduction. |
| Top-tier evidence package | **~65%** — strong core empirical evidence; missing written formal theory, publication figures, a related-work section, and the Part 6 essential experiment. |
| Paper-writing readiness | **~40%** — abundant, well-organized raw material (every claim already has a source document with exact numbers) but zero LaTeX, figures, or tables yet exist; a dedicated writing phase is still required. |

## 17. Final Decision

**B — the core contribution is strong, but a small number of targeted experiments are necessary before paper construction should be considered complete.**

Not A: the single most concentrated, most obvious external-validity gap in the entire evidence base — every attack-bearing run in the project (ten of them) uses the identical malicious client at the identical onset round — has not been tested at all, and is cheap and fast to close.

Not C: no major scientific gap exists. The core mechanism (FedTrace v3's design and its cross-seed validation; Multi-Krum's attack-geometry-dependent exclusion and attribution failure) is independently, rigorously, cross-seed-established; what remains is scoped, small, and specific, not a broad experimental expansion.

**Minimum recommended experiment package, ordered by information gain:**

1. **(Essential)** Malicious-client-identity and/or attack-onset generalization check — 1–2 runs, reusing 100% existing infrastructure, closing the single largest external-validity gap (Part 6).
2. **(High-value, optional)** A second Dirichlet `α` level, reusing the already-characterized `EXP_006` honest baseline — directly strengthens the heterogeneity-severity framing central to the paper's whole premise.
3. **(High-value, optional)** A second external baseline (coordinate-median/trimmed-mean), already literature-verified and scoped in `FEDTRACE_EXTERNAL_BASELINE_SELECTION_PLAN.md` — strengthens Contribution 4's generality beyond one specific robust-aggregation method.

Items 2–3 are explicitly **not** required before beginning paper construction; item 1 is recommended before the paper's generalization claims are finalized, though drafting (Abstract through Related Work, Problem Formulation, Method, and the already-complete Main Results/Robust-Aggregation sections) can proceed in parallel.

---

## Final Report

1. **File created:** `analysis/ICML_ICLR_EVIDENCE_GAP_AUDIT.md`.
2. **Central paper thesis:** attribution/onset localization under strong Non-IID heterogeneity is a distinct problem from robust aggregation, requiring per-client longitudinal calibration; demonstrated via FedTrace v3's cross-seed-validated success and Multi-Krum's cross-seed-validated, attack-geometry-dependent failure as an attribution proxy (Part 2).
3. **Strongest contributions:** (1) FedTrace v3 itself — STRONG; (2) naive-baseline heterogeneity motivation — STRONG/MODERATE; (3) the v1→v2→v3 evidence-driven design process — MODERATE/STRONG (supporting); (4) the Multi-Krum robust-aggregation-vs-attribution demonstration — STRONG; (5) the "coverage, not fusion" honest characterization — STRONG (Part 3).
4. **Current strongest empirical result:** the fully-crossed 2-seed × 2-attack-family Multi-Krum dichotomy (100% vs. 0% attacker exclusion, both seeds, no exception) paired with FedTrace's unaffected, identical success across the same four conditions (Part 1, Part 3-Contribution 4).
5. **Current strongest methodological result:** the strict, enforced development-vs-validation boundary across three successive FedTrace design iterations, each failure diagnosed from real data before the next fix was attempted (Part 1, Part 3-Contribution 3).
6. **Main novelty assessment:** PARTIALLY READY — individual formulas are deliberately simple by design; novelty rests on the validated framework and the empirical robust-aggregation-vs-attribution contrast, not on any single new algorithm (Part 5-A).
7. **Top-tier readiness assessment:** strong on experimental rigor, ablation completeness, and limitations-reporting; not ready on dataset/model/client-scale breadth, theory, visualization, and cross-experiment reproducibility bookkeeping (Part 5).
8. **Essential missing evidence:** a malicious-client-identity/onset-round generalization check — the single most concentrated confound in the whole evidence base (Part 6).
9. **Optional evidence:** second heterogeneity level, second external baseline, larger client count, second architecture — none blocking (Part 6, Part 7).
10. **Experiments explicitly not recommended:** additional seeds at already-tested configurations, any parameter sweep, a third attack family without a specific hypothesis, FoolsGold, new dataset/domain extensions, reopening FedProx (Part 6).
11. **Theory/mathematical opportunities:** two propositions, one lemma, one structural observation, and three mechanistic remarks are genuinely achievable without inventing unsupported theory; no formal statistical guarantee currently exists or should be claimed (Part 8).
12. **Main-paper figure plan summary:** seven figures specified — onset-score trajectory, magnitude/temporal complementarity, Gaussian-vs-directional Multi-Krum rejection heatmap, robust-aggregation-vs-traceability timeline, 2×2 utility comparison, and clean-rejection/partition-size association in the main paper; development-history evolution in the appendix (Part 11).
13. **Main-paper table plan summary:** five tables — setup, Stage B validation, ablation/detector comparison, external baseline, and a placeholder generalization table pending the essential experiment (Part 12).
14. **Multi-Krum positioning:** "robust aggregation and forensic attribution answer different questions," never "FedTrace beats Multi-Krum as a detector"; six precise, reviewer-safe supporting points specified (known-`f` asymmetry, clean rejection behavior, Gaussian success, directional failure, utility cost, structural exclusion-vs-attribution distinction) (Part 10).
15. **Overleaf structure verdict:** a 12-section-plus-appendices outline is fully mapped to existing source documents; only the Theoretical Analysis section and the Related Work literature review have no existing written content to draw from (Part 13).
16. **Master workbook/sync-script status:** `FedTrace_Master_Results.xlsx` exists; `sync_master_results.py` does not exist anywhere in the repository; the database is stale by 21 of 25 experiments (`EXP_005`–`EXP_025` entirely unregistered), and no schema exists yet for attack/attribution/Multi-Krum-specific fields (Part 14).
17. **Research continuity status:** reconstructable from existing files, but only via a manually-ordered reading list (provided in Part 15); a master status document is proposed but not created.
18. **Completion percentages:** method development ~95%; experimental core ~85%; external baseline ~90% (one baseline)/~30–35% (plural); reproducibility ~60%; top-tier evidence package ~65%; paper-writing readiness ~40% (Part 16).
19. **Final decision:** **B** — strong core contribution; one essential, small, targeted experiment recommended before finalizing generalization claims (Part 17).
20. **Exact minimum next experiment package:** (1, essential) malicious-client-identity/onset generalization, 1–2 runs, existing infrastructure only; (2–3, optional/high-value) second `α` level, second external baseline (Part 17).
21. **Confirmation no implementation/config/result changed:** confirmed — every file read in producing this audit (`research_plans/*.md`, `analysis/*.md`, `Research_Database/*`, `src/analysis/fedtrace_detector.py`) was read-only; nothing in `src/`, `configs/`, `tests/`, `README.md`, or `Research_Database/` was created, edited, or deleted.
22. **Confirmation no experiment was run:** confirmed — no training, attack, or aggregation code was executed.
23. **Confirmation whether commands were executed:** none — this task used only `Read` and `Glob` tool calls; no shell, PowerShell, Bash, or Python command was executed at any point.
