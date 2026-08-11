# Robust Aggregation vs. Traceability: Synthesis

**Status:** Analysis/synthesis document only. No code, test, config, README, FedTrace, attack, or Research Database file was created or modified in producing it. No experiment was run. FedTrace v3 and Multi-Krum's implementation (`f=1`, `m=n-f`, scoring, tie-breaking, weighting, buffer handling, the pre-registered `P_krum=3` rejection proxy) remain exactly as frozen. This document elevates a specific finding from `analysis/EXP_022_MULTIKRUM_GAUSSIAN_SEED123_ANALYSIS.md` §G into a standalone conceptual synthesis, because it bears directly on this project's central methodological argument and is referenced independently from the run-specific analysis.

---

## 1. The Conceptual Distinction This Project Is Built Around

Two structurally different questions can be asked of a federated round's five submitted client updates:

- **Robust aggregation asks:** *"Which updates should influence the global model this round?"* — a per-round, relative, cross-sectional decision. Krum/Multi-Krum (Blanchard et al., NeurIPS 2017) answers exactly this question, and only this question: given `n` updates and an assumed Byzantine bound `f`, which `m=n-f` of them are closest to the geometric consensus this round?
- **Traceability asks:** *"Which client exhibits persistent, newly-emerging malicious behavior, when did it begin, and what evidence supports that attribution?"* — a longitudinal, client-specific, evidentiary decision. FedTrace v3 (`src/analysis/fedtrace_detector.py`, frozen per `research_plans/FEDTRACE_V3_TEMPORAL_RESIDUAL_CALIBRATION_PLAN.md`) answers exactly this question: each client is judged against its *own* historical calibration window (`T_ref=20`), not against a single round's relative ranking, and a client is only implicated after `P=3` consecutive rounds of confirmed deviation from *its own* established baseline.

`research_plans/FEDTRACE_EXTERNAL_BASELINE_SELECTION_PLAN.md` §1 and `research_plans/MULTIKRUM_EXTERNAL_BASELINE_IMPLEMENTATION_PLAN.md` §1 pre-registered this distinction *before* any Multi-Krum run existed, and pre-registered that Multi-Krum's rejection pattern would be evaluated only as an explicitly-labeled, secondary "rejection proxy" — never presented as a native or equivalent attribution capability. This document reports the first empirical test of that distinction against a genuinely attacked run.

## 2. The Empirical Test: EXP_022

`EXP_022` (Gaussian update-noise attack, `client_002`, onset round 20, `rho=1.0`, Dirichlet `α=0.1`, seed 123, `aggregation: multikrum`) is directly comparable to `EXP_016` (the same attack under FedAvg, already independently validated by FedTrace v3 — `analysis/EXP_016_STAGE_B_GAUSSIAN_SEED123_VALIDATION.md`) and to `EXP_020` (the clean Multi-Krum control at the same seed/partition — `analysis/EXP_020_CLEAN_MULTIKRUM_SEED123_ANALYSIS.md`). Full derivation of every number below is in `analysis/EXP_022_MULTIKRUM_GAUSSIAN_SEED123_ANALYSIS.md` §B–§H; this document restates only what is needed for the conceptual argument.

**What happens inside EXP_022, in one run:**

| Phase | Persistently rejected client (`P_krum=3`) | Confirmation round | Is there an attacker active? |
|---|---|---|---|
| Rounds 0–19 (clean) | `client_004` (the largest partition, 11,785/45,000 samples) | **Round 4** | No |
| Rounds 20–49 (attacked) | `client_002` (the true attacker) | **Round 22** | Yes, from round 20 |

**Applying the Multi-Krum rejection proxy's own pre-registered rule — "predicted suspicious client = the earliest persistently rejected client" — to the full run, honestly, with no foreknowledge of round 20, produces `client_004`.** `client_004`'s confirmation (round 4) precedes `client_002`'s (round 22) by 18 rounds. This is not a marginal or ambiguous tie: under the proxy's own tie-break rule (earliest confirmation wins), `client_004` is the unambiguous answer, and it is wrong.

**FedTrace v3, applied to the paired FedAvg run under the identical attack, seed, and partition (`EXP_016`), produces `client_002`** — onset error 0 (estimated onset round 20, true onset round 20), confirmation round 22, detection delay 2 rounds, and **zero honest clients confirmed at any point in the entire 50-round run**, including `client_004`.

## 3. Why the Two Methods Disagree

The disagreement is not incidental — it follows directly from each method's design, and this run makes the mechanism visible rather than theoretical:

- **Multi-Krum's per-round score has no concept of "normal for this client."** `MultiKrum.aggregate()` (`src/federated/aggregation/multikrum.py`) is stateless across rounds: every call scores only the five updates submitted *that round*, relative to each other, with no memory of any client's own history. `client_004`'s update is, every clean round, further (in raw squared Euclidean distance) from the other four clients' updates than any other honest client's — not because anything is wrong with it, but because (per the partition-size association already established in `analysis/EXP_020_CLEAN_MULTIKRUM_SEED123_ANALYSIS.md` §4 and reconfirmed in `EXP_022`'s own clean phase) it is the largest partition, and under this project's fixed `local_epochs=1` training schedule, a larger partition performs more local SGD steps per round than a smaller one — a structural, partition-size artifact, not a behavioral one. Multi-Krum's raw score cannot distinguish "this client is always somewhat far from the pack, and that is its normal, honest behavior" from "this client just became far from the pack for the first time."
- **FedTrace v3's score is explicitly relative to each client's own calibration window.** `z_m` and `z_t_res` are computed per-client, centered and scaled against that same client's own pre-`T_ref` history. `client_004`'s magnitude and temporal-residual signals never cross threshold in `EXP_016` (`analysis/EXP_016_STAGE_B_GAUSSIAN_SEED123_VALIDATION.md` §5: max honest `z_m` observed anywhere is 1.85) precisely because being "the largest, most distinct client" *is* `client_004`'s own established baseline — FedTrace measures deviation *from that baseline*, not distance from the other four clients this round.
- **The `P_krum=3` persistence rule does not fix this**, because it is applied to a binary, memoryless per-round outcome (rejected/not-rejected) rather than to a continuously-calibrated deviation score. Three consecutive rounds of "furthest from the pack, as usual" look identical, under this rule, to three consecutive rounds of "furthest from the pack, which is new."

## 4. What This Does and Does Not Establish

**Established by this evidence:**
- In this exact experimental configuration (5 clients, strong Non-IID Dirichlet `α=0.1`, fixed `local_epochs=1`, one non-colluding Gaussian attacker), the Multi-Krum rejection proxy — applied the only way a real, non-oracle deployment could apply it, without foreknowledge of the true attack onset — misattributes the honest largest-partition client instead of the true attacker.
- FedTrace v3, applied to the paired FedAvg run under identical attack/seed/partition conditions, correctly attributes the true attacker with zero honest false positives.
- The mechanism for this disagreement (stateless, relative-this-round scoring vs. per-client historical calibration) is directly traceable to each method's design, not an unexplained coincidence.
- Once restricted to the post-onset window only — i.e., once the answer is already known — Multi-Krum's raw geometric separation of the true attacker is excellent and matches FedTrace's own timing exactly (confirmation round 22, onset error 0, in both). Multi-Krum's per-round distance metric is not incapable of separating this attacker; the failure is specifically in using its *unweighted, memoryless, whole-run* rejection pattern as a standalone attribution rule.

**Not established, and not claimed, by this evidence:**
- That Multi-Krum's robust-aggregation function (as opposed to its repurposed rejection pattern) is defective — that is evaluated separately, on its own metrics, in `analysis/EXP_022_MULTIKRUM_GAUSSIAN_SEED123_ANALYSIS.md` §E/§F, and the two questions must not be merged (`research_plans/FEDTRACE_EXTERNAL_BASELINE_SELECTION_PLAN.md` §8's own invalid-comparison warning).
- That this disagreement generalizes beyond this configuration — one seed (123), one attack family (Gaussian), one attack strength (`rho=1.0`), one partition draw. The seed-2026 clean controls (`analysis/EXP_021_CLEAN_MULTIKRUM_SEED2026_ANALYSIS.md`) already show the *clean-phase* false-positive behavior replicates at a second seed (a different largest-partition client, `client_001`, confirmed at an identical round-4 timing); whether the *attacked-run misattribution* demonstrated here also replicates at seed 2026 is the next test recommended in `analysis/EXP_022_MULTIKRUM_GAUSSIAN_SEED123_ANALYSIS.md` §K, not yet performed.
- That FedTrace v3 would succeed under every possible attack, partition, or heterogeneity configuration — only that it succeeded, correctly and with zero false positives, on the exact conditions tested here and in the already-completed Stage B matrix (`EXP_016`–`EXP_019`).
- That Multi-Krum is unfit for its native purpose. A method built to answer "which updates should count this round" is not obligated to also answer "which client has been behaving abnormally since when" — the finding here is that it does not answer that second question well when repurposed to do so in this setting, which is exactly what `research_plans/MULTIKRUM_EXTERNAL_BASELINE_IMPLEMENTATION_PLAN.md` §11 pre-registered as a live possibility before this run existed, not a post-hoc reframing of an unexpected result.

## 5. Contribution to the Project's Central Argument

The project's evidence chain, updated with this finding:

1. FedTrace v3 was independently validated (Stage B, `EXP_016`–`EXP_019`) across two attack families and two seeds, with zero honest false positives in every run.
2. Multi-Krum's clean-control behavior (`EXP_020`, `EXP_021`) showed that a standard, literature-faithful robust-aggregation baseline, repurposed as an attribution proxy, produces a persistent false positive under this project's strong Non-IID heterogeneity — even with **no attacker present at all** — replicated at both tested seeds.
3. **This document's evidence (`EXP_022`) shows that false positive does not merely coexist with, but actively outraces, a genuine attacker's own signal** — inside the same run, the honest client's misattribution is confirmed 18 rounds before the true attacker's, meaning a real, non-oracle user applying the rejection proxy as an attribution tool would report the wrong client first, not merely alongside the right one.

This is the concrete, quantified answer to the reviewer question `research_plans/FEDTRACE_EXTERNAL_BASELINE_SELECTION_PLAN.md` §12 anticipated ("why not just use Krum's rejection pattern instead of a purpose-built detector?"): because, in this experimental setting, doing so does not merely produce a *less precise* answer than FedTrace — it produces a **wrong** answer, delivered with more apparent confidence (earlier confirmation) than the correct one.

## 6. Limitations and Next Step

- **Single seed, single attack family, single attack strength.** All findings here are reported as single-run observations pending replication.
- **FedTrace v3 was not re-run on `EXP_022`'s own (Multi-Krum) trace** — the FedTrace evidence used is `EXP_016`'s (FedAvg) already-frozen result, on the grounds that it shares the identical attack/seed/partition and was the run this task explicitly designated as the "attack-matched aggregation comparison." A fresh FedTrace v3 application to a Multi-Krum-aggregated run's own trace was not pre-registered in Stage B's validation matrix and is not attempted here.
- **The mechanism connecting partition size to clean-phase Krum rank** (unequal local SGD steps under fixed `local_epochs=1`) remains a hypothesis, not directly measured.
- **Recommended next step:** the seed-2026 Gaussian Multi-Krum replication (`analysis/EXP_022_MULTIKRUM_GAUSSIAN_SEED123_ANALYSIS.md` §K), which would directly test whether this section's central finding — the rejection proxy's clean-phase false positive outracing the true attacker's own confirmation — replicates at a second seed, before any claim of generality is made. **Its config is not created by this document.**

---

## Final Report Addendum (This Document)

- **File created:** `analysis/ROBUST_AGGREGATION_VS_TRACEABILITY_SYNTHESIS.md` (companion to `analysis/EXP_022_MULTIKRUM_GAUSSIAN_SEED123_ANALYSIS.md`, whose Final Report already covers both files jointly).
- **No code, config, test, README, FedTrace, attack, or Research Database file was changed; no experiment was run; no command was executed.**
