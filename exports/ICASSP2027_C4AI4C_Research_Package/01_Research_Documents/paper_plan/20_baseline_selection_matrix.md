# 20 — Baseline Selection Matrix

> **Review note — 2026-10-26 (Prompt 33, P-49).** FLDetector and FLTracer rows aligned with P-54 and P-47.

Prepared 2026-10-26 (Prompt 32). Labels: **VERIFIED-SOURCE** (read from the primary paper / page, checklist
`10_prior_work_checklist.md` §A0–§A5), **CLAUDE-REPORTED** (repository code and our runs), **UNVERIFIED**,
**PROPOSED**. No third-party code has been run.

Classes: **DIRECT** (same task and inputs: online, per-client, monitor-only on our traces/logs, K = 5),
**PARTIAL** (subset of the task, or needs adaptation that changes the method), **RELATED ONLY**, **INCOMPATIBLE**.

## 1. Statistical baselines implemented in this repository (CLAUDE-REPORTED)

| Method | Code | Inputs | Attribution | Onset | Benign drift evaluated | K = 5 / CIFAR-100 | Cost | Class |
|---|---|---|---|---|---|---|---|---|
| FedTrace v3 (frozen) | `src/analysis/fedtrace_detector.py`; online `src/rev/online.py` | trace (norm, self-cosine) | yes (first confirmation) | yes (streak start) | yes (X4) | yes / yes | negligible (CPU, seconds) | method under study |
| v3 ablations: magnitude-only, temporal-residual-only, peer-only (v1) | same module (`components`), `evaluate_detector` | trace | yes | yes | archived: no; rev: computable | yes | negligible | DIRECT (ablation) |
| Two-sided norm z-score + persistence (ZP-m, h 1.75) | `src/rev/baselines.py::zscore_persistence`, `src/rev/phase_s.py` | trace | yes | yes | yes (falsified) | yes | negligible | DIRECT |
| Norm CUSUM (k 0.5, h 11.75) | `baselines.py::cusum` | trace | yes | yes (change time) | yes (falsified) | yes | negligible | DIRECT |
| Page–Hinkley (δ 0.5) on norm / self-cosine / residuals | `baselines.py::page_hinkley` | trace | yes | yes | not on drift | yes | negligible | DIRECT (weak: misses or late, L-56) |
| Residual references: inclusive median (= v3), leave-one-out median, trimmed mean, client-historical | `src/rev/reference_audit.py`, `phase_s.v_alarms` | trace | yes | yes | drift: inclusive median only | yes | negligible | DIRECT (reference ablation) |
| Multi-Krum (f = 1) + rejection-persistence proxy | `src/federated/aggregation/multikrum.py`; archived EXP_020–025 | per-round updates; changes aggregation | proxy only | no | no | yes | one GPU run per condition | PARTIAL (aggregation defence, not forensic) |
| Naive peer-agreement detector | `apply_naive_detector` (NDSS) | trace peer field | yes | first flag | no | yes | negligible | PARTIAL (order-dependent field, L-16) |
| D1 leave-one-source-out, label-shift residual | `src/rev/source_attribution.py`, `src/rev/label_shift.py` | trace / probes | localisation only | – | yes | yes | CPU | **not baselines** — falsified exploratory analyses (doc 19 §6) |

## 2. Published FL methods

| Method (venue) | Code | Threat model | Inputs | Attacks | Client attribution | Onset | Benign drift | K = 5 | CIFAR-100 | Effort | Cost | Fair comparison? | Class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **FLTracer** (IEEE TIFS 2024) | github.com/Eyr3/FLTracer (README read; code not run) | ≤ 50 % malicious; untargeted + backdoor incl. BadNets; backdoors after convergence (r200) | per-round plaintext updates, previous global, separate reference updates | noise, sign-flip, dirty label, MB, Fang, BadNets, MRA, DBA, Blind, FRA, Patch-BN | yes (per-client flags; type / location) | first flagged round only (no change-point) | gradual drift absorbed; **abrupt benign shift "sacrificed"**; one BDD100K day→night test | **degenerate cross-client channels** (median / MAD of 5); per-client Kalman channel applicable in principle; constants set for 20–100 clients | evaluated on CIFAR-10, not CIFAR-100 | high: export logs to its npz layout, reference-update policy, constants, 6 feature channels | CPU-feasible on our logs (8 logged runs); archived runs have no logs | **only partially** (out-of-regime constants or a re-calibration we would have to define) | **PARTIAL** (closest prior for attribution + attack time) |
| **FLDetector** (KDD 2022) | github.com/zaixizhang/FLDetector (stated in arXiv v1; not inspected) | model poisoning (Fang, Scaling, DBA, ALIE), 28 % malicious; theory IID | per-round updates + global trajectory (L-BFGS Hessian, window N = 10) | untargeted, backdoor | yes (k-means / gap statistic over clients; removes and restarts) | **no** ("ignorant of when the attack starts"); detection from round 50 | not evaluated | clustering of 5 scores is degenerate; the consistency score itself is per-client | evaluated on other datasets (not CIFAR-100 per §A1) | medium (estimate; **P-54: feasibility audit only, no implementation** — doc 27) | CPU: Hessian-vector products on 11 M parameters, 5 clients × 50 rounds — hours | partial: the **per-client consistency score** is a fair *component* baseline; the published decision rule is not | **PARTIAL** (closest prior for temporal evidence) |
| DeepSight (NDSS 2022) | not found (UNVERIFIED) | backdoor only; N_A < N/2 | client models, random inputs | backdoors | per-model labels; cluster exclusion | no | static non-IID (central claim); no temporal drift | HDBSCAN on 5 models degenerate | evaluated on CIFAR-10 etc. (UNVERIFIED for CIFAR-100) | high | moderate | no | RELATED ONLY (classifier-row / NEUP evidence) |
| FedREDefense (ICML 2024) | github.com/xyq7/FedREDefense (read) | **model poisoning**; data poisoning "cannot be detected by our defense" (Sec. 6.1); static benign distributions | per-client distillation each round | Fang, LIE, Min-Max, Min-Sum, MPAF, Scaling, DBA | yes (removal) | no | assumes static | per-client: possible | – | high (bi-level distillation) | GPU-heavy | **no** for data poisoning / drift | INCOMPATIBLE with our attack arms |
| FoolsGold (RAID 2020) | github.com/DistributedML/FoolsGold | sybils (multiple colluders); single attacker: "malicious and honest clients are indistinguishable" | historical aggregated updates, output layer | label flip, backdoor | no (learning-rate reweighting) | no | no | single attacker out of scope | – | low | low | no | INCOMPATIBLE (single attacker) |
| FLForensics (NeurIPS 2025) | not found (UNVERIFIED) | targeted poisoning; post hoc from a misclassified target input | stored checkpoints + target input | Scaling, ALIE, Edge | yes (post hoc) | no | no | HDBSCAN on 5 clients degenerate | – | high | moderate | no (post hoc, different question) | RELATED ONLY |
| BackdoorIndicator (USENIX Sec 2024) | not found (UNVERIFIED) | backdoors; server plants an OOD indicator task | modifies the broadcast model; server OOD data | backdoors | per-update flag | no | no (FPR 15–25 % default) | per-update: possible | evaluated on CIFAR-100 (as OOD source) | high; **changes training** | GPU (new runs) | no (active, modifies the model) | RELATED ONLY |
| CrowdGuard (NDSS 2024) | not found (UNVERIFIED) | backdoors; trusted client TEEs | client-side validation with local data | backdoors (+2 untargeted) | per-model accept / reject | no | static non-IID | needs other clients' data | – | very high | TEEs | no | INCOMPATIBLE (threat model) |

## 3. Selection for the workshop paper (PROPOSED)

- **Direct baselines (all CPU, already implemented, zero GPU):** FedTrace v3; its magnitude-only / temporal-residual-only
  ablations; ZP-norm; CUSUM-norm; reference ablations (leave-one-out, trimmed, historical). Report Page–Hinkley in the
  text only (dominated).
- **Partial baselines:** FLDetector — feasibility audit only (P-54, doc 27), no implementation in Prompt 33;
  FLTracer — qualitative comparator (P-47) unless a fair quantitative adaptation becomes feasible.
- **Related work only:** DeepSight, FLForensics, BackdoorIndicator; **incompatible, cite with reason:** FedREDefense
  (data poisoning excluded by its authors), FoolsGold (single attacker), CrowdGuard (TEE threat model).
- Never claim superiority over FLTracer / FLDetector without a direct, fair, approved comparison.
