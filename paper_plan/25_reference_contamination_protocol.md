# 25 — Reference Contamination in FedTrace v3: Formal Statement and Frozen CPU Falsification Protocol

Status: **FROZEN 2026-10-26, before any contamination or unified-table output was computed on a real trace.**
Implementation `src/rev/contamination.py` (SHA-256 `368c268e5fd6500c1231ddc9a1d67d8cae01f52c6a3308bee8fbb596dc3453a4`),
`src/rev/unified_outcomes.py` (`a3c7b106…f9fe`), synthetic tests `tests/test_rev_contamination.py` (`7fe68dc4…02dc`).
P-52 (conditional approval). CPU only; existing traces only; nothing fitted; no threshold chosen from outcomes.
Labels: **EXACT** (algebraic identity), **CONDITIONAL** (exact under stated assumptions), **EMPIRICAL** (tested here).

Disclosed before freezing: the deployable proxy was first written with j's absolute historical level; the synthetic
common-trend test showed that this conflates a common-mode drift with a single-source displacement, so the proxy was
changed to j's historical **offset** from the others' median (§2.4). This happened on synthetic data only. Known
beforehand from development diagnostics: the seed-2026 contamination (L-43 – L-45, L-59) and its absence at seed 123
(L-54); none of the 21 archived traces had been examined for crossings or displacement.

## 1. Setting (exact v3 reference)

Round r, clients i = 1..K with defined self-cosines t_i(r); v3's round reference is the **inclusive** median
c(r) = med{t_1(r), …, t_K(r)}; residual d_k(r) = c(r) − t_k(r); calibration μ_k, σ_k = mean and sample sd of d_k over
rounds 1..19; signed z_k = (d_k − μ_k)/σ_k; v3 temporal score z_t_res = max(0, z_k); magnitude score z_m (two-sided
norm z); score S = max(z_m, z_t_res); confirmation when S ≥ τ = 3 for P = 3 consecutive rounds; attribution = first
confirmed client (tie-break by streak mean); onset estimate = start of the confirming streak.

## 2. Mechanism

### 2.1 Direct contribution of client j to the reference (Lemma A — EXACT)

Fix the other K − 1 values, sorted s_1 ≤ … ≤ s_{K−1}, and view c as a function of j's value x.
- K odd (K = 2m − 1): c(x) = clip(x, s_{m−1}, s_m).
- K even (K = 2q): c(x) = ½[clip(x, s_{q−1}, s_q) + clip(x, s_q, s_{q+1})].

Consequences (exact): c is non-decreasing in x; moving j from x0 to x1 changes the reference by
Δc_j = c(x1) − c(x0) with |Δc_j| ≤ min(|x1 − x0|, s_m − s_{m−1}) (odd K); **Δc_j ≠ 0 only if j enters, leaves or
traverses the others' middle band [s_{m−1}, s_m]** ("crossing"); if Δc_j ≠ 0 its sign is sign(x1 − x0). A move
entirely above or entirely below the band leaves the reference unchanged. *This is an elementary property of the
median; it is stated for completeness and is not claimed as a contribution.*

### 2.2 Transfer to other clients' residuals (Lemma B — EXACT, holding the others' values fixed)

For every k ≠ j: Δd_k = Δc_j (identical for all other clients). In v3 units Δz_k = Δc_j / σ_k. **All heterogeneity
of the impact across honest clients comes from σ_k**: clients whose residual was most stable during calibration (small
σ_k) are the most affected. (The source's own calibration σ_j is small when j was the median client, L-46.)

### 2.3 One-sided score, persistence and attribution (Proposition 1 — CONDITIONAL)

With the score at round r written as S_k = max(z_m,k, max(0, z_k)):
1. Only an **upward** displacement (Δc_j > 0) can raise an honest client's temporal score; a downward one cannot
   (exact).
2. Holding everything else fixed, j's displacement is **necessary** for k's confirmation at round a (streak
   R = {a − P + 1, …, a}) iff, for at least one r ∈ R, max(z_m,k(r), max(0, z_k(r) − Δz_{k←j}(r))) < τ (exact, given
   the frozen rule).
3. An honest false *accusation* (wrong attribution) through this channel additionally requires: (i) an upward
   crossing that persists for P consecutive rounds (the band can move with the others — dynamic coupling, doc 15
   Lemma 1); (ii) Δc_j / σ_k ≥ τ − z_k^{(no-shift)} in each of those rounds (small σ_k or an already elevated
   residual); (iii) k's confirmation precedes j's own (e.g. j's own residual turns negative so its z_t_res = 0 and
   only its norm evidence remains, L-32). **A crossing does not by itself imply a false accusation.**
4. Statements 1–3 are deterministic consequences of the frozen rule; whether real traces satisfy (i)–(iii) is
   EMPIRICAL.

**What is informative beyond elementary median properties** (to be judged by §5): (a) the σ_k-scaling predicts
*which* honest clients are accused; (b) the one-sidedness predicts that only upward crossings contaminate; (c) the
necessity test separates reference-driven from norm-driven accusations; (d) a deployable proxy (§2.4) can flag
contamination online without a counterfactual. If the deployable proxy is no better than a plain description of
reference movement (B0), the mechanism adds no practical information.

### 2.4 Retrospective vs deployable quantities

| Quantity | Definition | Availability |
|---|---|---|
| Δc_j^retro(r) | c(r) − c with t_j(r) replaced by j's value in the matched clean run (others actual) | **retrospective only** (needs the counterfactual; diagnosis, never a detector input) |
| Δc_j^dep(r) | c(r) − c with t_j(r) replaced by x̂_j(r) = m_{−j}(r) + o_j, where m_{−j} is the median of the other clients and o_j the mean over calibration rounds 1..19 of (t_j − m_{−j}) | **deployable** (current round + calibration history) |
| B0: Δc^obs(r) | c(r) − mean over calibration rounds of c | **deployable**, names no source |
| B0′: total median change | c(r) − c^clean(r) | retrospective |
| Δz^total_k(r) | (d_k(r) − d_k^clean(r)) / σ_k | retrospective |

## 3. Traces included

- All **21 archived traces** (doc 18): FedAvg K = 5 seeds 42 / 123 / 2026 (EXP_014–019, 026, 027), E2 (2), E3 (2),
  Multi-Krum (6), K = 10 (E1, 3).
- All **8 development runs** X1–X4 × seeds 123 / 2026.
- Matched clean counterfactual: X1 of the same seed when trace rows are identical before the onset (verified per run
  by `matched_clean`; expected for the 10 FedAvg K = 5 archived runs at seeds 123 / 2026 and the 6 intervened
  development runs).
- Truth (evaluation only): archived — the trace's attack fields; development — client_002 from round 20 (malicious
  for BadNets / relabel-only; benign for drift).
- Clients and rounds: all clients, rounds ≥ 20 (T_ref); displacement tests on rounds onset … onset + 9.

## 4. Analyses

| ID | Question | Quantity | Units |
|---|---|---|---|
| A | Implementation correctness (not evidence) | max |clip form − recomputed median| over all runs, rounds, j | value |
| B | Does the source's displacement account for honest residual change? (retrospective) | explained fraction EF = 1 − Σ(Δz^total − Δz^retro_src)² / Σ(Δz^total)², honest clients, matched runs; compared with B0′ | client-rounds (descriptive) |
| C | Does the deployable proxy predict the direction and magnitude of the retrospective displacement? | sign agreement of Δz^dep_src vs Δz^retro_src on honest client-rounds with |Δz^retro| ≥ 0.5; MAE vs retro; same for B0 | client-rounds (descriptive; no CI) |
| D | Are the observed honest accusations explained? | for every v3 honest confirmation in all 29 traces: classification (t_res-necessary vs norm-driven: z_m ≥ τ in every streak round); deployable necessity test over all j ≠ k (best source j*); B0 necessity test; retrospective test where matched | alarm events |
| E | Reference movement without false alarms | honest client-rounds with max_j Δz^dep ≥ 1, split by confirmation; runs with an upward source crossing but no accusation | client-rounds / runs |
| F | Non-triviality against B0 in runs without intervention | per clean run (X1 ×2, Multi-Krum clean ×2, E1 clean K = 10): rate of honest client-rounds flagged (≥ 1 z) by the mechanism (max over j of Δz^dep) vs by |B0| | rates per run |

## 5. Primary prediction, competing explanation and failure criteria (FIXED)

- **Primary question:** can the mechanism predict the direction and magnitude of reference-induced residual changes,
  and does it explain observed honest accusations beyond a trivial median-displacement description?
- **Competing explanations:** (B0) reference movement without a named source; (norm) honest accusations driven by
  z_m (e.g. late-training norm drift, L-11, L-24); (own change) the honest client's own self-cosine change.
- **H-C2 (direction / magnitude, deployable):** PASS iff pooled sign agreement ≥ 0.90 **and** MAE(dep) < MAE(B0).
- **H-C3 (explanation of accusations):** PASS iff every t_res-necessary honest confirmation is explained by the
  deployable necessity test and, in intervened runs, the best source j* is the intervened client; FAIL otherwise
  (each failure listed). Norm-driven confirmations are reported as outside the mechanism.
- **H-C4 (non-triviality):** PASS iff the mechanism's flag rate is lower than B0's in all clean runs but at most one.
- **Overall:** "mechanism supported (development / archived scope)" iff H-C2 and H-C3 PASS; "informative beyond B0"
  iff additionally H-C4 PASS. Otherwise reported as not supported, without redesign within Prompt 33.
- Fixed reporting constants: 0.5 z (sign-test inclusion), 1.0 z (affected), 0.90 (sign agreement); not tuned.

## 6. Unified outcome table (descriptive)

`src/rev/unified_outcomes.py`: every trace × {v3, v3 magnitude-only, v3 t_res-only, v3 with leave-one-out /
trimmed / historical reference, ZP-norm h 1.75, CUSUM-norm h 11.75, PH-norm h 6.5}; one row each; columns: family
(attack | aggregation | K), detection, attribution, honest false accusations, onset error, delay, final top-1, ASR
(development only). No pooling across families, aggregation rules or K. Norm thresholds were calibrated on the two
clean development runs (K = 5 FedAvg); rows for Multi-Krum and K = 10 are out of their calibration regime and are
labelled so.

## 7. Calibration uncertainty and temporal dependence

- σ_k is estimated from ≈ 19 strongly autocorrelated rounds (lag-1 0.82–0.91, L-61): analysis D is repeated with σ_k
  scaled by 0.8 and 1.25 (fixed); conclusions that change are labelled fragile.
- Client-rounds are **not** independent replications (temporal autocorrelation, cross-client correlation ≈ 0.95,
  shared global model): all client-round counts are descriptive; no confidence interval or p-value is computed from
  them. Units of replication are runs (seeds), of which there are two development seeds plus the archived seeds 42 /
  123 / 2026 (the same seeds as development, different attack families).

## 8. Statistical reliability audit (Task 6; CLAUDE-REPORTED)

| Item | Status |
|---|---|
| Independent seeds | 123, 2026 (development; also used by the archived runs); 42 (archived development); no confirmation seed used |
| Within-run dependence | lag-1 autocorrelation of clean norm z 0.82–0.91 per client; AR(1) effective n ≈ 1.5–2.9 of 30 rounds (L-61) |
| Cross-client dependence | mean pairwise correlation 0.94 / 0.95 (L-61); all clients share the global model (doc 15 Lemma 1) |
| Calibration sample size | 19 rounds (rounds 1–19), effective n ≈ 2 per client |
| Threshold-selection leakage | ZP / CUSUM / PH thresholds were set on the clean runs of the same seeds and partitions as the attacked runs (L-57) → their honest-alarm counts are optimistic; v3's τ, P predate these runs (frozen in NDSS) |
| Estimable uncertainty | per-seed outcomes; the spread across two seeds; exact binomial intervals for **run-level** counts (e.g. 2/2) are computable but uninformative at n = 2 |
| Not estimable | false-accusation *rates*, detection probabilities, onset-error distributions; any interval built from rounds or client-rounds would treat dependent observations as independent and is not reported |

## 9. Publication-integrity flags (Task 10)

- ICASSP 2027 Author Guidelines require disclosure, in the acknowledgments, of AI-generated content (text, figures,
  images, code), naming the system and the sections; editing assistance disclosure is recommended (doc 17 row 10c).
  This project's analysis code, protocols and drafts were produced with Claude; the authors must write the scientific
  contributions, verify every result and disclose the AI use.
- **CVPR overlap (for PI review; no eligibility statement made):** the workshop paper would publish, in IEEE Xplore,
  the contamination analysis, the matched-counterfactual protocol and the development / archived results. A later
  CVPR 2027 submission reusing these would overlap substantially. CVPR's dual-submission / prior-publication rules
  were not read; ICASSP's dual-submission policy was not found on the pages read. The PI must check both before
  submitting.

## 10. Commands (CPU)

```
python -m src.rev.contamination
python -m src.rev.unified_outcomes
```

## 11. Amendments

### Amendment 1 — 2026-10-26 (before any contamination outcome was seen)

The first execution stopped with an exception while computing calibration offsets for a trace in which a client has
no self-cosine in rounds 1–19 (the seed-42 development trace EXP_014 predates the self-cosine field, doc 18); no
output was written and no outcome was displayed. Missing-data rule added: a trace in which any client lacks
self-cosines in the calibration rounds is recorded as **NOT EVALUABLE** for the mechanism (v3's temporal residual is
undefined there as well) and excluded from analyses A–F; it remains in the unified outcome table. No other change.

### Amendment 2 — 2026-10-26 (Prompt 34, P-57; before any confirmation run)

`src/rev/contamination.py` and `src/rev/unified_outcomes.py` changed **only** by adding the protocol 26 confirmation
run list (`CONF`, `RUNS`, `clean_run_of`, `--confirmation`, separate output files); no computation, constant or
criterion changed. Regression: the development outputs `contamination.json` and `unified_outcomes.json` were
reproduced byte-identically after the change. The analysis is applied unchanged to the confirmation runs.
