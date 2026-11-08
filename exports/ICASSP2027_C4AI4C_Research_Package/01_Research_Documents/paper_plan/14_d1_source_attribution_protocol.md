# 14 — D1 Explaining-Away Source Localization: Exploratory Feasibility Protocol (development only)

Status: **FROZEN 2026-10-25, before any D1 outcome was computed** on a development run (no
`results_rev/pilot/source_attribution/` output existed at freeze). Implementation `src/rev/source_attribution.py`
(SHA-256 `9ea33149e6db9e0a082ad9950e43e7fbbdb14b05b6d62e6d00ac291b2a4d8a2c`), tested on synthetic data only
(`tests/test_rev_source_attribution.py`). Changes only by dated amendment (§12).

This is an exploratory feasibility audit (PI P-37, Prompt 29), not a detector evaluation and not independent
validation. Seeds 123 and 2026 are DEVELOPMENT seeds; sealed confirmation seeds are not used. CPU only, existing
traces only. D1 is not selected as a method by this protocol, whatever its outcome.

## 1. Question

When a client's behaviour changes, can the server localize the **source** of the shared-reference movement — the
client whose removal from the reference best restores every *other* client to its own calibrated behaviour — and does
that add information beyond the trivial order-statistic influence of each client on the median? Source localization
is not malicious attribution (§9).

## 2. Inputs (online information)

Per run, only the server-side trace: for each round r and client k, the self-cosine t_k(r) (cosine between k's update
at r and at r − 1; computable by the server from submitted plaintext updates) and the update norm n_k(r). **Not
inputs:** client identity of the attacker or drifted client, the onset, the matched clean run, attack/drift labels,
or any round after r (scores at r use rounds ≤ r). The ranking function `analyse(rows, estimators, eval_round)` has
no other argument; ground truth is used only by `evaluate` (retrospective). Nothing is fitted; all constants below
are fixed a priori.

## 3. Definitions

| # | Item | Definition |
|---|---|---|
| 1 | Candidate source set | C_r = all clients with a defined t_k(r) in round r (all 5 in every development run) |
| 2 | Reference estimator E | **MED** (primary): median of the set (mean of the two middle values for 4 members); sensitivity: **TRIM** (drop one minimum and one maximum, mean of the rest; for 4 members this equals MED) and **MEAN** |
| 3 | Source-exclusion operation | For source j: reference over C_r \ {j}; residual of every other client k ≠ j: d_k(r; C \ {j}) = E(C_r \ {j}) − t_k(r) |
| — | Inclusive residual | d_k(r; C) = E(C_r) − t_k(r) |
| — | Calibration | For every (k, set rule), μ and sample sd of d_k over rounds 1…19 (< T_ref = 20) **under the same set rule** (the excluded source j is also excluded in calibration); z_k = (d_k − μ)/sd. This absorbs the stationary mechanical offset that removing j creates |
| 4 | Scoring statistic (B) | I_j(r) = mean over k ∈ C_r \ {j} with both z defined of [ z_k(r; C)² − z_k(r; C \ {j})² ]. **Improvement in consistency** = positive I_j: removing j lowers the other clients' squared standardised deviation from their own calibrated behaviour (two-sided, so upward and downward movements both count). j's own residual is excluded from both terms; raw update magnitude does not enter |
| 5 | Ranking rule | Descending cumulative score at the evaluation round |
| 6 | Tie-breaking | Scores within 1e-12 are exact ties, broken by client id and **flagged**; near tie = margin ≤ 1 % of the top score (reported, not resolved); undefined scores ranked last and flagged |
| 7 | Time aggregation | Online cumulative mean over rounds T_ref = 20 … r; primary evaluation round **r = 29** (rounds 20–29); per-round scores and cumulative tops reported for every round 20–49 |
| 8 | Missing values | A residual is undefined if t is missing, the set has < 2 members, fewer than 2 calibration values exist or the calibration sd ≤ 1e-12; an I_j term is dropped if either z is undefined; I_j(r) undefined if < 2 terms; undefined rounds are skipped in the cumulative mean; a candidate with no defined round is undefined. Nothing is imputed |

### Analysis A — standard leave-one-out candidate scoring (distinct from B)

A_k(r) = z_k(r; C \ {k})², the candidate scored against the reference of the *other* clients (calibrated under the same
rule). A asks "is k anomalous relative to the others?"; B asks "does removing j make the others normal?". They are not
equivalent: an already-extreme client that moves further out raises A but leaves the median, and hence B, unchanged
(synthetic test `test_A_and_B_differ_when_an_extreme_client_moves_further_out`).

### Order-statistic leverage baselines (descriptive; Task 4)

| Baseline | Per-round quantity | Aggregate (rounds 20…r) |
|---|---|---|
| ABS | D_j(r) = E(C_r) − E(C_r \ {j}) (raw self-cosine units) | mean |D_j| |
| SIGNED | D_j(r) | |mean D_j| |
| SELFCOS | t_j(r) | |mean t_j − μ_pre| / sd_pre (rounds 1–19) |
| NORM | n_j(r) | |mean n_j − μ_pre| / sd_pre (rounds 1–19) |

All are ranked descending with the same tie rule. They are conventional descriptive baselines, not tuned detectors.

Known algebra (stated before outcomes): for MED with |C| = 5 and sorted values a1 ≤ … ≤ a5, E(C \ {j}) is
(a3 + a4)/2 for j at position 1 or 2, (a2 + a4)/2 at position 3, and (a2 + a3)/2 at position 4 or 5. Within one round,
same-side candidates share the exclusion reference; B separates them only through their own-term exclusion and their
calibration history. Ties and near-ties are therefore expected and are reported.

## 4. Runs

The eight existing development traces: X1 clean, X2 BadNets, X3 relabel-only, X4 benign drift × seeds 123, 2026.
Retrospective truth (evaluation only): intervention source client_002 in X2/X3/X4 (malicious in X2/X3, benign in X4);
no source in X1.

## 5. Outputs per run

True source (evaluation only); D1 (B, MED) top source, full ranking, rank of client_002, margin first − second;
stability (per-round top vs cumulative top in rounds 20–29, distinct per-round tops, round from which the cumulative
top is unchanged through 29, exact / near ties, order-statistic position of each per-round top); tops under TRIM and
MEAN; the same rankings for A, ABS, SIGNED, SELFCOS, NORM; Kendall τ of B vs ABS and B vs SIGNED. Clean runs:
rankings reported as descriptive diagnostics only; no source is claimed.

## 6. Three outcomes kept separate (Task 6)

1. **Source localization** — the D1 top-ranked client (most explains the reference movement).
2. **Anomaly detection** — whether a change was confirmed, taken from the existing frozen detectors (v3, norm z-score
   h 1.75, norm CUSUM h 11.75; L-69) and, descriptively, whether the D1 top score exceeds both clean runs' top score
   at the same round. The latter is a description, not a decision rule.
3. **Malicious attribution** — D1 provides none. A D1 top-ranked client is never reported as malicious; in the drift
   runs a correct localization of client_002 is a localization of a benign change.

## 7. Prior knowledge (disclosed before outcomes)

Known from earlier diagnostics (L-60, L-66, L-70): at seed 2026 client_002's change raises the inclusive median for
all other clients (+1.1 … +2.6 z, every arm); at seed 123 client_002 sits above the median in the clean run and the
movement comes from round 20 only. Known v3 / baseline decisions (L-69). Hence D1 is expected to be favoured at seed
2026 and stressed at seed 123; the window start T_ref = 20 coincides with the true onset in every run, which favours
any cumulative score. These are reasons for caution, not part of the method.

## 8. Comparison and reporting rules

D1 is compared with ABS, SIGNED, SELFCOS, NORM and A by the rank of the true source at round 29 (evaluation only).
Reported as CLAUDE-REPORTED, DEVELOPMENT ONLY, POST-HOC EXPLORATORY.

## 9. Interpretation limits

Two seeds, one intervened client, one onset, K = 5. No claim of novelty, generalisation or malice. A favourable
outcome would justify only a further falsification test, not adoption.

## 10. Primary falsification criteria (FIXED; computed by `verdicts`)

- **F1 (motivating case):** FALSIFIED if B-MED does not rank client_002 first at round 29 in either seed-2026
  poisoning run (BadNets, relabel-only).
- **F2 (added information):** against each of ABS and SIGNED separately, over the six intervened runs: "ADDS RANK
  INFORMATION" only if B-MED ranks client_002 strictly better in ≥ 1 run and strictly worse in none; otherwise "NO
  EVIDENCE OF ADDED INFORMATION".
- **F3 (misleading winner):** if any clean run's B-MED top − second margin at round 29 is ≥ the smallest margin of a
  correct B-MED localization, the localization is recorded as **not more decisive than a no-change "winner"**.
- **F4 (instability):** a run is UNSTABLE if the B-MED per-round top equals its round-29 cumulative top in fewer than
  5 of the 10 rounds 20–29.
- **F5 (estimator dependence):** recorded if the B top at round 29 differs among MED, TRIM and MEAN in a seed-2026
  poisoning run.

## 11. Command (CPU)

```
python -m src.rev.source_attribution --seeds 123 2026
```

## 12. Amendments

None.
