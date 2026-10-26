# 27 — FLDetector Feasibility Audit (P-54; no implementation, no execution)

Prepared 2026-10-26 (Prompt 33, Task 8). Sources: FLDetector paper (KDD 2022; full text verified in checklist §A1)
and its official repository `github.com/zaixizhang/FLDetector` — README, `train_cifar.py`, `nd_aggregation.py`
read as text on 2026-10-26 (no code run). Labels: VERIFIED-SOURCE (read), CLAUDE-REPORTED, PROPOSED.

## 1. The original score (VERIFIED-SOURCE)

| Element | Paper / code |
|---|---|
| Client message | **FedSGD**: each worker sends one gradient of its whole shard per round (`train_cifar.py`; `batch_x = each_worker_data[i][:]`, one `backward`) |
| Predicted update | ĝ_i^t = g_i^{t−1} + H̃^t (w_t − w_{t−1}) — `pred_grad.append(old_gradients[i] + hvp)` (`nd_aggregation.py`); H̃ from `lbfgs(args, weight_record, grad_record, weight - last_weight)`: compact L-BFGS over the last **10** pairs of global weight differences and aggregated-gradient differences (one Hessian approximation shared by all clients) |
| Distance | L2 norm of predicted − received per worker; **normalised to sum to 1 across workers** (`distance / np.sum(distance)`) |
| Score | per-worker sum of the last 10 normalised distances (`np.sum(malicious_score[-10:], axis=0)`); detection from `e > 20`, first decision once ≥ 11 rows (≈ round 31) |
| Decision | min–max normalised scores; gap statistic over k = 1…7 with 10 uniform reference sets; if k > 1, k-means (k = 2), higher-score cluster = malicious; training stops at the first detection (`train_cifar.py`) |
| Onset | none (paper: "ignorant of when the attack starts or ends") |
| Hard-coded assumptions | `range(100)` workers and `np.ones(100)` in the decision code; MXNet 1.9.1; aggregation `median` by default |

## 2. Can it be computed from our logs? (CLAUDE-REPORTED)

| Requirement | Our artifacts |
|---|---|
| g_i^t per client and round | yes, as the **pseudo-gradient** Δ_i^t = u_i^t − θ_t (one local epoch of SGD, not one gradient); available in the 8 logged development runs (`logs_rev/X*/update_log`) and in the proposed W runs (logging on); **not** in the 21 archived runs (no update logs) |
| g_i^{t−1} | yes (every client submits every round) |
| Global weight differences w_t − w_{t−1} | yes (every global state logged) |
| Aggregated "gradient" differences | yes (FedAvg update = sample-weighted mean of Δ_i^t, recomputable bit-exactly from the logs, L-21) |
| Additional stored state | only the 10-round window of global pairs and the previous round's client updates — all derivable from the logs; nothing extra needs to be stored during training |

## 3. Differences between the original and any K = 5 monitor-only adaptation

1. **FedSGD vs FedAvg:** the Cauchy mean-value prediction g^{t−1} + H(w_t − w_{t−1}) is derived for single gradients;
   one-epoch pseudo-gradients make it an approximation the authors did not evaluate.
2. **Decision stage is not applicable at K = 5:** the gap statistic over k = 1…7 cannot be evaluated with 5 points
   (k ≤ n required), and k-means on 5 normalised scores is degenerate; any cap on k or replacement of the decision rule
   is our modification.
3. **Normalisation across 5 workers:** one attacker is 20 % of the mass of the normalised distance; scores of honest
   clients are coupled to the attacker's (a shared-reference effect analogous to doc 25).
4. **Timing:** the original starts scoring after round 20 and decides from ≈ round 31; our onset is round 20, so a
   faithful window would include only post-onset rounds — attack onset is not estimable either way.
5. **Monitor-only:** the original stops training at detection (and the paper removes clients); a monitor-only score
   changes nothing in training (fair for comparison, but not the published pipeline).
6. **Aggregation:** the reference code aggregates by coordinate-wise median; our runs use FedAvg.
7. **Attacks:** evaluated on model poisoning (Fang, Scaling, DBA, ALIE); our data-poisoning and drift arms are outside
   its evaluated threat model.

## 4. Cost estimate (PROPOSED)

- Implementation: ≈ 1 working day (PyTorch re-implementation of the compact L-BFGS product on flattened 11.2 M-parameter
  vectors, the distance and window score; unit tests against a small quadratic model where the Hessian is known).
- CPU: per run 50 rounds × 6 state loads (≈ 45 MB each, ≈ 13 GiB per run) + L-BFGS products with 10 pairs (≈ 10⁹
  flops per round) ≈ 5–10 min per run; 8 development runs (+ 6 W runs) ≈ 1–2 h total.
- No GPU.

## 5. Recommendation

A **component-level** diagnostic is feasible and fair in a narrow sense: compute FLDetector's per-client
consistency score (without its decision stage) on the logged runs and report, per run, the rank of the intervened
client and whether honest / drifting clients receive the highest scores. It would answer one reviewer question ("does
the closest temporal-consistency evidence avoid the contamination / drift failures?") but it **cannot** be presented as
a reproduction of FLDetector, gives no onset estimate, is unavailable for the archived runs, and costs ≈ 1 day that
competes with drafting. **Recommendation: do not implement before the submission unless the GPU confirmation is
declined (freeing time); cite FLDetector qualitatively (Table 2) with the differences above.** Decision: P-58.
