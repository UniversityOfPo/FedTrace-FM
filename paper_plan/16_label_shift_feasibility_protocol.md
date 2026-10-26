# 16 — Label-Shift-Invariant Conditional-Shift Feasibility Audit: Exploratory Protocol (development only)

Status: **FROZEN 2026-10-25, before any logit or residual was computed** on a development model (only the probe
audit, which reads no predictions, had run). Implementation `src/rev/label_shift.py` (SHA-256
`6eb981efb8bf25791adbc389beb01479865e33442354d16321ea8b70814b1985`), probe audit `src/rev/probe_audit.py`
(`9505be9e…18e625`), synthetic tests `tests/test_rev_label_shift.py` (`f31ec61c…4fc1`). Changes only by dated
amendment (§12).

Hypothesis-falsification exercise under PI P-42 – P-44 (Prompt 31): not detector development, not independent
validation. Seeds 123 and 2026 are DEVELOPMENT seeds; sealed confirmation seeds are not used. CPU only; existing
model states only; no retraining; no new FL run.

## 1. Threat model (probe-augmented, disclosed)

Doc 15 Amendment 1 (A1) observation model plus a server-held, **unlabelled** image set A fixed before training and
disjoint from every client's data and from final evaluation. Here A = the run seed's 5,000-image validation subset
(held centrally by the training runner, never partitioned to clients), minus exact pixel duplicates of client-training
or test images. Labels of A exist on disk but are never read by `label_shift.py`.

## 2. Hypothesis H-LS (falsifiable)

**Ideal identity.** Let p_old(x, y) = p_old(y) p(x∣y) and p_new(x, y) = p_new(y) p(x∣y) with the **same**
class-conditionals p(x∣y) (pure label shift), and supp p_new(y) ⊆ supp p_old(y), p(x) > 0. Then by Bayes' rule

    p_new(y∣x) = p(x∣y) p_new(y) / Σ_c p(x∣c) p_new(c) = p_old(y∣x) w(y) / Σ_c p_old(c∣x) w(c),  w(y) = p_new(y)/p_old(y),

i.e. in logit space z_new(x) = z_old(x) + log w + const(x): a **class-only** additive correction, the same for every
x. It fails under (i) covariate shift (p(x) changes, p(y∣x) fixed: the ideal correction is w ≡ 1, but a model trained
on the shifted p(x) changes in x-dependent ways off the common support) and (ii) concept shift (p(y∣x) changes in an
x-dependent way, e.g. relabelling source-class images to class 0, or making a trigger predictive of class 0), which no
class-only w can represent in general.

**Ideal vs actual.** The identity concerns Bayes posteriors. A client's submitted model is the result of one local
epoch of SGD from θ_r, with BatchNorm statistics re-estimated on its data; it is neither the Bayes posterior of its
data nor a reweighting of θ_r. Finite-step training, feature-extractor changes, BN-statistic changes, memorisation of
duplicated samples and forgetting of absent classes can all produce x-dependent changes **under pure label shift**.
Whether they are small relative to concept-shift changes is the empirical question.

**The development intervention only approximates label shift.** X4 (protocol 13) replaces each non-retained-class
sample by a uniformly drawn sample of a retained class from the client's own data: class priors change (≈ 44–47 % of
samples, 25/53 or 30/57 classes retained), class-conditionals are the client's own empirical ones, but with
duplication (≈ 0.53–0.56 unique) — so p(x∣y) is the same in law but not in sample, and over-fitting to duplicates is
an x-dependent effect. Relabel-only (X3) is concept shift on the poisoned indices; BadNets (X2) is concept shift on
the trigger region plus the relabelled samples.

**H-LS (operational).** For the intervened client, the held-out residual left after the best class-only correction of
the pre-round global posterior increases less under benign class-composition drift than under relabel-only
poisoning, in both development seeds, and the correction (not the probe set alone) is responsible for the
separation.

## 3. Probe data (provenance; `src/rev/probe_audit.py`, `results_rev/pilot/label_shift/probe_audit.json`)

| Item | Seed 123 | Seed 2026 |
|---|---|---|
| Split reconstruction | 45,000 / 5,000 stratified (sklearn, seed = run seed), replayed exactly; the re-derived Dirichlet partition equals the saved partition in all four arms | same |
| Probe pool | 5,000 validation images (training file indices) | 5,000 |
| Index overlap with any client / with the 45,000 training subset | 0 / 0 | 0 / 0 |
| Exact pixel duplicates excluded (label-free rule) | 2 of client-training images (val 22657 ≡ train 8599; 31828 ≡ 16646), 2 of test images (15597 ≡ test 9524; 30697 ≡ test 2996) | 2 of client-training images (20635 ≡ 32666; 28293 ≡ 41860) |
| Final probes | 4,996 (fit 2,417 / eval 2,579) | 4,998 (fit 2,506 / eval 2,492) |
| Manifest (v2) SHA-256 | `39e2a6a197552398606e09e6c009517f082be878cdfc69a534001c2a523a63f4` | `eff184c28c5a4f07299d536013bbe2db26340231a7d0d149a1f1f6912560ef41` |
| Remaining exact duplicates with client data / test | 0 / 0 | 0 / 0 |

Fit / eval assignment: first byte of SHA-256("FedTrace-probe-split-v1|seed|index") even → fit; label-free and fixed.
**Not checked:** near-duplicates (CIFAR is known to contain them); final evaluation images are never used here, so
near-duplicates can affect only probe-side statistics, not any reported accuracy. The validation subset was also the
server's per-round monitoring set (no model selection; the final model is the last round).

Model states (gate G4/G5): all 84 states per run (pre-round globals of rounds 16–29 and the five submissions of
rounds 16–29) exist in sealed logs; submissions are full states (parameters and BN buffers), load strictly into
`resnet18_cifar(100)`; content SHA-256 is verified for every state before use.

## 4. Quantities (exact)

For run, client k, round r ∈ 16…29: z_g = logits of the pre-round global θ_r (logged global of round r − 1),
z_k = logits of k's submitted state; both on every probe in evaluation mode (fp32 forward, float64 analysis);
q = softmax(z).

| Quantity | Definition |
|---|---|
| Reference posterior | q_g(·∣x) of the pre-round global model |
| Uncorrected distance D | mean over EVAL of KL(q_k ∥ q_g) |
| Class-only correction | v̂ = argmin_v mean over FIT of CE(q_k, softmax(z_g + v)) + (λ/2)‖v‖², λ = 1e-6 (numerical regularisation only; makes v unique and finite when a class's probability vanishes; minimiser has mean(v) ≈ 0 since softmax is shift-invariant). Convex; L-BFGS, float64, strong-Wolfe line search, max 1,000 iterations, converged if max |∇| < 1e-6. **Unsupervised**: uses the two models' predictions only, never labels |
| Held-out residual R | mean over EVAL of KL(q_k ∥ softmax(z_g + v̂)) |
| In-sample residual R_fit | the same over FIT (optimistic; reported to measure the fit's optimism) |
| Explained fraction | 1 − R / D |
| Top-1 disagreement | fraction of EVAL probes where argmax z_k ≠ argmax z_g (descriptive) |

Clients: all five, every round 16–29 (the intervened client_002 is the primary unit; honest clients give the clean
variation). Rounds: 16–19 pre-onset (own-history baseline), 20 reported only, 21–29 post-onset window.

## 5. Aggregation

- **Primary (deployable, no counterfactual):** own-history excess E_X(k) = mean_{r=21..29} X_k(r) − mean_{r=16..19}
  X_k(r), for X ∈ {R, D, R_fit}.
- **Diagnostic (uses the matched clean run):** M_X(k) = mean_{r=21..29} [X_k^arm(r) − X_k^clean(r)].
- **Clean scale:** S_X = pooled within-client sd of X_k(r), r = 21..29, over the five clients of the matched clean run
  (45 values, 5 client means removed, df = 40).
- **Separation index:** Sep_X = (E_X(relabel) − E_X(drift)) / S_X for client_002.

## 6. Sensitivity analyses (fixed)

- Probe size (evaluation): nested EVAL subsets of 250, 500, 1,000 by the label-free order
  SHA-256("FedTrace-probe-order-v1|seed|index"), plus all.
- Probe size (fit): v̂ refitted on nested FIT subsets of 250, 500, 1,000 (client_002, rounds 21–29), R on full EVAL.
- Probe composition (label-free): EVAL split by a second hash bit (two random halves) and by θ_16's predicted class
  (0–49 vs 50–99); v̂ unchanged, R and D recomputed on each half.

## 7. What is known before computation (disclosed)

All non-probe outcomes of these runs (L-63 – L-76). No logit, prediction distance or residual of any development
model on these probes has been computed. The protocol was written knowing that X4 is a class-composition change and
X3 / X2 are poisoning (by design).

## 8. Falsification criteria (FIXED; computed by `verdicts`)

| ID | Check | Rule |
|---|---|---|
| A | Correction reduces benign-drift residuals | PASS iff, in both seeds, R < D for client_002 in every drift round 21–29 **and** M_R(drift) < M_D(drift) |
| B | **Primary:** drift below relabel-only | PASS iff E_R(drift) < E_R(relabel) for client_002 in both seeds |
| C | BadNets sensitivity without trigger probes | PASS iff E_R(BadNets) − E_R(drift) ≥ 2 S_R in both seeds |
| D | Clean clients' unexplained residuals | SIMILAR iff max |E_R| over honest clients (clean run, all five; relabel run, four honest) ≥ E_R(relabel, client_002) in either seed |
| E | Probe-composition dependence | STRONG DEPENDENCE iff the B ordering differs on any of the four EVAL halves in any seed |
| F | Held-out vs in-sample | SEPARATION DISAPPEARS iff B holds with R_fit in both seeds but not with R |
| G | Uncorrected baseline | CORRECTION ADDS SEPARATION iff Sep_R > Sep_D in both seeds; otherwise BASELINE EQUALLY GOOD OR BETTER |
| H-LS | Core verdict | NOT FALSIFIED (2 development seeds) iff A PASS, B PASS and F CONSISTENT; otherwise FALSIFIED |

The constant 2 in C is a fixed reporting scale, not a tuned threshold; no criterion is a detector decision.
A large residual is never interpreted as malice (Doc 15, A4).

## 9. Missing data

A state whose content hash fails, or a missing logits file, makes that (run, client, round) undefined; any undefined
value in a window used by a criterion makes the criterion NOT EVALUABLE (and H-LS FALSIFIED as not supported).
Non-converged fits are reported with their gradient norm and used as computed.

## 10. Commands (CPU)

```
python -m src.rev.label_shift --stage logits --seeds 123 2026     # ≈ 672 model evaluations on 4,996–4,998 probes
python -m src.rev.label_shift --stage analyse --seeds 123 2026
```

## 11. Interpretation limits

Two seeds, one intervened client, one onset, K = 5, one drift type that only approximates label shift, probes from
the training distribution. A NOT FALSIFIED outcome would justify only a further test on fresh benign types (label
noise, new-class acquisition) and new seeds; a FALSIFIED outcome is reported as such and the method is not redesigned
within Prompt 31.

## 12. Amendments

None.
