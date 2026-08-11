# EXP_015 — Norm-Preserving Directional Attack Traceability Analysis (Paired vs. EXP_007, Seed 42)

**Analysis date:** 2026-08-08
**Subject experiments (paired, seed 42 only):**
- `EXP_007` — Honest FedAvg, Dirichlet α=0.1, 50 rounds, seed 42 (no attack)
- `EXP_015` — Same protocol + `norm_preserving_directional` attack on `client_002`, `attack_start_round=20`, `directional_target_cosine=0.0`

**Data sources (verified, no invented values):**
- `results/federated/cifar100_fedavg_dirichlet_a01_direction_attack_client002_round20_seed42/`
  (`summary.json`, `final_test_metrics.json`, `round_history.csv`,
  `client_round_trace.csv`, `client_partition_summary.csv`)
- `results/federated/cifar100_fedavg_dirichlet_a01_50rounds/` (`EXP_007`)
- `research_plans/EXP_015_Norm_Preserving_Directional_Attack_Plan.md`,
  `analysis/EXP_014_Gaussian_Attack_Traceability_Analysis.md`,
  `analysis/EXP_014_Norm_Aware_Attribution_Study.md`

This is a forensic analysis of an already-completed run — no experiment
was executed to produce it, and no code, config, test, README, or
Research Database file was created or modified. This is a single seed
(42); no statistical-significance claim is made anywhere below.

---

## 1. Pairing Verification

`summary.json` and `client_partition_summary.csv` confirm exact pairing
on every scientific setting with `EXP_007`: `partition="dirichlet"`
(`dirichlet_alpha=0.1` per both configs), `model="resnet18_cifar"`,
`total_parameters=11,220,132`, `num_clients=5`, `clients_per_round=5`,
`communication_rounds_configured/_completed=50/50`, `local_epochs=1`,
`optimizer="sgd"`, `learning_rate=0.01`, `aggregation_strategy="fedavg"`,
`seed=42`. Per-client sample counts are identical to `EXP_007`:
`client_000`: 10,883; `client_001`: 8,003; `client_002`: 8,868;
`client_003`: 8,574; `client_004`: 8,672. All 50 rounds completed with
5/5 successful clients in both runs.

**One honest data point worth flagging directly:** `EXP_015`'s pre-attack
`round_history.csv` (rounds 0–19) is **bit-identical to `EXP_014`'s**
(same honest code path, same seed, attack provably inactive in both)
but is **not** bit-identical to `EXP_007`'s own rounds 0–19 — e.g. at
round 19, `EXP_007` reports validation top-1 46.00% vs. `EXP_014`/`EXP_015`'s
44.48% (a −1.52pp gap that exists **before** the attack ever activates).
This is consistent with this project's already-documented GPU/cuDNN
run-to-run non-determinism across separate process invocations (and,
here, a different code revision — `EXP_007` predates the Phase 5
`update_transform_callback` wiring) rather than any pairing defect: the
configuration is identical, the partition realization is identical, only
the specific floating-point trajectory differs slightly. This −1.52pp
pre-attack gap is used below (§10–11) as a baseline noise reference when
interpreting the post-attack gap.

---

## 2. Attack Execution Verification

`client_round_trace.csv`'s `attack_active` column confirms:
`attack_active=False` for **all** clients at rounds 0–19;
`attack_active=True` **only** for `client_002` at rounds 20–49;
`attack_active=False` for every other client at every round, including
20–49. `directional_target_cosine=0.0` is constant on every row. This
exactly matches the pre-registered `EXP_015` configuration
(`malicious_client_id=client_002`, `attack_start_round=20`).

---

## 3. Norm Preservation (`delta_attack_norm_ratio`)

Over `client_002`'s 30 active rounds (20–49):

| Statistic | Value |
|---|---:|
| Mean | 1.0000000000 (≈1, deviation ~1e-10) |
| Min | 0.9999999997765017 (round 39) |
| Max | 1.000000000147643 (round 44) |

**The `1e-4` relative-tolerance invariant (plan §9) is satisfied by
roughly six orders of magnitude of margin.** This is expected — unlike
`gaussian_update_noise`'s empirical rescaling of independent random
noise, the directional construction is an exact orthonormal rotation
(`analysis/EXP_015_Directional_Attack_Implementation_Verification.md`
§§10–11), so the only source of deviation is `float32`/`float64`
round-off across ~11.2M parameters, which is what the observed ~1e-10
scale reflects.

---

## 4. Directional Severity (`realized_cosine_to_honest`)

Over the same 30 active rounds:

| Statistic | Value |
|---|---:|
| Mean | ≈0.0000000000 (deviation ~1e-10) |
| Min | −9.288771585119106e-10 (round 49) |
| Max | 9.858728581190206e-10 (round 29) |

**The realized cosine matches the configured `target_cosine=0.0` within
roughly seven orders of magnitude tighter than the required `1e-3`
absolute tolerance.** The attack's geometry is behaving exactly as
designed on real, full-scale ResNet-18 data — this is independent,
strong corroboration of
`analysis/EXP_015_Directional_Attack_Implementation_Verification.md`'s
code-level verification, now confirmed on the actual `EXP_015` run
rather than only on synthetic test tensors.

---

## 5. Does the Existing Norm Z-Score Detector Still Trivially Identify the Attacker?

**No — the norm-only shortcut is removed.** Computing
`z_norm(round) = (update_l2_norm(round) − mean_pre) / std_pre` for
`client_002` using its own pre-attack reference (`mean_pre=6.0034`,
`std_pre=0.2690`, identical to `EXP_014`'s since rounds 0–19 are
unaffected):

| Round | 20 | 25 | 30 | 35 | 40 | 45 | 49 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `update_l2_norm` | 5.996 | 6.291 | 6.468 | 6.589 | 6.591 | 6.675 | 7.065 |
| `z_norm` | −0.03 | 1.07 | 1.73 | 2.18 | 2.18 | 2.49 | **3.95** |

`z_norm` **does not exceed the `3` threshold until round 49 — the very
last round of the run** (a **29-round detection delay** relative to the
attack's actual onset at round 20), and only barely (3.95, vs. `EXP_014`'s
every-round `z_norm > 9`). No honest client's `z_norm` ever exceeds `1.3`
in the post-attack window (`client_000` max 0.72, `client_001` never
positive after round 20, `client_003` max 1.06, `client_004` max 1.21),
so a detector using this exact rule would, if anything, get exactly one
correct-but-extremely-late flag at round 49 and nothing before it. **This
is a qualitatively different outcome from `EXP_014`**, where the same
rule flagged `client_002` on every single one of its 30 active rounds.

Mechanistically: `client_002`'s own mean post-attack `update_l2_norm`
(6.5875) is only **+9.7%** above its pre-attack mean (6.0034) — far
smaller than `EXP_014`'s +47.0% — because the attack preserves norm
*relative to that round's own honest counterfactual*, and it is only the
*honest* update norm's own gradual drift over 30 rounds (itself partly a
consequence of the attack's own damage to the global model, §10–11) that
slowly pushes `client_002`'s absolute norm upward late in the run.

---

## 6. Does the Existing Peer-Cosine 3-Sigma Detector Flag Anyone?

**No client is ever flagged.** `naive_flag` is `False` on all 250 rows of
`client_round_trace.csv`; `first_flagged_round` is blank for every
client, including `client_002`.

**Pre/post distributions, all five clients** (means, rounds 0–19 vs.
20–49; pre-attack values identical to `EXP_014` since the honest path is
unaffected):

| Client | Pre-attack mean | Post-attack mean |
|---|---:|---:|
| `client_000` (honest) | 0.1400 | ≈ +0.036 → −0.015 (declining) |
| `client_001` (honest) | 0.1257 | ≈ +0.029 → −0.010 (declining) |
| `client_002` (**malicious**) | 0.1227 | **≈ 0.0000, flat throughout** |
| `client_003` (honest) | 0.1196 | ≈ +0.023 → −0.019 (declining) |
| `client_004` (honest) | 0.1274 | ≈ +0.037 → −0.017 (declining) |

**`client_002` does separate from the honest clients — but not in the way
the detector's fixed, pre-attack-derived threshold (−0.0687) can use.**
Every honest client follows the same declining trajectory already
documented for `EXP_014` (positive early, crossing toward small negative
values by round 49) — this is the shared, attack-independent secular
trend. `client_002`'s post-attack `peer_mean_cosine_similarity` instead
stays **pinned within ±0.0007 of exactly zero for all 30 rounds**,
tracking neither the honest decline nor any large deviation. Concretely,
this produces a **sign-crossing pattern relative to its peers**: at
round 20, `client_002` (0.00015) is the *lowest* of the five clients
(honest values 0.019–0.037); by round 35 onward, `client_002` is
consistently the *highest* of the five (honest values already negative).
A fixed absolute threshold cannot exploit this crossing pattern, and in
any case `client_002`'s values never approach the (very loose,
early-training-derived) −0.0687 threshold at any round. **The peer-cosine
detector's specific failure mode from `EXP_014` — a threshold too loose
relative to the late-training feature scale — persists here for a
different underlying reason.**

---

## 7. `self_prev_round_cosine`: An Unplanned, Very Strong Signal

**Pre-attack (rounds 1–19), `client_002`:** starts at 0.242 (round 1),
declines smoothly to 0.267 (round 19) — closely tracking the honest
clients' own smoothly-declining self-consistency over the same window
(e.g. `client_000`: 0.380 → 0.222; all four honest clients show the same
qualitative shape throughout the entire 50-round run).

**Post-attack (rounds 20–49), `client_002`:** collapses **immediately and
completely** to ≈0.0000 at round 20 (`−0.000249`) and **stays there for
every one of the remaining 29 rounds** (range: `−0.00045` to `+0.00070`
— compare to its own round-19 value of `0.267`, a **same-round drop of
essentially 100%**).

**Every honest client's `self_prev_round_cosine` stays solidly positive
throughout rounds 20–49** — smoothly declining like every other
trajectory-driven feature in this project, but never approaching zero
(minimum observed across all four honest clients, all 30 post-attack
rounds: **≈0.122**, `client_000`, round 48–49). `client_002`'s *largest*
magnitude value during the entire attack window (`≈0.0007`) is roughly
**170x smaller** than the smallest honest value ever observed. **There is
zero overlap between the two distributions.**

**This is a distinctive temporal discontinuity, exactly at round 20, that
does not exist in `EXP_014`.** The mechanism is a direct consequence of
this specific attack's construction: each active round draws an
*independent* random rotation (seeded fresh per `round_id`, per
`_derive_directional_attack_seed`), so `client_002`'s submitted update
direction has essentially zero correlation not only with its own honest
counterfactual (§4) but also with its own immediately preceding round's
submitted direction — a property no honest client's smoothly-evolving
optimizer trajectory ever exhibits.

---

## 8. Rounds 17–25, All Four Diagnostics, `client_002` vs. Honest Clients

| Round | Client | `update_l2_norm` | `peer_mean_cosine_similarity` | `self_prev_round_cosine` | `realized_cosine_to_honest` |
|---|---|---:|---:|---:|---:|
| 17 | client_002 | 5.884 | 0.058 | 0.280 | — |
| 17 | honest range | 5.514–6.355 | 0.055–0.063 | 0.268–0.287 | — |
| 18 | client_002 | 5.928 | 0.042 | 0.270 | — |
| 18 | honest range | 5.507–6.364 | 0.047–0.054 | 0.228–0.278 | — |
| 19 | client_002 | 5.915 | 0.042 | 0.267 | — |
| 19 | honest range | 5.500–6.415 | 0.037–0.046 | 0.222–0.277 | — |
| **20** (attack active) | client_002 | 5.996 | 0.00015 | **−0.00025** | **4.2e-10** |
| 20 | honest range | 5.453–6.346 | 0.022–0.037 | 0.218–0.274 | — |
| 21 | client_002 | 6.206 | −0.00026 | **0.00010** | 3.5e-10 |
| 21 | honest range | 5.452–6.331 | 0.009–0.022 | 0.209–0.266 | — |
| 22 | client_002 | 6.226 | −0.00014 | **−0.00045** | 4.2e-10 |
| 22 | honest range | 5.422–6.360 | 0.008–0.018 | 0.198–0.264 | — |
| 23 | client_002 | 6.227 | −0.00031 | **−0.00028** | −3.8e-10 |
| 23 | honest range | 5.456–6.334 | 0.004–0.013 | 0.200–0.259 | — |
| 24 | client_002 | 6.278 | 0.00043 | **0.00007** | 6.4e-10 |
| 24 | honest range | 5.472–6.370 | 0.002–0.013 | 0.191–0.263 | — |
| 25 | client_002 | 6.291 | 0.00013 | **−0.00004** | −1.1e-10 |
| 25 | honest range | 5.484–6.437 | 0.005–0.014 | 0.187–0.253 | — |

`update_l2_norm` and `peer_mean_cosine_similarity` both stay *inside* or
only modestly outside the honest range through this window (consistent
with §§5–6's "not immediately or trivially separable" finding);
`self_prev_round_cosine` shows a stark, unambiguous break at round 20
that persists identically through round 25 and beyond (§7).

---

## 9. Is There Any One Simple Scalar Diagnostic That Perfectly Identifies `client_002`?

**Verified from data, not assumed:**

- **`update_l2_norm` (raw or z-scored): no**, not with any single,
  fixed, immediately-usable threshold — separation only becomes
  unambiguous at round 49 (§5), a 29-round-delayed, single-round signal.
- **`peer_mean_cosine_similarity`: no.** `client_002`'s rank among the
  five clients *changes sign* partway through the post-attack window
  (lowest of five at round 20, mid-pack by round 30, highest of five by
  round 35 onward, §6) — no single static threshold or fixed rank rule
  applied uniformly across rounds 20–49 separates it cleanly at every
  round.
- **`self_prev_round_cosine`: yes.** Every one of `client_002`'s 30
  active-round values (`−0.00045` to `+0.00070`) is separated from every
  one of the 120 honest client-round values across the same window
  (minimum honest value ≈0.122) by roughly two orders of magnitude, with
  **zero overlap** — a single fixed threshold anywhere in, e.g.,
  `(0.01, 0.10)` would flag every one of `client_002`'s active rounds
  starting at round 20 and no honest client-round at any point.

**This diagnostic was not designed or evaluated as a formal detector by
either this plan or `EXP_014`'s** — it is logged infrastructure only
(`research_plans/EXP_015_Norm_Preserving_Directional_Attack_Plan.md`
§15), with no pre-registered threshold rule. Relative to the two
**pre-registered, already-built** naive baselines (peer-cosine 3-sigma,
norm z-score), attribution remains non-trivial (§§5–6). This finding is
reported as a data observation for future consideration, not a proposal
for a new detector or a new attack (see §13).

---

## 10. Paired Global-Model Performance Comparison

| Metric | EXP_007 (honest) | EXP_015 (directional attack) | Δ (015 − 007) |
|---|---:|---:|---:|
| Final test loss | 1.3758617393493653 | 1.6682338859558106 | **+0.2924 (worse)** |
| Final test top-1 | 59.68% | 54.10% | **−5.58pp** |
| Final test top-5 | 88.65% | 83.88% | **−4.77pp** |
| Macro precision | 0.6509177318450269 | 0.6259386555918216 | −0.0250 |
| Macro recall | 59.65% | 54.10% | **−5.55pp** |
| Macro F1 | 0.5875828060131298 | 0.5076327053811284 | −0.0800 |
| Balanced accuracy | 59.65% | 54.10% | **−5.55pp** |
| Best validation top-1 | 59.22% (round 49) | 53.38% (round 49) | **−5.84pp** |

**Every metric moved substantially in the "worse" direction** — a
markedly different outcome from `EXP_014` (Gaussian attack, all
differences ≤0.9pp,
`analysis/EXP_014_Gaussian_Attack_Traceability_Analysis.md` §5). The
magnitude here (≈5–6pp) is several times larger than both the −1.52pp
pre-attack noise already observed between `EXP_015` and `EXP_007` (§1)
and this project's previously-established 2.85pp cross-seed noise floor
(`analysis/EXP_009_Three_Seed_Baseline_Analysis.md` §3) — though, per the
explicit instruction for this analysis, **this is one paired seed and no
significance claim is made**; a second/third seed would be needed to
treat this magnitude as a confirmed, general effect size for this attack
family.

---

## 11. Attack Boundary: Rounds 15–25 Trajectory

| Round | EXP_007 val. loss | EXP_015 val. loss | EXP_007 top-1 | EXP_015 top-1 |
|---|---:|---:|---:|---:|
| 15 | 2.1936 | 2.1381 | 39.42% | 40.80% |
| 16 | 2.0963 | 2.0959 | 41.28% | 42.76% |
| 17 | 2.0325 | 2.1003 | 42.86% | 42.16% |
| 18 | 2.0120 | 1.9697 | 43.84% | 44.62% |
| 19 | 1.9075 | 1.9879 | 46.00% | 44.48% |
| **20** | **1.8824** | **2.0950** | **46.58%** | **42.50%** |
| 21 | 1.8948 | 2.0985 | 46.60% | 42.98% |
| 22 | 1.8398 | 2.1169 | 47.74% | 43.10% |
| 23 | 1.7968 | 2.1289 | 48.86% | 43.68% |
| 24 | 1.7910 | 2.0303 | 49.20% | 44.62% |
| 25 | 1.7533 | 2.0374 | 49.60% | 44.84% |

**A visible, immediate boundary effect.** At round 19→20, `EXP_007`
continues its ordinary improving trend (loss −0.0252, top-1 +0.58pp);
`EXP_015` reverses sharply in the *same* transition (loss **+0.1071**,
top-1 **−1.98pp**) — the single largest one-round degradation either
curve shows anywhere in this 11-round window, landing exactly on the
attack's first active round. The gap between the two runs widens rapidly
over the next several rounds (round 19 gap: −1.52pp; round 25 gap:
−4.76pp) before growing more slowly for the rest of the run (round 25 →
round 49 gap: −4.76pp → −5.84pp), consistent with a real, sustained
injury to the global model's convergence rather than a single-round blip
that recovers.

---

## 12. Three Separate Conclusions

**A. Attack correctness — confirmed.** Norm preserved to ~1e-10 (§3),
realized direction matches the configured target to ~1e-10 (§4), only
`client_002` transformed, only on active rounds (§2), no round or client
failures anywhere in the 50-round run.

**B. Global-model damage — real and substantial, unlike `EXP_014`.** An
immediate, boundary-localized degradation (§11) and a sustained ≈5–6pp
final-accuracy gap (§10) — both far larger than this attack's paired
Gaussian predecessor produced. Mechanistically plausible: replacing
`client_002`'s genuine gradient contribution with an independently-drawn
random direction of the same magnitude, sustained for 30 consecutive
rounds at ~20% aggregation weight, removes real learning signal from
roughly a fifth of the federation every active round — a materially
different mechanism from adding zero-mean noise on top of an otherwise
intact gradient.

**C. Attribution / onset-localization difficulty — mixed, and this is
the central, nuanced finding.** Against the two **pre-registered** naive
baselines, attribution is genuinely harder than `EXP_014`: peer-cosine
never flags anyone (§6, same failure mode as before, for a related but
distinct reason), and the norm z-score detector's advantage is
essentially eliminated (§5: one correct flag, 29 rounds late, instead of
immediate, every-round detection). Onset localization via either
pre-registered baseline is therefore **not achievable within this run**
(peer-cosine: never; norm z-score: only at the final round, useless for
localizing an onset at round 20). However, an **unplanned diagnostic**
already being logged (`self_prev_round_cosine`) shows a clean, immediate,
zero-overlap separation exactly at round 20 (§§7–9) — not because it was
designed to catch this attack, but as a direct consequence of this
specific attack's per-round-independent-redraw construction. Attribution
via the *pre-registered* toolset remains non-trivial; attribution via the
*full set of already-logged* diagnostics is not.

---

## 13. Does `EXP_015` Satisfy the Intended Benchmark Criteria?

Evaluated against the five criteria named in the task, using only the
pre-registered naive baselines (consistent with §12-C's scoping):

- **Norm preserved:** yes (§3, ~1e-10 precision).
- **Direction materially altered:** yes (§4, realized cosine ≈0.0,
  matching the configured target exactly).
- **Attack causes meaningful behavioral consequence:** yes (§§10–11,
  ≈5–6pp final-accuracy degradation with an immediate boundary effect —
  a materially stronger, more visible consequence than `EXP_014`
  produced).
- **Norm-only shortcut removed:** yes (§5 — detection, if it occurs at
  all, is delayed from immediate to the very last round of the run).
- **Attribution remains non-trivial:** yes, with respect to the two
  pre-registered naive baselines this benchmark series has actually
  built and evaluated (§§6, 12-C).

**All five criteria are satisfied.** The `self_prev_round_cosine`
observation (§§7–9) is reported as an important data point for future
benchmark evolution — a specific, concrete instance of *how* an even
harder benchmark could eventually be constructed (e.g. an attack that
also persists or correlates its direction across consecutive rounds) —
but it does not constitute `EXP_015` failing the stated criteria, since
those criteria are scoped to the baselines this project has actually
pre-registered and built, not to every diagnostic that happens to already
be logged.

---

## 14. Is the Evidence Gate for Beginning FedTrace Design Met?

Per `research_plans/PHASE5_Malicious_Client_Traceability_Plan.md` §21 /
`research_plans/PHASE4_RESEARCH_STRATEGY.md` §9, the gate requires (1) a
working attack simulation with measurable global-model impact and (2) a
naive baseline evaluated against it and found to have a specific,
documented weakness. Both conditions are met, and are now supported by
**two qualitatively different attack families evaluated against the same
two naive baselines**: `EXP_014` (magnitude-based) showed the peer-cosine
detector fails due to a shared secular-trend confound; `EXP_015`
(direction-based, norm-preserving) shows the norm z-score detector fails
by design against a magnitude-preserving attack, and the peer-cosine
detector again fails, for a related but distinct reason (§6). **The gate
appears met**, with a stronger evidentiary basis than after `EXP_014`
alone. Caveat, as required: both `EXP_014` and `EXP_015` are single-seed
(42) pilots for their respective attack families — this conclusion rests
on qualitative weakness-documentation, not on any claim of statistical
significance, and neither attack family has yet been replicated at a
second or third seed.

---

## Files Created

- `analysis/EXP_015_Directional_Attack_Traceability_Analysis.md`

## Files Modified

- None.

## Commands Executed

- None. Only Read and Write tool calls were used.
