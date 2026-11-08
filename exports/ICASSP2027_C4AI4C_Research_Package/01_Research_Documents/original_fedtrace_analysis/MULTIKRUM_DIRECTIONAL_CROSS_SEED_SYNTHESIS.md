# Multi-Krum Directional Attack Cross-Seed Synthesis (EXP_024 + EXP_025)

**Status:** Analysis/synthesis document only. No code, test, config, README, FedTrace, attack, or Research Database file was created or modified in producing it. No experiment was run. Multi-Krum's `f=1`/`m=4`, tie-breaking, weighting, buffer handling, and the pre-registered `P_krum=3` rejection proxy were not modified or tuned. FedTrace v3 remains frozen. Multi-Krum's exclusion pattern is referred to throughout as the **"Multi-Krum rejection proxy"**, never as an attribution detector.

---

## Part 10 — EXP_024 + EXP_025 Cross-Seed Directional Synthesis

| | Seed 123 | Seed 2026 |
|---|---|---|
| Clean Multi-Krum | EXP_020 | EXP_021 |
| Attacked Multi-Krum | EXP_024 | EXP_025 |
| Attacked FedAvg | EXP_018 | EXP_019 |
| Persistently rejected honest client | `client_004` (largest partition, 11,785) | `client_001` (largest partition, 12,318) |
| Attacker post-attack rejection count | **0/30** | **0/30** |
| Attacker post-attack selection count | **30/30** | **30/30** |
| Attacker rank behavior | 2 (R20–21) → 3 (R22–49) | 3, constant, zero variation |
| Regime switch? | **No** | **No** |
| Full-run proxy predicted client | `client_004` (wrong) | `client_001` (wrong) |
| Proxy attribution correct? | **No** | **No** |
| Malicious client ever confirmed by proxy? | **Never** | **Never** |
| Attacked Multi-Krum Top-1 | 44.36% | 44.50% |
| Attacked FedAvg Top-1 | 53.10% | 54.56% |
| Attack effect under Multi-Krum (Top-1) | −5.55 pp | −6.79 pp |
| Aggregation effect under attack (Top-1) | −8.74 pp | −10.06 pp |

### Replication Classification

**A. Attacker evades Multi-Krum.** 0/30 rejections at both seeds, identically. **Fully replicated.**

**B. An honest heterogeneity client remains persistently rejected.** `client_004` (seed 123) and `client_001` (seed 2026) — different identities, but both are, in every case, that seed's single largest Dirichlet partition (consistent with the fairness association already established across the entire Multi-Krum program). **Fully replicated** (as a structural role, not a fixed identity — exactly as already established for the clean and Gaussian branches).

**C. No rejection switch.** Confirmed independently at both seeds via direct round-by-round inspection (Part 5 of each run's own analysis). **Fully replicated.**

**D. Full-run proxy never identifies the attacker.** At both seeds, `client_002` is never rejected even once, so it is never confirmed by the `P_krum=3` rule at any point, under any interpretation of "confirmation." **Fully replicated** — and notably a *cleaner*, more absolute replication than the Gaussian family's own analog (`analysis/MULTIKRUM_GAUSSIAN_CROSS_SEED_SYNTHESIS.md` Part 8, finding D), where the attacker was at least eventually confirmed after the honest false positive; here there is no such eventual correction at either seed.

**E. Directional attack materially degrades Multi-Krum.** −5.55 pp (seed 123) and −6.79 pp (seed 2026) Top-1 relative to the clean Multi-Krum control — both substantial, both negative, both immediate-onset-then-persistent-plateau in shape (Part 7 of each run's analysis). **Fully replicated**, including the qualitative trajectory shape (unlike the Gaussian family's own attacked-vs-clean comparison, where the trajectory *shape* only partially replicated — `analysis/MULTIKRUM_GAUSSIAN_CROSS_SEED_SYNTHESIS.md` finding E).

**F. Attacked Multi-Krum substantially underperforms attacked FedAvg.** −8.74 pp (seed 123) and −10.06 pp (seed 2026) Top-1. **Fully replicated.**

**All six seed-123 directional findings replicate fully at seed 2026** — a cleaner, more uniform two-seed confirmation than the Gaussian family produced (where one of six analogous findings was only partially replicated due to a trajectory-shape difference). No universal claim is made beyond these two tested seeds, this one attack strength (`target_cosine=0.0`), and this one partition/client-count configuration.

## Part 11 — Gaussian vs. Directional, Across Both Seeds

| | Gaussian, seed 123 (`EXP_022`) | Gaussian, seed 2026 (`EXP_023`) | Directional, seed 123 (`EXP_024`) | Directional, seed 2026 (`EXP_025`) |
|---|---|---|---|---|
| Attacker post-attack rejection rate | **100% (30/30)** | **100% (30/30)** | **0% (0/30)** | **0% (0/30)** |
| Attacker rank post-attack | 5, every round | 5, every round | 2→3, then constant | 3, constant |
| Regime switch at round 20? | **Yes, instant** | **Yes, instant** | **No** | **No** |
| Full-run proxy attribution | Wrong (honest FP), attacker eventually confirmed (round 22) | Wrong (honest FP), attacker eventually confirmed (round 22) | Wrong (honest FP), attacker **never** confirmed | Wrong (honest FP), attacker **never** confirmed |
| Attacked vs. clean Multi-Krum (Top-1) | **+4.52 pp** | **+3.49 pp** | **−5.55 pp** | **−6.79 pp** |
| Attacked Multi-Krum vs. attacked FedAvg (Top-1) | −2.07 pp | −3.30 pp | **−8.74 pp** | **−10.06 pp** |

**Central question: does attack geometry reproducibly determine whether Multi-Krum geometrically excludes the attacker?**

**Yes — across all four tested runs (two seeds × two attack families), the answer is perfectly consistent and tied to attack geometry, not to seed.** Every Gaussian run shows 100% post-attack rejection with an instantaneous regime switch; every directional run shows 0% post-attack rejection with no switch at all. This is now a 2×2, fully-crossed, same-seed-pairs experimental design — the cleanest evidentiary basis this program has produced for any single claim — and it holds without exception across all four cells.

**This claim is scoped exactly to what was tested and no further:**
- Two seeds (123, 2026), not a larger seed sample.
- Two specific attack parameterizations — `rho=1.0` for Gaussian, `target_cosine=0.0` for directional — not a sweep across attack strengths. An intermediate directional rotation (a `target_cosine` between 0.0 and 1.0) or a weaker Gaussian `rho` were not tested and might plausibly land anywhere between the two observed extremes.
- One partition scheme (Dirichlet `α=0.1`), one client count (5), one model/dataset (ResNet-18/CIFAR-100).
- **Not claimed:** that raw-Euclidean-distance robust aggregation is blind to all direction-only attacks in general, or that it reliably catches all magnitude-revealing ones — only that, within this tested space, magnitude-revealing and norm-preserving attacks produce opposite, reproducible outcomes for this specific mechanism.

## Part 12 — FedTrace vs. Multi-Krum, Directional Family, Both Seeds

Using the already-frozen, already-validated FedTrace v3 results for the paired FedAvg runs (not re-run on either Multi-Krum trace in this task):

| | FedTrace v3, `EXP_018` (seed 123) | FedTrace v3, `EXP_019` (seed 2026) | Multi-Krum proxy, `EXP_024` (seed 123) | Multi-Krum proxy, `EXP_025` (seed 2026) |
|---|---|---|---|---|
| Attacker attribution | `client_002` — correct | `client_002` — correct | `client_004` — **wrong** | `client_001` — **wrong** |
| Onset localization | 20 (error 0) | 20 (error 0) | 2 (the honest FP's own onset) | 2 (the honest FP's own onset) |
| Confirmation delay | 2 (round 22) | 2 (round 22) | Not applicable — attacker never confirmed | Not applicable — attacker never confirmed |
| Honest persistent FP | **0** | **0** | **1** | **1** |
| Dominant signal | `z_t_res`, 30/30 anomalous rounds | `z_t_res`, 30/30 anomalous rounds | Raw per-round Euclidean distance, no history | Raw per-round Euclidean distance, no history |
| Requires known `f`? | No | No | Yes (`f=1`) | Yes (`f=1`) |

**Does the two-seed directional evidence support the claim that robust aggregation and forensic traceability are distinct objectives? Yes — more strongly than the Gaussian two-seed evidence alone.** FedTrace v3 succeeds identically, correctly, and with zero honest false positives at both seeds, via the same dominant signal (`z_t_res`, 30/30 both times) — demonstrating that its design's cross-attack coverage (magnitude signal for the Gaussian family, temporal-residual signal for the directional family) is not a seed-specific accident. The Multi-Krum rejection proxy, at both seeds, provides not merely an imprecise or delayed signal about the true attacker but **zero signal of any kind** — the same honest largest-partition client is misattributed at both seeds, and `client_002` is never confirmed even once, at either seed. **This is the most unambiguous evidentiary pairing this program has produced for the claim that "which updates should count this round" (robust aggregation) and "which client has a persistently, newly changed baseline, and since when" (forensic attribution) are different problems** — the same underlying distance-based mechanism that (per the Gaussian evidence) can at least geometrically separate one attack type entirely fails, at both seeds, to do so for a different, equally real attack type, while FedTrace's purpose-built, per-client-calibrated design succeeds at both, unchanged, for both.

---

## Final Report Addendum (This Document)

- **File created:** `analysis/MULTIKRUM_DIRECTIONAL_CROSS_SEED_SYNTHESIS.md`. See `analysis/EXP_025_MULTIKRUM_DIRECTIONAL_SEED2026_ANALYSIS.md`'s Final Report and `analysis/MULTIKRUM_EXTERNAL_BASELINE_FINAL_SYNTHESIS.md` for the complete report covering all three files created in this task.
- No code, config, test, README, FedTrace, attack, or Research Database file was changed; no experiment was run; no command was executed.
