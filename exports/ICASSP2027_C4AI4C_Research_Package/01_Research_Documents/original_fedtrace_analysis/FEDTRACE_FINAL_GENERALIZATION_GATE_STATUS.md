# FedTrace v3 Final Generalization Gate Status

**Status:** Synthesis/gate-status document only. No code, config, test, README, FedTrace, attack, Research Database, or master-workbook file was created or modified in producing it. No experiment was run. This document supersedes `analysis/FEDTRACE_GENERALIZATION_GATE_STATUS.md` as the authoritative, complete gate-status record (both targeted generalization experiments are now complete); the earlier document is left unmodified, per this task's restriction against changing existing files.

---

## 1. Origin

`analysis/ICML_ICLR_EVIDENCE_GAP_AUDIT.md` (Part 6, Part 17) identified exactly **one essential remaining evidence gap** before FedTrace v3's generalization claims could be finalized: every attack-bearing run through `EXP_025` (ten runs) used the identical malicious client (`client_002`) at the identical attack onset (round 20). The audit recommended closing this with a small, targeted, two-axis experiment package — attacker identity and attack onset, each varied independently, one factor at a time — rather than a broad sweep.

**Both halves of that package are now complete:**

- `EXP_026` (`analysis/EXP_026_ATTACKER_IDENTITY_GENERALIZATION_ANALYSIS.md`): attacker identity changed (`client_002` → `client_004`, the largest-partition client at seed 123), onset held at round 20, anchored against `EXP_018`.
- `EXP_027` (`analysis/EXP_027_ONSET_GENERALIZATION_ANALYSIS.md`): attack onset changed (round 20 → round 30), attacker identity held at `client_002`, anchored against `EXP_018` (not `EXP_026`, keeping the two axes independent rather than compounded).

## 2. Gate Ledger

| Axis | Status | Evidence | Result |
|---|---|---|---|
| Attacker identity | **Tested (one new identity)** | `EXP_026`: `client_004`, directional attack, round-20 onset, seed 123 | **PASS** — attribution correct, onset error 0, delay 2, honest FP 0, `z_t_res` dominant 30/30 |
| Attack onset | **Tested (one new onset)** | `EXP_027`: `client_002`, directional attack, round-30 onset, seed 123 | **PASS** — attribution correct, onset error 0, delay 2, honest FP 0, `z_t_res` dominant 20/20 |

**Both axes individually pass, on one targeted test each.** No run has tested both axes changed simultaneously (deliberately, to keep each comparison a clean single-variable change against the shared `EXP_018` anchor), and no run has tested either axis beyond the one alternative value each was given.

## 3. What Is Now Supported

- Frozen FedTrace v3, applied with **zero modification, retuning, or reinterpretation** across `EXP_018`, `EXP_026`, and `EXP_027`, correctly attributed the malicious client and exactly localized attack onset (onset error 0, delay 2 — the theoretical minimum under `P=3`) in all three runs, with zero honest persistent false positives in all three.
- The detector's decision is **onset-relative, not tied to the specific round-20 value used in every run through `EXP_025`**: `EXP_027`'s confirmation round tracked its shifted true onset exactly (`30+2=32`, mirroring `EXP_018`'s `20+2=22`).
- The detector's decision is **not specific to `client_002`**: `EXP_026`'s result for `client_004` (the largest-partition client, a structurally distinct identity already independently flagged elsewhere in this project — `analysis/MULTIKRUM_DIRECTIONAL_CROSS_SEED_SYNTHESIS.md` Part 10) reproduced every pre-registered criterion.
- In both new runs, the temporal-residual signal (`z_t_res`) — not magnitude — was the sole channel driving the correct, timely detection (100% dominance in both), consistent with every prior directional-attack result in this project (`EXP_015`, `EXP_018`, `EXP_019`).
- The previously-identified round-0/round-1 structural calibration artifact (self-inclusive `T_ref=20` window boundary effect) reproduced identically and harmlessly in both new runs, further confirming it is a deterministic, well-understood property of this seed/partition's early-round data, not noise, and not something that interacts with either generalization axis.

## 4. What Is Still NOT Supported

**Stated conservatively, as instructed — one targeted test per axis is evidence of generalization within that one comparison, not proof of arbitrary-client or arbitrary-onset invariance:**

- **Not an arbitrary-identity guarantee.** Only two of five possible malicious-client identities have ever been tested (`client_002`, `client_004`) — both directional-attack instances, both seed 123. `client_000`, `client_001`, and `client_003` as the malicious client remain untested.
- **Not an arbitrary-onset guarantee.** Only two onset rounds have ever been tested (20, 30), both safely *after* the fixed `T_ref=20` magnitude-calibration window closes. **An onset earlier than `T_ref=20` (e.g., round 10 or 15) is a structurally different, harder case — the calibration window itself would contain some already-attacked rows — and has never been tested by any run in this project.** An onset very close to the end of the 50-round window (leaving little room for `P=3` confirmation) is likewise untested.
- **No joint identity-and-onset change has been tested** — `EXP_026` and `EXP_027` each varied exactly one axis relative to `EXP_018`; whether the two effects compose without interaction is untested (and, per the audit's own experiment-budget discipline, not treated as essential to test).
- **Both generalization runs are seed-123-only and directional-attack-only.** Neither axis has been tested against the Gaussian attack family, nor at seed 2026, nor at any other seed.
- **No change to dataset, architecture, client count, participation rate, or heterogeneity level accompanies either generalization run** — these remain exactly the pre-existing `analysis/ICML_ICLR_EVIDENCE_GAP_AUDIT.md` Part 5 gap-audit findings (F/G/I/J: NOT READY / PARTIALLY READY), unaffected by `EXP_026`/`EXP_027`.
- **No adaptive adversary, no multi-attacker scenario, no severity sweep** — unchanged from every prior limitations statement in this project.

## 5. Final Gate Decision

**A — the essential ICML/ICLR evidence gap identified by `analysis/ICML_ICLR_EVIDENCE_GAP_AUDIT.md` Part 6/Part 17 is sufficiently closed; proceed to final synthesis / paper preparation.**

**Justification:** the audit specified this gap narrowly and explicitly — "malicious-client identity / onset generalization," to be closed by a small, targeted, two-axis package, not a broad sweep (Part 6 of the audit explicitly weighed against over-recommending experiments and named this exact package as the one essential item). Both halves of that named package are now complete, and both passed every pre-registered criterion (attribution, onset error, delay, honest FP) with zero retuning of the frozen detector. No methodological issue was discovered in either run; the one incidental finding (the round-0 structural artifact) is fully explained, non-persistent, reproduced identically and harmlessly a third time, and does not bear on any of the pre-registered conclusions.

**This is a decision that the one *named, essential* gap is closed — it is explicitly not a claim that FedTrace v3's generalization has been broadly, exhaustively established.** §4's limitations list stands as-is and should be carried into the paper's own limitations section largely unchanged, with `EXP_026`/`EXP_027` added as the two new, narrowly-scoped positive data points they are.

**Not B:** no further targeted experiment is essential to close the specific gap the audit named. The candidate extensions in §4 (a third identity, a before-`T_ref` onset, a second seed for either axis, a joint identity+onset run, extending to the Gaussian family) are all real, legitimate, but **optional** extensions — consistent with this project's own repeatedly-applied experiment-budget discipline (`analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md` §22, `analysis/ICML_ICLR_EVIDENCE_GAP_AUDIT.md` Part 6's ESSENTIAL/HIGH-VALUE/NOT-WORTH-RUNNING triage), none of them individually rises to essential-before-submission the way the original identity/onset confound did (that confound was concentrated across *every one* of ten prior runs; no comparably total confound remains).

**Not C:** no methodological issue was discovered in frozen FedTrace v3's behavior in either `EXP_026` or `EXP_027`. The round-0 artifact is a documentation-completeness finding about one sentence in a prior analysis document (`EXP_018`'s own §10), not a defect in any pre-registered decision, and this task was explicitly instructed not to (and did not) change any detector logic because of it.

## 6. Is Any Further GPU Experiment Scientifically Necessary Before Final Synthesis?

**No.** The one essential gap the ICML/ICLR audit named is closed by `EXP_026` + `EXP_027`. Proceeding directly to final synthesis / paper preparation is scientifically defensible on the evidence now in hand, **provided** the paper's own claims and limitations sections state precisely what was tested (one new identity, one new onset, each independently, both at seed 123, both directional-attack-only, both with onset after `T_ref`) — exactly as §3/§4 above state it, not a broader claim.

The candidates in §4 remain available as **high-value-if-budget-allows** items for a stronger revision or a follow-up paper, in the same spirit as `analysis/ICML_ICLR_EVIDENCE_GAP_AUDIT.md` Part 6/7's own tiering — they are not re-elevated to essential by this document.

## 7. Instruction Compliance Note

Per this task's explicit instruction, no paper figures, LaTeX, polished tables, or manuscript text are produced in this document or in `analysis/EXP_027_ONSET_GENERALIZATION_ANALYSIS.md`. Reaching gate decision **A** here authorizes proceeding to a *future* final-synthesis task (which was not requested in this task and is not begun here) — it does not itself constitute that synthesis.

---

## Final Report Addendum (This Document)

- **File created:** `analysis/FEDTRACE_FINAL_GENERALIZATION_GATE_STATUS.md`. See `analysis/EXP_027_ONSET_GENERALIZATION_ANALYSIS.md`'s own Final Report for the complete report covering both files created in this task.
- **Combined evidence-gap status:** both axes of the ICML/ICLR audit's named essential gap (identity, onset) are now each supported by one targeted, passing test; final gate decision **A**.
- **Whether further GPU experiments are necessary:** no — see §6.
- No code, config, test, README, FedTrace, attack, Research Database, or master-workbook file was changed; no experiment was run.
