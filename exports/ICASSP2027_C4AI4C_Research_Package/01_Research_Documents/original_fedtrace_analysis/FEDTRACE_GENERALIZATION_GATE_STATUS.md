# FedTrace v3 Generalization Gate Status

**Status:** Gate-status/tracking document only. No code, config, test, README, FedTrace, attack, or Research Database file was created or modified in producing it. No experiment was run.

---

## 1. Origin

`analysis/ICML_ICLR_EVIDENCE_GAP_AUDIT.md` (Part 6, Part 17) identified exactly **one essential remaining evidence gap** before FedTrace v3's generalization claims could be finalized for a top-tier submission: every attack-bearing run in this project through `EXP_025` (ten runs: `EXP_014`–`EXP_019`, `EXP_022`–`EXP_025`) used the identical malicious client (`client_002`) at the identical attack onset (round 20). The audit recommended closing this as two independent, single-variable-change sub-questions — attacker identity and attack onset — rather than one compounded change.

## 2. Gate Ledger

| Axis | Status | Evidence | Verdict |
|---|---|---|---|
| Attacker identity | **Narrowed** | `EXP_026` (`analysis/EXP_026_ATTACKER_IDENTITY_GENERALIZATION_ANALYSIS.md`): `client_004` (largest partition, seed 123), directional attack, round-20 onset — frozen FedTrace v3 matched every pre-registered criterion from `EXP_018` exactly (correct attribution, onset error 0, delay 2, honest FP 0, `z_t_res` dominant 30/30) | **PASS** (one new identity; not proof of arbitrary-identity invariance) |
| Attack onset | **Untested** | No run in this project has ever used `attack_start_round ≠ 20` | **OPEN** |

## 3. Current Overall Verdict

**A — attacker-identity generalization passes; one onset-generalization run is the recommended final essential experiment.** This mirrors `EXP_026`'s own §9 Next-Gate Decision exactly; this document exists only to make that state machine explicit and trackable across tasks, not to re-derive it.

## 4. Recommended Next Step (Not Performed Here)

One run, anchored against `EXP_018` (not `EXP_026`, to keep the identity and onset axes independently varied rather than compounded):

- `malicious_client_id: client_002` (held at `EXP_018`'s value)
- `attack_type: norm_preserving_directional`
- `directional_target_cosine: 0.0`
- `seed: 123`
- `dirichlet_alpha: 0.1`, `aggregation: fedavg`, all other fields unchanged from `EXP_018`
- `attack_start_round`: changed from `20` to a different round — **round 30 suggested** (substantially distinct from round 20; leaves a 20-round post-onset observation window, comparable in kind to the window `EXP_018`'s own onset left) — **not binding, a recommendation only**

**Next experiment ID, if authorized: `EXP_027`** (the next unused sequential ID after `EXP_026`, not yet verified against the repository at the time this document was written — must be re-verified, not assumed, when that task is undertaken).

**Not recommended, per this project's own standing experiment-budget discipline** (`analysis/FEDTRACE_STAGE_B_FINAL_SYNTHESIS.md` §22, `analysis/ICML_ICLR_EVIDENCE_GAP_AUDIT.md` Part 6): a seed-2026 replication of `EXP_026`, a combined identity-and-onset change in a single run, or any further identity beyond `client_004` — none of these were authorized by, or are implied as necessary by, the current gate state.

## 5. Update Discipline

This document should be updated (not re-created under a new name) as each subsequent generalization run completes, so the gate ledger in §2 remains the single place this project's identity/onset generalization status can be checked without re-reading every individual `EXP_0XX` analysis document.

---

## Final Report Addendum (This Document)

- **File created:** `analysis/FEDTRACE_GENERALIZATION_GATE_STATUS.md`. See `analysis/EXP_026_ATTACKER_IDENTITY_GENERALIZATION_ANALYSIS.md`'s own Final Report for the complete report covering both files created in this task.
- No code, config, test, README, FedTrace, attack, or Research Database file was changed; no experiment was run.
