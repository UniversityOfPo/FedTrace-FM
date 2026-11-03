# 11 — Seed Registry

Append-only. A seed's role never changes once recorded. No seed listed here as CONFIRMATION or VALIDATION may be used
for method design.

| Seed | Role | First use / sealing | Notes |
|---|---|---|---|
| 42 | NDSS development (archived) | NDSS EXP_001 onward | never validation |
| 123 | DEVELOPMENT (archived NDSS + reviewer-response pilots P1, relabel-only) | NDSS EXP_008; P1 2026-10-24 | permanently excluded from confirmation and validation |
| 2026 | DEVELOPMENT (archived NDSS + G1, relabel-only) | NDSS EXP_009; G1 2026-10-25 | permanently excluded from confirmation and validation |
| 978310317 | DEVELOPMENT-CONFIRMATION C1 | sealed 2026-10-25T15:22:10Z | not run; reserved for confirming a candidate defined on seeds 123/2026 (protocol 09 §6) |
| 478485063 | DEVELOPMENT-CONFIRMATION C2 | sealed 2026-10-25T15:22:10Z | not run; same purpose |
| 1082418221 | CONFIRMATION SPARE | sealed 2026-10-25T15:22:10Z | used only if C1 or C2 must be replaced, with a dated record of the reason |

## Derivation rule (confirmation seeds)

`d = SHA-256(UTF-8 "FedTrace-rev-development-confirmation|2026-10-25")` =
`3a4fd4ad1c851a47c084642d252c54f479c2dea97f962e1936396cd8af5bae3c`; split `d` into eight big-endian 32-bit words,
take each word modulo 2^31, and walk them in order, skipping any value already in this registry (42, 123, 2026) and
any of the first five candidates of the previously proposed (still unsealed) validation rule
`SHA-256("FedTrace-rev-validation-2026")` (860095100, 1762654165, 1087939910, 689923765, 1721922713). The first two
remaining values are C1, C2; the third is the spare. Selected before any relabel-only outcome existed (the relabel-only
runs had not started at the sealing time).

## Registry check performed at sealing

Seeds found in `configs/**/*.yaml` and in every `summary.json` under `results/`, `results_ndss/`, `results_rev/`:
{42, 123, 2026} only; the FedTrace-FM registry contains no runs.

Validation seeds are not yet sealed; when they are, they must exclude every seed above.

## 2026-10-26 — Workshop confirmation seeds (Prompt 33; P-50 option S1) — derived, NOT RUN

| Seed | Role | First use / sealing | Notes |
|---|---|---|---|
| 1178015705 | WORKSHOP-CONFIRMATION W1 | derived and recorded 2026-10-26; not run | protocol 26; run only after PI approval (P-56) |
| 1463065807 | WORKSHOP-CONFIRMATION W2 | derived and recorded 2026-10-26; not run | same |
| 1966286922 | WORKSHOP-CONFIRMATION SPARE W3 | derived and recorded 2026-10-26; not run | replaces W1 or W2 only for a dated, recorded reason (protocol 26 §7) |

Derivation: `d = SHA-256(UTF-8 "FedTrace-C4AI4C-workshop-confirmation|2026-10-26")` =
`463717d9d7349ccff533284ad757eed899b4202886d6e4cb3cc2f19b2937d9dc`; eight big-endian 32-bit words, each modulo 2^31,
walked in order, skipping every seed above (42, 123, 2026, 978310317, 478485063, 1082418221) and the five candidates
of the unsealed validation rule. The sealed C1 / C2 / spare are untouched. Configs found to use seeds: {42, 123, 2026}.

## 2026-11-03 — Workshop extension seeds (protocol 37; PI approval 2026-11-03)

| Seed | Role | First use / sealing | Notes |
|---|---|---|---|
| 1966286922 | WORKSHOP-EXTENSION W3 (formerly spare W3) | role changed 2026-11-03 by PI approval; run under protocol 37 | not a replacement of W1 / W2 |
| 1465380568 | WORKSHOP-EXTENSION W4 | derived 2026-11-03 (rule above, position 4); run under protocol 37 | — |
| 431235112 | WORKSHOP-EXTENSION W5 | derived 2026-11-03 (rule above, position 5); run under protocol 37 | client_002 holds 5 target-class samples (disclosed) |

Positions 4 and 5 of the same derivation; neither collides with any seed above or with the five skipped validation
candidates. Sealed C1 / C2 / spare remain untouched.
