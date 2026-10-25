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
