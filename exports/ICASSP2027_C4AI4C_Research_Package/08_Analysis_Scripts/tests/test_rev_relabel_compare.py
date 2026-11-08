"""Relabel-only comparison: the amended rho rules (fixed before outcomes) and effect definitions."""

from __future__ import annotations

import pytest

from src.rev import relabel_compare as rc


@pytest.mark.parametrize("relabel, badnets, cls, flag", [
    (0.11, 0.12, "HIGH", None),
    (0.06, 0.12, "MID", None),
    (0.02, 0.12, "LOW", None),
    (0.01, 0.12, "LOW", "NO_DETECTABLE_RELABEL_EFFECT"),
    (-0.05, 0.12, "LOW", "OPPOSITE_SIGN"),
    (0.09, 0.12, "HIGH", None),          # 0.75 exactly
    (0.03, 0.12, "LOW", None),           # 0.25 exactly
])
def test_classify_rho(relabel, badnets, cls, flag) -> None:
    out = rc.classify_rho(relabel, badnets)
    assert out["class"] == cls and out.get("flag") == flag
    assert out["rho"] == pytest.approx(relabel / badnets)


@pytest.mark.parametrize("relabel, badnets, reason", [
    (None, 0.12, "missing"), (0.1, None, "missing"), (0.1, 0.01, "near-zero"), (0.1, -0.019, "near-zero"),
])
def test_classify_rho_undefined(relabel, badnets, reason) -> None:
    out = rc.classify_rho(relabel, badnets)
    assert out["class"] == "UNDEFINED" and out["rho"] is None and reason in out["reason"]


def test_overall_preserves_the_original_rule() -> None:
    assert rc.overall(["HIGH", "HIGH"]) == "relabelling sufficient"
    assert rc.overall(["LOW", "LOW"]) == "trigger-specific"
    assert rc.overall(["MID", "MID"]) == "mixed"
    assert rc.overall(["HIGH", "MID"]) == "inconsistent across seeds"
    assert rc.overall(["HIGH", "UNDEFINED"]) == "UNDEFINED"
    assert set(rc.CAUTIOUS) == {"relabelling sufficient", "trigger-specific", "mixed", "inconsistent across seeds", "UNDEFINED"}
    assert (rc.LOW, rc.HIGH, rc.EPSILON) == (0.25, 0.75, 0.02)


def _rows(attacker_t, attacker_norm):
    rows = []
    for r in range(40):
        for c in ("client_000", "client_001", "client_002", "client_003", "client_004"):
            t = None if r == 0 else 0.25 + 0.01 * (int(c[-1]) % 3)
            n = 10.0 + (r % 3) * 0.1
            if c == "client_002":
                t = None if r == 0 else attacker_t(r)
                n = attacker_norm(r)
            rows.append({"round_id": r, "client_id": c, "self_prev_round_cosine": t, "update_l2_norm": n})
    return rows


def test_effect_definitions() -> None:
    clean = _rows(lambda r: 0.25 + 0.001 * (r % 4), lambda r: 10.0 + 0.1 * (r % 3))
    arm = _rows(lambda r: 0.25 + 0.001 * (r % 4) + (0.1 if r >= 20 else 0.0), lambda r: (10.0 + 0.1 * (r % 3)) * (0.8 if r >= 20 else 1.0))
    assert rc.self_cos_change(clean, arm) == pytest.approx(0.1)
    assert rc.norm_effect(clean, arm) == pytest.approx(0.2)
    assert rc.self_cos_change(clean, clean) == pytest.approx(0.0)
    missing = [r for r in arm if not (r["client_id"] == "client_002" and r["round_id"] == 25)]
    assert rc.self_cos_change(clean, missing) is None  # incomplete window -> undefined, never silently averaged
