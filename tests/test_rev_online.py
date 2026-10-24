"""Causal wrapper around frozen FedTrace v3: prefix invariance, event semantics, documented look-ahead."""

from __future__ import annotations

import random

import pytest

from src.analysis.fedtrace_detector import (
    _select_predicted_client,
    apply_persistence_rule_v3,
    compute_all_scores_v3,
)
from src.rev.online import (
    OnlineConfig,
    causal_scores_v3,
    online_detect_v3,
    online_vs_offline,
    prefix_invariance_violations,
)

CLIENTS = ("client_000", "client_001", "client_002", "client_003", "client_004")


def make_rows(rounds: int = 40, attacker: str | None = "client_002", onset: int = 25, jump: float = 8.0,
              overrides: dict | None = None, seed: int = 11) -> list[dict]:
    """Synthetic trace rows: honest norms ~ 10 +- 0.3, self-cosines ~ 0.5 +- 0.02; attacker norm jumps at onset."""
    rng = random.Random(seed)
    rows = []
    for r in range(rounds):
        for cid in CLIENTS:
            norm = 10.0 + rng.gauss(0, 0.3)
            cos = None if r == 0 else 0.5 + rng.gauss(0, 0.02)
            if attacker == cid and r >= onset:
                norm += jump
            if overrides and (r, cid) in overrides:
                norm = overrides[(r, cid)]
            rows.append({"round_id": r, "client_id": cid, "update_l2_norm": norm, "self_prev_round_cosine": cos,
                         "attack_configured": attacker is not None, "malicious_client_id": attacker,
                         "attack_start_round": onset if attacker else None})
    return rows


def test_offline_v3_leaks_future_information_before_t_ref_but_not_after() -> None:
    rows = make_rows()
    cfg = OnlineConfig(t_ref=20)
    violations = prefix_invariance_violations(rows, cfg)
    leaking_rounds = {v[0] for v in violations}
    assert leaking_rounds, "offline v3 scores of calibration rounds must depend on later rounds"
    assert max(leaking_rounds) < cfg.t_ref
    assert not prefix_invariance_violations(rows, cfg, rounds=range(cfg.t_ref, 40))


def test_causal_scores_equal_offline_scores_from_t_ref_on_and_are_absent_before() -> None:
    rows = make_rows()
    cfg = OnlineConfig()
    causal = {(s.round_id, s.client_id): s for s in causal_scores_v3(rows, cfg)}
    offline = {(s.round_id, s.client_id): s for s in compute_all_scores_v3(rows)}
    assert min(r for r, _ in causal) == cfg.t_ref
    for key, score in causal.items():
        assert score == offline[key]


def test_online_events_for_a_detected_attack() -> None:
    rows = make_rows(onset=25)
    det = online_detect_v3(rows, OnlineConfig(t_ref=20, persistence=3))
    attacker = det.per_client["client_002"]
    assert attacker.alarm_round == 27 and attacker.estimated_onset == 25 and attacker.actionable_round == 28
    assert det.attributed_client == "client_002" and det.attributed_alarm_round == 27
    assert all(det.per_client[c].alarm_round is None for c in CLIENTS if c != "client_002")
    pre = online_detect_v3(rows, OnlineConfig(monitoring_point="pre_aggregation"))
    assert pre.per_client["client_002"].actionable_round == 27
    assert det.retrospective_onset["client_002"] == 25  # offline estimate, reported separately


def test_alarm_depends_only_on_rounds_up_to_the_alarm() -> None:
    rows = make_rows(onset=25)
    full = online_detect_v3(rows)
    at_alarm = online_detect_v3([r for r in rows if r["round_id"] <= 27])
    before = online_detect_v3([r for r in rows if r["round_id"] <= 26])
    assert at_alarm.per_client["client_002"] == full.per_client["client_002"]
    assert before.per_client["client_002"].alarm_round is None


def test_streak_straddling_t_ref_starts_at_t_ref_online_but_earlier_offline() -> None:
    # One outlier inside the calibration window (round 19) followed by a sustained shift: offline v3 scores
    # round 19 against a calibration that includes round 19 itself and confirms a streak starting at 19.
    overrides = {(r, "client_001"): 30.0 for r in range(19, 40)}
    rows = make_rows(attacker=None, overrides=overrides)
    offline = apply_persistence_rule_v3(compute_all_scores_v3(rows), ("m", "t_res"))
    assert (offline["client_001"].onset_estimate, offline["client_001"].confirmation_round) == (19, 21)
    online = online_detect_v3(rows)
    events = online.per_client["client_001"]
    assert (events.estimated_onset, events.alarm_round) == (20, 22)
    assert any("client_001" in d for d in online_vs_offline(rows))


def test_self_inclusive_calibration_masks_a_short_burst_inside_the_calibration_window() -> None:
    # Property of frozen v3 (documented, not changed): with 20 calibration rounds, k equal outliers reach at most
    # z = sqrt((n-k)(n-1)/(k n)) -> 2.32 for k = 3 < tau = 3, so a 3-round calibration-period burst is never
    # confirmed offline, and the online wrapper never scores those rounds at all.
    overrides = {(r, "client_001"): 30.0 for r in range(5, 8)}
    rows = make_rows(attacker=None, overrides=overrides)
    offline = apply_persistence_rule_v3(compute_all_scores_v3(rows), ("m",))
    assert offline["client_001"].confirmation_round is None
    assert online_detect_v3(rows).per_client["client_001"].alarm_round is None


def test_tie_break_uses_only_scores_up_to_the_alarm_unlike_offline_v3() -> None:
    # Two clients alarm in the same round. Up to the alarm client_003 has the higher scores; after it,
    # client_001 has much higher ones. Offline v3's tie-break (mean over the whole streak) uses those later rounds.
    overrides = {}
    for r in range(25, 28):
        overrides[(r, "client_001")] = 14.0
        overrides[(r, "client_003")] = 16.0
    for r in range(28, 40):
        overrides[(r, "client_001")] = 60.0
        overrides[(r, "client_003")] = 15.0
    rows = make_rows(attacker=None, overrides=overrides)
    online = online_detect_v3(rows)
    assert online.per_client["client_001"].alarm_round == online.per_client["client_003"].alarm_round == 27
    assert online.attributed_client == "client_003"
    offline = apply_persistence_rule_v3(compute_all_scores_v3(rows), ("m", "t_res"))
    assert _select_predicted_client(offline) == "client_001"  # look-ahead in the offline tie-break


def test_online_config_validation() -> None:
    with pytest.raises(ValueError):
        OnlineConfig(t_ref=1)
    with pytest.raises(ValueError):
        OnlineConfig(monitoring_point="whenever")
