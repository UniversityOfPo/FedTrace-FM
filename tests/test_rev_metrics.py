"""Metric semantics and statistical edge cases (runs are the unit; misses are never dropped)."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from src.rev.metrics import (
    Proportion,
    attack_success_counts,
    clopper_pearson,
    paired_impact,
    predict,
    run_outcome,
    summarize,
)
from src.rev.online import ClientEvents, OnlineConfig, OnlineDetection

CLIENTS = ("client_000", "client_001", "client_002")


def _detection(alarms: dict[str, int | None], last_round: int = 49, attributed: str | None = None,
               onsets: dict[str, int] | None = None) -> OnlineDetection:
    per_client = {}
    for cid in CLIENTS:
        a = alarms.get(cid)
        per_client[cid] = ClientEvents(cid, a, None if a is None else (onsets or {}).get(cid, a - 2),
                                       None if a is None else a + 1, None if a is None else 5.0, a)
    if attributed is None:
        alarmed = sorted((a, c) for c, a in alarms.items() if a is not None)
        attributed = alarmed[0][1] if alarmed else None
    return OnlineDetection(OnlineConfig(), last_round, per_client, attributed,
                           alarms.get(attributed) if attributed else None, {"client_002": 20})


def test_clopper_pearson_known_values() -> None:
    lo, hi = clopper_pearson(5, 5)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 5), abs=1e-9)
    lo, hi = clopper_pearson(0, 5)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 5), abs=1e-9)
    lo, hi = clopper_pearson(3, 10)
    assert lo == pytest.approx(0.0667, abs=1e-4) and hi == pytest.approx(0.6525, abs=1e-4)
    assert clopper_pearson(0, 0) is None
    with pytest.raises(ValueError):
        clopper_pearson(6, 5)


def test_attack_success_counts_and_deterministic_prediction() -> None:
    assert attack_success_counts([0, 0, 3, 0], target_class=0) == Proportion.of(3, 4)
    assert attack_success_counts([], target_class=0).rate is None
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.Dropout(0.5))
    model.train()
    data = TensorDataset(torch.randn(10, 4), torch.zeros(10, dtype=torch.long))
    state = torch.get_rng_state()
    assert predict(model, data) == predict(model, data)
    assert model.training and torch.equal(state, torch.get_rng_state())


def test_paired_impact_lists_unpaired_seeds_instead_of_dropping_them_silently() -> None:
    impact = paired_impact({1: 0.40, 2: 0.30, 5: 0.2}, {1: 0.45, 2: 0.35, 3: 0.5})
    assert impact.per_seed == pytest.approx({1: -0.05, 2: -0.05})
    assert impact.mean == pytest.approx(-0.05) and impact.n_pairs == 2
    assert impact.unpaired_attacked == (5,) and impact.unpaired_clean == (3,)
    assert paired_impact({}, {1: 0.1}).mean is None


def test_detected_run() -> None:
    o = run_outcome("r", 1, _detection({"client_002": 22}), "client_002", 20)
    assert o.detected and o.attribution_correct and o.alarm_round == 22 and o.actionable_round == 23
    assert o.estimated_onset == 20 and o.onset_abs_error == 0 and o.delay == 2 and o.delay_censored is False
    assert o.honest_alarm_clients == () and o.eligible_honest_client_rounds == 2 * 30


def test_missed_run_is_censored_not_dropped() -> None:
    o = run_outcome("r", 1, _detection({}), "client_002", 20)
    assert o.detected is False and o.attribution_correct is False
    assert o.onset_abs_error is None and o.delay == 29 and o.delay_censored is True


def test_wrong_attribution_and_honest_false_alarm() -> None:
    o = run_outcome("r", 1, _detection({"client_000": 21, "client_002": 24}), "client_002", 20)
    assert o.detected and not o.attribution_correct
    assert o.honest_alarm_clients == ("client_000",) and o.honest_alarm_client_rounds == 1


def test_attacker_alarm_before_true_onset_is_a_false_alarm_not_a_detection() -> None:
    o = run_outcome("r", 1, _detection({"client_002": 22}), "client_002", 30)
    assert o.pre_onset_attacker_alarm is True and o.detected is False and o.delay_censored is True
    assert o.attribution_correct is False  # attributed to the attacker, but before the attack existed


def test_clean_run_has_no_detection_fields() -> None:
    o = run_outcome("r", 1, _detection({"client_001": 30}), None, None)
    assert o.detected is None and o.honest_alarm_clients == ("client_001",)
    assert o.eligible_honest_client_rounds == 3 * 30
    with pytest.raises(ValueError):
        run_outcome("r", 1, _detection({}), "client_002", None)


def test_summary_keeps_misses_in_every_denominator() -> None:
    outcomes = [
        run_outcome("a", 1, _detection({"client_002": 22}), "client_002", 20),
        run_outcome("b", 2, _detection({"client_002": 25}, onsets={"client_002": 21}), "client_002", 20),
        run_outcome("c", 3, _detection({}), "client_002", 20),
        run_outcome("d", 4, _detection({"client_000": 21, "client_002": 23}), "client_002", 20),
    ]
    s = summarize(outcomes)
    assert s.n_runs == 4
    assert (s.detection.k, s.detection.n) == (3, 4) and (s.attribution.k, s.attribution.n) == (2, 4)
    assert s.n_missed == 1 and s.n_wrong_attribution == 1
    assert s.onset_mae_n == 3 and s.onset_errors == (0, 1, 1) and s.onset_mae == pytest.approx(2 / 3)
    assert s.delays_observed == (2, 5, 3) and s.delays_censored_at_least == (29,)
    assert (s.runs_with_honest_alarm.k, s.runs_with_honest_alarm.n) == (1, 4)
    assert s.honest_alarms_per_1000_client_rounds == pytest.approx(1000 * 1 / (4 * 60))
    assert any("missed" in n for n in s.notes)


def test_all_missed_and_edge_cases() -> None:
    s = summarize([run_outcome(str(i), i, _detection({}), "client_002", 20) for i in range(3)])
    assert s.detection.k == 0 and s.onset_mae is None and s.onset_mae_n == 0 and s.n_missed == 3
    assert s.detection.ci95[1] == pytest.approx(1 - 0.025 ** (1 / 3), abs=1e-9)
    empty = summarize([])
    assert empty.n_runs == 0 and empty.detection is None and empty.runs_with_honest_alarm.rate is None


def test_pseudo_replication_and_mixed_conditions_are_rejected() -> None:
    a = run_outcome("a", 1, _detection({}), "client_002", 20)
    b = run_outcome("b", 1, _detection({}), "client_002", 20)
    with pytest.raises(ValueError, match="pseudo-replication"):
        summarize([a, b])
    clean = run_outcome("c", 2, _detection({}), None, None)
    with pytest.raises(ValueError, match="one condition"):
        summarize([a, clean])
