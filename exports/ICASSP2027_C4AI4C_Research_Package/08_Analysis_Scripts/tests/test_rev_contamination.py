"""Protocol 25 contamination mechanism: synthetic traces only (no development outcome used)."""

from __future__ import annotations

import math
import random
import statistics

import pytest

from src.rev import contamination as cm
from src.rev import unified_outcomes as uo

CLIENTS = [f"client_00{i}" for i in range(5)]
BASE = [0.20, 0.24, 0.28, 0.32, 0.36]  # client_002 is the median client


def trace(change=None, trend=0.0, rounds=50, base=BASE, clients=CLIENTS):
    out = []
    for r in range(rounds):
        for i, c in enumerate(clients):
            t = None if r == 0 else base[i] + 0.004 * math.sin(1.3 * r + 2.1 * i) + trend * r
            if change and c in change and r >= change[c][0] and t is not None:
                t = change[c][1] + 0.004 * math.sin(1.3 * r + 2.1 * i) + trend * r
            n = 10 + i + 0.05 * math.sin(r + i)
            out.append({"round_id": r, "client_id": c, "update_l2_norm": n, "self_prev_round_cosine": t})
    return out


def test_clip_form_equals_median_for_odd_and_even_k() -> None:
    rng = random.Random(0)
    for n_others in (4, 9, 2, 3):
        for _ in range(500):
            others = [rng.random() for _ in range(n_others)]
            x = rng.uniform(-0.2, 1.2)
            assert cm.clip_form(others, x) == pytest.approx(cm.median_with(others, x), abs=1e-15)


def test_lemma_a_bounds_and_crossing_condition() -> None:
    rng = random.Random(1)
    for _ in range(500):
        others = sorted(rng.random() for _ in range(4))
        lo, hi = others[1], others[2]
        x0, x1 = rng.random(), rng.random()
        dc = cm.median_with(others, x1) - cm.median_with(others, x0)
        assert abs(dc) <= min(abs(x1 - x0), hi - lo) + 1e-15
        if (x0 <= lo and x1 <= lo) or (x0 >= hi and x1 >= hi):
            assert dc == 0.0
        elif dc != 0.0:
            assert (dc > 0) == (x1 > x0)


def test_upward_crossing_transfers_equally_and_explains_honest_alarms() -> None:
    rows = trace(change={"client_002": (20, 0.50)})
    clean = trace()
    run = cm.analyse_run(rows, clean, "client_002")
    assert run["identity_max_abs_error"] < 1e-12
    for r in (21, 25):
        rec = run["per"][r]
        assert rec["dep"]["client_002"] > 0.03 and rec["retro"]["client_002"] > 0.03
        for k in ("client_000", "client_001", "client_003", "client_004"):  # Lemma B: same Δd for every other client
            assert rec["dz_total"][k] * run["sigma"][k] == pytest.approx(rec["retro"]["client_002"], abs=1e-12)
    alarms = cm.v3_alarms(rows, run["res"])
    honest = [c for c in CLIENTS if c != "client_002"]
    expl = cm.explain_alarms(rows, run, honest)
    assert expl and all(a["classification"] == "t_res-necessary" and a["dep_explained"] and a["dep_best_source"] == "client_002"
                        and a["retro_explained"] for a in expl)
    assert "client_002" not in alarms or alarms["client_002"][0] > min(a["alarm"] for a in expl)  # honest first


def test_downward_crossing_raises_no_honest_t_res_alarm() -> None:
    rows = trace(change={"client_002": (20, 0.05)})
    run = cm.analyse_run(rows)
    assert run["per"][25]["dep"]["client_002"] < 0
    honest = [c for c in CLIENTS if c != "client_002"]
    assert not [a for a in cm.explain_alarms(rows, run, honest) if a["classification"] == "t_res-necessary"]


def test_no_crossing_no_displacement() -> None:
    rows = trace(change={"client_004": (20, 0.70)})  # already the highest client; moves further up
    run = cm.analyse_run(rows, trace(), "client_004")
    assert all(abs(run["per"][r]["retro"]["client_004"]) < 1e-12 for r in range(20, 30))
    assert all(abs(run["per"][r]["dep"]["client_004"]) < 0.01 for r in range(20, 30))


def test_common_trend_flags_b0_not_mechanism() -> None:
    run = cm.analyse_run(trace(trend=0.002))
    rates = cm.clean_rates(run)
    assert rates["B0_flag_rate"] > 0.5 and rates["mechanism_flag_rate"] < rates["B0_flag_rate"]


def test_displacement_tests_on_constructed_case() -> None:
    rows, clean = trace(change={"client_002": (20, 0.50)}), trace()
    run = cm.analyse_run(rows, clean, "client_002")
    t = cm.displacement_tests(run, [c for c in CLIENTS if c != "client_002"], range(20, 30))
    assert t["n_sign_test"] > 0 and t["sign_agree_dep"] == t["n_sign_test"]
    assert t["explained_fraction_of_total_by_source_term"] == pytest.approx(1.0, abs=1e-9)


def test_unified_detectors_and_family_labels() -> None:
    rows = trace(change={"client_002": (20, 0.50)})
    dets = uo.detectors(rows)
    assert set(dets) == {"v3", "v3 magnitude-only", "v3 t_res-only", "v3 LOO-ref", "v3 trimmed-ref", "v3 historical-ref",
                         "ZP-norm", "CUSUM-norm", "PH-norm"}
    assert dets["v3 magnitude-only"] == {}  # norms unchanged in this construction
    assert uo.family("X2_badnets_d123", rows) == "BadNets data poisoning"
    assert "clean" in uo.family("federated/cifar100_multikrum_dirichlet_a01_seed123_50rounds", rows)
