"""Protocol 26 run registration and analysis parametrisation (no run executed; synthetic traces only)."""

from __future__ import annotations

from pathlib import Path

from src.rev import contamination as cm
from src.rev import evidence_matrix as em
from src.rev import unified_outcomes as uo
from src.rev.config import load_rev_config
from src.rev.preflight import CONFIRMATION_BASES, PLANNED
from tests.test_rev_evidence_matrix import _series, _write_run

ROOT = Path(__file__).resolve().parents[1]
W = {"W1": 1178015705, "W2": 1463065807}


def test_preflight_lists_exactly_the_six_frozen_runs() -> None:
    conf = {n for n in PLANNED if n.startswith("W")}
    assert conf == {f"{w}_{a}" for w in W for a in ("clean", "badnets", "drift")}
    for n in conf:
        assert (ROOT / PLANNED[n].config).exists() and PLANNED[n].rounds == 50 and PLANNED[n].clients == 5
    assert all((ROOT / b).exists() for b in CONFIRMATION_BASES)


def test_confirmation_configs_match_protocol_26() -> None:
    for w, seed in W.items():
        base = (ROOT / f"configs/cifar100_fedavg_dirichlet_a01_seed{seed}_50rounds.yaml").read_text(encoding="utf-8")
        assert f"\nseed: {seed}\n" in base and "\nclients_per_round: 5\n" in base and "\nnum_clients: 5\n" in base
        for arm in ("clean", "badnets", "drift"):
            text = (ROOT / f"configs/rev/{w}_{arm}.rev.yaml").read_text(encoding="utf-8")
            assert f"seed{seed}_50rounds.yaml" in text and f"run_id: {w}_{arm}" in text
            load_rev_config(ROOT / f"configs/rev/{w}_{arm}.rev.yaml")
        assert "poison_fraction: 0.5" in (ROOT / f"configs/rev/{w}_badnets.rev.yaml").read_text(encoding="utf-8")
        assert "retained_sample_fraction: 0.5" in (ROOT / f"configs/rev/{w}_drift.rev.yaml").read_text(encoding="utf-8")


def test_sealed_seeds_absent_from_all_configs() -> None:
    sealed = ("978310317", "478485063", "1082418221", "1966286922")
    for p in (ROOT / "configs").rglob("*.yaml"):
        text = p.read_text(encoding="utf-8")
        assert not any(s in text for s in sealed), p


def test_registry_and_truth_for_confirmation_runs() -> None:
    assert set(cm.CONF) == {f"{w}_{a}" for w in W for a in ("clean", "badnets", "drift")}
    assert cm.truth_of([], "W1_badnets") == ("client_002", 20, True)
    assert cm.truth_of([], "W2_drift") == ("client_002", 20, False)
    assert cm.truth_of([], "W1_clean") == (None, None, False)
    assert cm.clean_run_of("W2_badnets", "W2") == "W2_clean"
    assert cm.clean_run_of("X2_badnets_d123", "123") == "X1_clean_d123"
    assert uo.family("W1_drift", []) == "benign class-composition drift (confirmation)"


def test_evidence_matrix_confirmation_conditions(tmp_path: Path) -> None:
    for cond, (pattern, _) in em.CONFIRMATION_CONDITIONS.items():
        _write_run(tmp_path, pattern.format(s="W1"), _series("2026", cond), 0.5)
    d = tmp_path / "results_rev/pilot/W1_badnets__vs__W1_clean"
    d.mkdir(parents=True)
    (d / "summary.json").write_text("{}")
    for name in ("W1_badnets", "W1_clean"):
        (d / f"effect_{name}.csv").write_text("round_id,clean_top1,untriggered_target_rate,asr\n49,0.5,0.001,0.4\n")
    rep = em.build(tmp_path, ["W1"], with_logs=False, out_dir=tmp_path / "out", conditions=em.CONFIRMATION_CONDITIONS)
    assert sorted(rep["rows"]) == ["W1/badnets", "W1/clean", "W1/drift"]
    assert rep["rows"]["W1/clean"]["final_effect"]["asr"] == 0.4  # clean effect read via the BadNets pair
    assert rep["rows"]["W1/badnets"]["malicious"] and not rep["rows"]["W1/drift"]["malicious"]
