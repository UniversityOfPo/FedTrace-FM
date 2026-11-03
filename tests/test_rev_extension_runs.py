"""Protocol 37 run registration: seeds W3-W5 (K = 5) and K = 10 at seeds 123 / 2026 (no run executed)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.rev.config import load_rev_config
from src.rev.preflight import CONFIRMATION_BASES, PLANNED

ROOT = Path(__file__).resolve().parents[1]
EXT = {"W3": 1966286922, "W4": 1465380568, "W5": 431235112}
ARMS = ("clean", "badnets", "drift")


def _rule_seeds() -> list[int]:
    d = hashlib.sha256(b"FedTrace-C4AI4C-workshop-confirmation|2026-10-26").digest()
    return [int.from_bytes(d[i:i + 4], "big") % 2**31 for i in range(0, 32, 4)]


def test_extension_seeds_follow_the_protocol_26_rule() -> None:
    assert _rule_seeds()[2:5] == list(EXT.values())


def test_preflight_lists_exactly_the_fifteen_extension_runs() -> None:
    ext = {n for n in PLANNED if n.startswith(("W3", "W4", "W5", "K10"))}
    assert ext == {f"{w}_{a}" for w in EXT for a in ARMS} | {f"K10_{a}_d{s}" for s in (123, 2026) for a in ARMS}
    for n in ext:
        run = PLANNED[n]
        assert (ROOT / run.config).exists() and run.rounds == 50
        assert run.clients == (10 if n.startswith("K10") else 5)
    assert all((ROOT / b).exists() for b in CONFIRMATION_BASES)


def test_extension_base_configs_differ_only_where_intended() -> None:
    w1 = (ROOT / "configs/cifar100_fedavg_dirichlet_a01_seed1178015705_50rounds.yaml").read_text(encoding="utf-8")
    for seed in EXT.values():
        base = (ROOT / f"configs/cifar100_fedavg_dirichlet_a01_seed{seed}_50rounds.yaml").read_text(encoding="utf-8")
        assert f"\nseed: {seed}\n" in base and "\nnum_clients: 5\n" in base
        body = [l for l in base.splitlines() if not l.startswith("#") and "seed" not in l and "experiment_id" not in l]
        ref = [l for l in w1.splitlines() if not l.startswith("#") and "seed" not in l and "experiment_id" not in l]
        assert body == ref
    for seed in (123, 2026):
        k5 = (ROOT / f"configs/cifar100_fedavg_dirichlet_a01_seed{seed}_50rounds.yaml").read_text(encoding="utf-8")
        k10 = (ROOT / f"configs/cifar100_fedavg_dirichlet_a01_k10_seed{seed}_50rounds.yaml").read_text(encoding="utf-8")
        assert "\nnum_clients: 10\n" in k10 and "\nclients_per_round: 10\n" in k10 and "\nminimum_updates: 10\n" in k10
        strip = lambda t: [l for l in t.splitlines() if not l.startswith("#") and "_50rounds" not in l  # noqa: E731
                           and not l.startswith(("num_clients", "clients_per_round", "minimum_updates"))]
        assert strip(k5) == strip(k10)


def test_extension_rev_configs_keep_the_frozen_arms() -> None:
    for name in [f"{w}_{a}" for w in EXT for a in ARMS] + [f"K10_{a}_d{s}" for s in (123, 2026) for a in ARMS]:
        text = (ROOT / f"configs/rev/{name}.rev.yaml").read_text(encoding="utf-8")
        assert f"run_id: {name}" in text
        assert "type: none" in text or "t_ref: 20" in text
        load_rev_config(ROOT / f"configs/rev/{name}.rev.yaml")
        if "_badnets" in name:
            assert all(s in text for s in ("malicious_client_id: client_002", "onset_round: 20", "target_class: 0",
                                           "poison_fraction: 0.5", "trigger_size: 3"))
        if "_drift" in name:
            assert all(s in text for s in ("drifted_client_id: client_002", "onset_round: 20",
                                           "retained_sample_fraction: 0.5"))
        if "_clean" in name:
            assert "type: none" in text
