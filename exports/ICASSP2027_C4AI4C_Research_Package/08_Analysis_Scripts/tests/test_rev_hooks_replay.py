"""Hooked runner equivalence, logging non-interference, and offline replay correctness (CPU toy federation)."""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
import torch

import src.experiments.run_federated_cifar100 as ndss
from src.attacks.traceability import AttackConfig, apply_gaussian_update_noise
from src.federated.client import FederatedClient
from src.federated.runner import FederatedRunner
from src.rev import update_log as ul
from src.rev.hashing import load_tensor_state, states_bit_identical
from src.rev.hooks import HookedRunner, RevHooks, ndss_hooks_installed
from src.rev.replay import (
    compare_trace_rows,
    recompute_trace_rows,
    replay_aggregation,
    verify_update_log,
)
from src.rev.update_log import UpdateLogConfig, UpdateLogger
from tests.rev_toy_federation import CLIENT_IDS, TinyNet, build_and_run, same_rng

ROUNDS = 4
TRAINABLE = frozenset(name for name, _ in TinyNet().named_parameters())
GAUSSIAN = AttackConfig(attack_type="gaussian_update_noise", malicious_client_id="client_002",
                        attack_start_round=2, attack_strength_rho=1.0)


def _gaussian(cid, round_id, state, pre):
    if GAUSSIAN.is_active(round_id, cid):
        return apply_gaussian_update_noise(state, pre, TRAINABLE, rho=1.0, seed=7, round_id=round_id, client_id=cid)
    return state


def _logger(tmp: Path, **kw) -> UpdateLogger:
    return UpdateLogger.open(UpdateLogConfig(root=tmp / "log", run_id="toy", min_free_bytes=0, **kw))


def _assert_identical_runs(a, b) -> None:
    assert len(a.globals_after) == len(b.globals_after) == ROUNDS
    for ga, gb in zip(a.globals_after, b.globals_after):
        assert states_bit_identical(ga, gb)
    assert [dict(h.train_metrics) for h in a.history] == [dict(h.train_metrics) for h in b.history]
    assert compare_trace_rows(a.trace_rows, b.trace_rows) == []
    assert same_rng(a.final_rng, b.final_rng)


def test_hooked_runner_without_hooks_is_the_original_runner(tmp_path: Path) -> None:
    plain = build_and_run(tmp_path / "a", rounds=ROUNDS)
    hooked = build_and_run(tmp_path / "b", rounds=ROUNDS, runner_factory=functools.partial(HookedRunner))
    _assert_identical_runs(plain, hooked)


def test_logging_does_not_change_training_and_replay_is_bit_exact(tmp_path: Path) -> None:
    plain = build_and_run(tmp_path / "a", rounds=ROUNDS, transform=_gaussian, attack_config=GAUSSIAN)
    hooks = RevHooks(update_logger=_logger(tmp_path, pre_transform_clients=("client_002",)))
    logged = build_and_run(tmp_path / "b", rounds=ROUNDS, transform=_gaussian, attack_config=GAUSSIAN,
                           runner_factory=functools.partial(HookedRunner, rev_hooks=hooks))
    hooks.update_logger.seal()
    _assert_identical_runs(plain, logged)

    root = tmp_path / "log"
    report = verify_update_log(root)
    assert report.ok, report.problems
    # Re-aggregating the logged submissions reproduces every logged global state bit-exactly ...
    assert replay_aggregation(root) == {r: True for r in range(ROUNDS)}
    # ... the logged post-round globals are exactly the states the live server held ...
    globals_ = sorted((r for r in ul.read_manifest(root) if r.get("kind") == "global"), key=lambda r: r["round_id"])
    for record, live in zip(globals_[1:], logged.globals_after):
        assert states_bit_identical(load_tensor_state(root / record["file"]), live)
    # ... and the NDSS trace rows recomputed offline equal the rows computed online, field by field.
    replayed = recompute_trace_rows(root, CLIENT_IDS, TRAINABLE, GAUSSIAN)
    from src.attacks.traceability import apply_naive_detector

    assert compare_trace_rows(apply_naive_detector(logged.trace_rows, GAUSSIAN), replayed) == []


def test_attacker_pre_transform_state_is_logged_separately(tmp_path: Path) -> None:
    hooks = RevHooks(update_logger=_logger(tmp_path, pre_transform_clients=("client_002",)))
    build_and_run(tmp_path / "b", rounds=ROUNDS, transform=_gaussian, attack_config=GAUSSIAN,
                  runner_factory=functools.partial(HookedRunner, rev_hooks=hooks))
    hooks.update_logger.seal()
    records = ul.read_manifest(tmp_path / "log")
    pre = {r["round_id"]: r for r in records if r.get("kind") == "pre_transform"}
    sub = {r["round_id"]: r for r in records if r.get("kind") == "submitted" and r["client_id"] == "client_002"}
    assert sorted(pre) == list(range(ROUNDS))
    for r in range(ROUNDS):
        same = pre[r]["content_sha256"] == sub[r]["content_sha256"]
        assert same == (r < 2), r  # identical before the attack starts, different after


def test_rev_attack_runs_before_the_callers_transform_and_the_logger_sees_the_final_state(tmp_path: Path) -> None:
    seen_by_inner: dict[tuple[str, int], torch.Tensor] = {}

    def rev_attack(cid, round_id, state, pre):
        if cid != "client_002":
            return state
        out = dict(state)
        out["fc2.bias"] = state["fc2.bias"] + 1.0
        return out

    def inner(cid, round_id, state, pre):  # stands in for the NDSS transform + trace bookkeeping
        seen_by_inner[(cid, round_id)] = state["fc2.bias"].clone()
        return state

    hooks = RevHooks(update_logger=_logger(tmp_path, pre_transform_clients=("client_002",)), attack_transform=rev_attack)
    build_and_run(tmp_path / "b", rounds=2, transform=inner,
                  runner_factory=functools.partial(HookedRunner, rev_hooks=hooks))
    hooks.update_logger.seal()
    root = tmp_path / "log"
    records = ul.read_manifest(root)
    for r in range(2):
        sub = next(x for x in records if x.get("kind") == "submitted" and x["client_id"] == "client_002" and x["round_id"] == r)
        pre = next(x for x in records if x.get("kind") == "pre_transform" and x["round_id"] == r)
        submitted = load_tensor_state(root / sub["file"])["fc2.bias"]
        honest = load_tensor_state(root / pre["file"])["fc2.bias"]
        assert torch.equal(seen_by_inner[("client_002", r)], submitted)
        assert torch.equal(submitted, honest + 1.0)


def test_peer_mean_cosine_depends_on_peer_order_in_the_last_bits() -> None:
    """Documents the NDSS limitation noted in src/rev/replay.py: the peer mean is an order-dependent fp64 sum."""
    from src.attacks.traceability import compute_peer_mean_cosine_similarity

    keys = ["w"]
    client = {"w": torch.tensor([1.0, 0.0], dtype=torch.float64)}
    peers = [{"w": torch.tensor([1e16, 1.0], dtype=torch.float64)}, {"w": torch.tensor([1.0, 1.0], dtype=torch.float64)},
             {"w": torch.tensor([-1e16, 1.0], dtype=torch.float64)}]
    forward = compute_peer_mean_cosine_similarity(client, peers, keys)
    backward = compute_peer_mean_cosine_similarity(client, list(reversed(peers)), keys)
    swapped = compute_peer_mean_cosine_similarity(client, [peers[0], peers[2], peers[1]], keys)
    assert len({forward, backward, swapped}) > 1


def test_logging_failure_aborts_the_run_instead_of_failing_a_client(tmp_path: Path, monkeypatch) -> None:
    hooks = RevHooks(update_logger=_logger(tmp_path))

    def boom(*a, **k):
        raise OSError("disk gone")

    monkeypatch.setattr(hooks.update_logger, "_check_storage", boom)
    with pytest.raises(RuntimeError, match="update logging failed in round 0"):
        build_and_run(tmp_path / "b", rounds=ROUNDS, runner_factory=functools.partial(HookedRunner, rev_hooks=hooks))
    assert not (tmp_path / "log" / ul.SEAL_NAME).exists()


def test_model_initialization_fingerprint_is_recorded(tmp_path: Path) -> None:
    a, b = RevHooks(), RevHooks()
    build_and_run(tmp_path / "a", rounds=1, runner_factory=functools.partial(HookedRunner, rev_hooks=a))
    build_and_run(tmp_path / "b", rounds=1, runner_factory=functools.partial(HookedRunner, rev_hooks=b))
    assert a.init_global_sha256 is not None and a.init_global_sha256 == b.init_global_sha256


def test_ndss_module_names_are_patched_and_restored() -> None:
    hooks = RevHooks(dataset_wrapper=lambda cid, ds: ds)
    with ndss_hooks_installed(ndss, hooks):
        assert isinstance(ndss.FederatedRunner, functools.partial) and ndss.FederatedRunner.func is HookedRunner
        assert ndss.FederatedClient is not FederatedClient
        with pytest.raises(RuntimeError, match="already patched"):
            with ndss_hooks_installed(ndss, hooks):
                pass
    assert ndss.FederatedRunner is FederatedRunner and ndss.FederatedClient is FederatedClient


def test_ndss_module_names_are_restored_after_an_exception() -> None:
    with pytest.raises(ValueError):
        with ndss_hooks_installed(ndss, RevHooks()):
            raise ValueError("x")
    assert ndss.FederatedRunner is FederatedRunner and ndss.FederatedClient is FederatedClient


def test_replay_detects_a_global_state_that_was_not_produced_by_aggregation(tmp_path: Path) -> None:
    hooks = RevHooks(update_logger=_logger(tmp_path))
    build_and_run(tmp_path / "b", rounds=2, runner_factory=functools.partial(HookedRunner, rev_hooks=hooks))
    logger = hooks.update_logger
    logger.seal()
    root = tmp_path / "log"
    record = next(r for r in ul.read_manifest(root) if r.get("kind") == "global" and r["round_id"] == 1)
    state = load_tensor_state(root / record["file"])
    state["fc2.bias"] = state["fc2.bias"] + 1e-6
    torch.save(state, root / record["file"])
    assert not verify_update_log(root).ok  # file and content hashes no longer match the manifest
