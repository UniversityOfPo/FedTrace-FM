"""Update logger fidelity, read-only behaviour, atomicity, interruption and storage limits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.rev import update_log as ul
from src.rev.hashing import load_tensor_state, state_content_sha256
from src.rev.replay import verify_update_log
from src.rev.update_log import LogInterruptedError, StorageLimitError, UpdateLogConfig, UpdateLogError, UpdateLogger
from tests.rev_toy_federation import rng_snapshot, same_rng, seed_everything


def _state(scale: float = 1.0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(int(scale * 1000))
    return {
        "w": torch.randn(5, 3, generator=g) * scale,
        "bn.running_mean": torch.randn(3, generator=g),
        "bn.num_batches_tracked": torch.tensor(17, dtype=torch.int64),
    }


def _open(tmp_path: Path, **kw) -> UpdateLogger:
    kw.setdefault("min_free_bytes", 0)
    return UpdateLogger.open(UpdateLogConfig(root=tmp_path / "log", run_id="t", **kw))


def _one_round(logger: UpdateLogger, round_id: int, pre: dict, subs: dict, after: dict) -> None:
    logger.begin_round(round_id)
    for cid, state in subs.items():
        logger.log_client(cid, round_id, state, pre, num_samples=10, pre_transform_state=state)
    logger.log_round_end(round_id, after, "completed", successful_client_ids=tuple(subs))


def test_logger_never_mutates_tensors_and_draws_no_random_numbers(tmp_path: Path) -> None:
    seed_everything(3)
    pre, sub, after = _state(1.0), _state(2.0), _state(3.0)
    copies = {k: v.clone() for k, v in sub.items()}
    versions = {k: v._version for k, v in sub.items()}
    before = rng_snapshot()
    logger = _open(tmp_path)
    _one_round(logger, 0, pre, {"c0": sub}, after)
    logger.seal()
    assert same_rng(before, rng_snapshot())
    assert all(torch.equal(sub[k], copies[k]) and sub[k]._version == versions[k] for k in sub)
    assert all(t.device.type == "cpu" for t in sub.values())


def test_logged_states_are_bit_exact_and_keep_their_dtypes(tmp_path: Path) -> None:
    pre, sub, after = _state(1.0), _state(2.0), _state(3.0)
    logger = _open(tmp_path, pre_transform_clients=("c0",))
    _one_round(logger, 0, pre, {"c0": sub}, after)
    logger.seal()
    records = ul.read_manifest(tmp_path / "log")
    kinds = [(r["kind"], r.get("client_id"), r.get("round_id")) for r in records[1:]]
    assert kinds == [("global", None, -1), ("submitted", "c0", 0), ("pre_transform", "c0", 0), ("global", None, 0)]
    loaded = load_tensor_state(tmp_path / "log" / records[2]["file"])
    assert set(loaded) == set(sub)
    for key in sub:
        assert loaded[key].dtype == sub[key].dtype and torch.equal(loaded[key], sub[key])
    assert records[2]["content_sha256"] == state_content_sha256(sub)
    assert records[2]["pre_round_global_sha256"] == state_content_sha256(pre)
    assert records[3]["identical_to_submitted"] is True
    assert verify_update_log(tmp_path / "log").ok


def test_interrupted_write_leaves_partial_file_without_manifest_record(tmp_path: Path, monkeypatch) -> None:
    logger = _open(tmp_path)
    pre = _state(1.0)
    real_save = torch.save

    def crash(obj, handle, *a, **k):
        handle.write(b"half-written")
        raise OSError("simulated power loss")

    monkeypatch.setattr(torch, "save", crash)
    with pytest.raises(OSError):
        logger.log_client("c0", 0, _state(2.0), pre, num_samples=3)
    monkeypatch.setattr(torch, "save", real_save)
    assert isinstance(logger.failure, OSError)
    root = tmp_path / "log"
    assert list(root.rglob("*.partial"))
    assert [r["kind"] for r in ul.read_manifest(root)] == ["header"]
    with pytest.raises(UpdateLogError):
        logger.seal()  # a failed log can never be sealed as COMPLETED
    report = verify_update_log(root)
    assert not report.ok and "not sealed" in report.problems[0]
    with pytest.raises(LogInterruptedError):
        UpdateLogger.open(UpdateLogConfig(root=root, run_id="t", min_free_bytes=0))


def test_sealed_log_is_never_reopened_or_overwritten(tmp_path: Path) -> None:
    logger = _open(tmp_path)
    _one_round(logger, 0, _state(1.0), {"c0": _state(2.0)}, _state(3.0))
    logger.seal()
    with pytest.raises(UpdateLogError, match="sealed"):
        UpdateLogger.open(UpdateLogConfig(root=tmp_path / "log", run_id="t", min_free_bytes=0))
    with pytest.raises(UpdateLogError):
        logger.log_round_end(1, _state(3.0), "completed")


def test_planned_size_over_budget_is_refused_before_any_state_is_written(tmp_path: Path) -> None:
    nbytes = sum(t.numel() * t.element_size() for t in _state().values())
    logger = _open(tmp_path, max_bytes=10 * nbytes, expected_client_states=10, expected_global_states=1)
    with pytest.raises(StorageLimitError, match="planned"):
        logger.log_client("c0", 0, _state(2.0), _state(1.0), num_samples=3)
    assert not list((tmp_path / "log" / "states").iterdir())


def test_per_write_budget_and_free_disk_reserve(tmp_path: Path, monkeypatch) -> None:
    nbytes = sum(t.numel() * t.element_size() for t in _state().values())
    logger = _open(tmp_path, max_bytes=2 * nbytes)
    logger.log_client("c0", 0, _state(2.0), _state(1.0), num_samples=3)  # writes initial global + submission
    with pytest.raises(StorageLimitError, match="budget"):
        logger.log_client("c1", 0, _state(4.0), _state(1.0), num_samples=3)

    class _Usage:
        free = 1000

    other = UpdateLogger.open(UpdateLogConfig(root=tmp_path / "log2", run_id="t", min_free_bytes=10**6))
    monkeypatch.setattr(ul.shutil, "disk_usage", lambda _p: _Usage())
    with pytest.raises(StorageLimitError, match="reserve"):
        other.log_client("c0", 0, _state(2.0), _state(1.0), num_samples=3)


def test_pre_round_global_must_match_the_logged_chain(tmp_path: Path) -> None:
    logger = _open(tmp_path)
    _one_round(logger, 0, _state(1.0), {"c0": _state(2.0)}, _state(3.0))
    with pytest.raises(UpdateLogError, match="differs"):
        logger.log_client("c0", 1, _state(2.0), _state(9.0), num_samples=3)


def test_logging_must_start_at_round_zero(tmp_path: Path) -> None:
    logger = _open(tmp_path)
    with pytest.raises(UpdateLogError, match="round 0"):
        logger.log_client("c0", 5, _state(2.0), _state(1.0), num_samples=3)


def test_verification_detects_tampering_unlisted_and_missing_files(tmp_path: Path) -> None:
    logger = _open(tmp_path)
    _one_round(logger, 0, _state(1.0), {"c0": _state(2.0), "c1": _state(4.0)}, _state(3.0))
    logger.seal()
    root = tmp_path / "log"
    assert verify_update_log(root).ok

    target = next(root.glob("states/r0000_submitted_c1.pt"))
    data = bytearray(target.read_bytes())
    data[-20] ^= 0xFF
    target.write_bytes(bytes(data))
    (root / "states" / "extra.pt").write_bytes(b"x")
    report = verify_update_log(root)
    assert not report.ok
    assert any("file hash mismatch" in p for p in report.problems)
    assert any("not in the manifest" in p for p in report.problems)

    with (root / ul.MANIFEST_NAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": "global", "round_id": 1}) + "\n")
    assert any("seal" in p for p in verify_update_log(root).problems)
