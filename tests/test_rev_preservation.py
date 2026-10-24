"""Backup and preservation checks: NDSS freeze and FM files unchanged, ignore rules, package isolation.

The preservation area lives outside the repository. Its location is
``$FEDTRACE_PRESERVATION_ROOT`` if set, otherwise ``FedTrace_Preservation`` at the
root of the drive holding the repository. Each manifest is first checked against
the backup's own ``SEAL.sha256``. Where the area is absent the hash checks are
skipped, unless ``FEDTRACE_REQUIRE_PRESERVATION=1`` (then they fail).
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRESERVATION_ROOT = Path(os.environ.get("FEDTRACE_PRESERVATION_ROOT") or Path(ROOT.anchor) / "FedTrace_Preservation")
REQUIRE = os.environ.get("FEDTRACE_REQUIRE_PRESERVATION") == "1"
MANIFEST = "manifests/MANIFEST.source.pre_copy.sha256.csv"
NDSS_BACKUP = "NDSS_freeze_20261020"
FM_BACKUP = "FM_untracked_20261024"
SMALL = 5 * 1024 * 1024
REV_RULES = ["results_rev/", "checkpoints_rev/", "logs_rev/"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(backup: str) -> list[dict[str, str]]:
    base = PRESERVATION_ROOT / backup
    manifest = base / MANIFEST
    if not manifest.exists():
        if REQUIRE:
            pytest.fail(f"preservation manifest missing: {backup}/{MANIFEST}")
        pytest.skip(f"preservation manifest not available on this machine: {backup}/{MANIFEST}")
    sealed = {}
    for line in (base / "SEAL.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.strip().split(" ", 1)
            sealed[name.lstrip("*")] = digest
    assert sealed.get(MANIFEST) == _sha(manifest), f"{backup}: manifest does not match its seal"
    with manifest.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_ndss_frozen_files_are_unchanged() -> None:
    """Every frozen file under 5 MiB (all code, configs, traces, summaries) except the approved .gitignore."""
    rows = [r for r in _rows(NDSS_BACKUP) if int(r["size_bytes"]) < SMALL and r["relative_path"] != ".gitignore"]
    assert len(rows) > 500
    changed = [r["relative_path"] for r in rows
               if not (ROOT / r["relative_path"]).is_file() or _sha(ROOT / r["relative_path"]) != r["sha256"]]
    assert changed == []


def test_fm_untracked_files_match_their_backup() -> None:
    rows = _rows(FM_BACKUP)
    assert len(rows) == 41
    assert [r["relative_path"] for r in rows if _sha(ROOT / r["relative_path"]) != r["sha256"]] == []


NDSS_BASE_COMMIT = "9307b96610b739442e14f99482158ffd3b5a624c"  # last NDSS-archive commit before the reviewer response


def test_gitignore_only_appends_the_three_reviewer_response_rules() -> None:
    committed = subprocess.run(["git", "-C", str(ROOT), "show", f"{NDSS_BASE_COMMIT}:.gitignore"],
                               capture_output=True, check=True).stdout.decode("utf-8").splitlines()
    current = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert current[: len(committed)] == committed
    added = [line for line in current[len(committed):] if line.strip() and not line.startswith("#")]
    assert added == REV_RULES


@pytest.mark.parametrize("path, ignored", [
    ("results_rev/R1/x.csv", True), ("checkpoints_rev/R1/x.pt", True), ("logs_rev/R1/update_log/manifest.jsonl", True),
    ("src/rev/update_log.py", False), ("configs/rev/R1_exp016_logged.rev.yaml", False),
    ("tests/test_rev_update_log.py", False), ("tests/rev_toy_federation.py", False),
])
def test_rev_outputs_are_ignored_and_rev_sources_are_trackable(path: str, ignored: bool) -> None:
    result = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", "--no-index", path], capture_output=True)
    assert (result.returncode == 0) == ignored


def test_rev_package_does_not_depend_on_paused_fm_code() -> None:
    pattern = re.compile(r"^\s*(from|import)\s+src\.fm\b", re.MULTILINE)
    offenders = [p.name for p in (ROOT / "src" / "rev").glob("*.py") if pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []
