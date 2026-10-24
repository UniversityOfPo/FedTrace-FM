"""CPU-only preflight for Stage B (no training, no CUDA context, no download, no files written unless --out).

    python -m src.rev.preflight                 # Stage B runs: S0a, S0b, R0, R1
    python -m src.rev.preflight --runs X1_clean_d123 X2_badnets_d123   # Stage C pilot pair
    python -m src.rev.preflight --runs R0 R1 --out results_rev/preflight.json

Checks (each PASS / FAIL / WARN):
    git          HEAD commit and branch; no tracked modification; every repository module imported by the
                 two entry points and every config used is committed and unmodified (execution-clean)
    configs      every planned config parses with the NDSS / rev loaders
    outputs      no planned output, checkpoint or log directory already exists with content
    archive      archived EXP_016 artifacts needed for the R0 comparison exist and match the NDSS freeze
                 manifest (when the preservation area is available)
    dataset      CIFAR-100 files present and pass torchvision's integrity check with download disabled
    environment  Python / torch / CUDA build / cuDNN versus the archived EXP_016 run; GPU name, driver and free
                 memory from nvidia-smi (WARN when unavailable: they cannot be compared without it)
    disk         free space >= planned outputs + update logs + reserve (default 100 GiB)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_GIB = 1024**3
STATE_BYTES = 44_919_088          # measured ResNet-18 (CIFAR-100) state dict: 11,220,132 fp32 params + buffers
RUN_OUTPUT_BYTES = 520 * 1024**2  # archived EXP_016: results 1.1 MiB + checkpoints 515 MiB
ARCHIVE = "cifar100_fedavg_dirichlet_a01_gaussian_attack_client002_round20_seed123"
ARCHIVE_FILES = (
    f"results/federated/{ARCHIVE}/client_round_trace.csv", f"results/federated/{ARCHIVE}/round_history.csv",
    f"results/federated/{ARCHIVE}/client_partition_indices.json", f"results/federated/{ARCHIVE}/summary.json",
    f"results/federated/{ARCHIVE}/final_test_metrics.json", f"checkpoints/federated/{ARCHIVE}/final_global_model.pt",
    f"checkpoints/federated/{ARCHIVE}/runner_latest.pt", f"checkpoints/federated/{ARCHIVE}/runner_latest_server.pt",
)


@dataclass(frozen=True)
class PlannedRun:
    name: str
    config: str
    kind: str                      # "ndss" (NDSS runner) or "rev" (rev entry point)
    rounds: int
    clients: int
    logged_clients_per_round: int  # submitted + pre-transform states per round (0 = no update log)
    gpu_hours: float               # estimate


PLANNED: dict[str, PlannedRun] = {
    "S0a": PlannedRun("S0a", "configs/rev/S0a_smoke_ndss.yaml", "ndss", 3, 5, 0, 0.1),
    "S0b": PlannedRun("S0b", "configs/rev/S0b_smoke_logged.rev.yaml", "rev", 3, 5, 6, 0.1),
    "R0": PlannedRun("R0", "configs/rev/R0_exp016_repro.yaml", "ndss", 50, 5, 0, 1.33),
    "R1": PlannedRun("R1", "configs/rev/R1_exp016_logged.rev.yaml", "rev", 50, 5, 6, 1.33),
    "X1_clean_d123": PlannedRun("X1_clean_d123", "configs/rev/X1_clean_d123.rev.yaml", "rev", 50, 5, 5, 1.39),
    "X2_badnets_d123": PlannedRun("X2_badnets_d123", "configs/rev/X2_badnets_d123.rev.yaml", "rev", 50, 5, 5, 1.39),
}
STAGE_B = ("S0a", "S0b", "R0", "R1")


@dataclass
class Check:
    name: str
    status: str
    detail: dict[str, Any] = field(default_factory=dict)


def planned_bytes(run: PlannedRun) -> int:
    log = (run.logged_clients_per_round * run.rounds + run.rounds + 1) * STATE_BYTES if run.logged_clients_per_round else 0
    return RUN_OUTPUT_BYTES + log


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_git(repo: Path, runs: list[PlannedRun]) -> Check:
    import src.experiments.run_federated_cifar100  # noqa: F401  (load what the runs will execute)
    import src.rev.run_rev_experiment  # noqa: F401
    import src.rev.compare_runs  # noqa: F401
    from src.rev.provenance import git_state

    used = [r.config for r in runs] + ["configs/cifar100_data.yaml",
                                       f"configs/{ARCHIVE}.yaml", "configs/rev/S0a_smoke_ndss.yaml",
                                       "configs/cifar100_fedavg_dirichlet_a01_seed123_50rounds.yaml"]
    state = git_state(repo, used_files=used)
    branch = subprocess.run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True,
                            text=True, check=False).stdout.strip()
    ok = state["execution_clean"] and not state["tracked_modified"]
    return Check("git", "PASS" if ok else "FAIL", {
        "commit": state["commit"], "branch": branch, "execution_clean": state["execution_clean"],
        "tracked_modified": state["tracked_modified"], "executed_uncommitted_modules": state["executed_uncommitted_modules"],
        "used_uncommitted_files": state["used_uncommitted_files"], "untracked_count": state["untracked_count"]})


def check_configs(repo: Path, runs: list[PlannedRun]) -> Check:
    from src.experiments.run_federated_cifar100 import load_federated_experiment_config
    from src.rev.config import load_rev_config

    problems = {}
    for run in runs:
        try:
            if run.kind == "ndss":
                cfg = load_federated_experiment_config(repo / run.config)
                if not str(cfg.output_dir).replace("\\", "/").startswith("results_rev/"):
                    problems[run.name] = "output_dir outside results_rev/"
            else:
                load_rev_config(repo / run.config, repo)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            problems[run.name] = f"{type(exc).__name__}: {exc}"
    return Check("configs", "FAIL" if problems else "PASS", {"problems": problems})


def output_locations(repo: Path, run: PlannedRun) -> list[str]:
    from src.experiments.run_federated_cifar100 import load_federated_experiment_config
    from src.rev.config import load_rev_config

    if run.kind == "ndss":
        cfg = load_federated_experiment_config(repo / run.config)
        return [Path(cfg.output_dir).as_posix(), Path(cfg.checkpoint_dir).as_posix()]
    cfg = load_rev_config(repo / run.config, repo)
    return [p.as_posix() for p in (cfg.output_dir, cfg.checkpoint_dir, cfg.update_log_dir) if p is not None]


def check_outputs(repo: Path, runs: list[PlannedRun]) -> Check:
    existing = {}
    for run in runs:
        found = [p for p in output_locations(repo, run) if (repo / p).exists() and any((repo / p).iterdir())]
        if found:
            existing[run.name] = found
    return Check("outputs", "FAIL" if existing else "PASS", {"already_present": existing})


def check_archive(repo: Path, preservation_root: Path) -> Check:
    missing = [f for f in ARCHIVE_FILES if not (repo / f).is_file()]
    if missing:
        return Check("archive", "FAIL", {"missing": missing})
    manifest = preservation_root / "NDSS_freeze_20261020" / "manifests" / "MANIFEST.source.pre_copy.sha256.csv"
    if not manifest.exists():
        return Check("archive", "WARN", {"reason": "preservation manifest not available; presence checked only"})
    with manifest.open(encoding="utf-8") as handle:
        expected = {r["relative_path"]: r["sha256"] for r in csv.DictReader(handle)}
    changed = [f for f in ARCHIVE_FILES if expected.get(f) != _sha(repo / f)]
    return Check("archive", "FAIL" if changed else "PASS", {"files": len(ARCHIVE_FILES), "changed": changed})


def check_dataset(repo: Path) -> Check:
    import yaml
    from torchvision.datasets import CIFAR100

    data_cfg = yaml.safe_load((repo / "configs/cifar100_data.yaml").read_text(encoding="utf-8"))
    root = (repo / data_cfg["data_root"]).resolve()
    detail = {"data_root": data_cfg["data_root"], "config_download_flag": data_cfg.get("download")}
    try:
        for train in (True, False):
            CIFAR100(str(root), train=train, download=False)
    except Exception as exc:  # noqa: BLE001
        return Check("dataset", "FAIL", {**detail, "error": type(exc).__name__,
                                         "note": "a run with download: true would try to download"})
    return Check("dataset", "PASS", {**detail, "integrity": "torchvision check passed (no download attempted)"})


def _nvidia_smi() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {}
    result = subprocess.run([exe, "--query-gpu=name,driver_version,memory.total,memory.free",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False, timeout=30)
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    name, driver, total, free = [x.strip() for x in result.stdout.strip().splitlines()[0].split(",")]
    return {"gpu_name": name, "driver_version": driver, "memory_total_mib": int(total), "memory_free_mib": int(free)}


def check_environment(repo: Path) -> Check:
    import platform

    import torch

    archive = json.loads((repo / f"results/federated/{ARCHIVE}/summary.json").read_text(encoding="utf-8"))
    final = torch.load(repo / f"checkpoints/federated/{ARCHIVE}/final_global_model.pt", map_location="cpu",
                       weights_only=False)
    current = {"python": platform.python_version(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
               "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
               **_nvidia_smi()}
    archived = {"torch": str(final.get("torch_version")), "cuda_build": final.get("cuda_version"),
                "gpu_name": archive.get("gpu_name"), "driver_version": None, "cudnn": None}
    mismatches = [k for k in ("torch", "cuda_build", "gpu_name") if current.get(k) != archived.get(k)]
    status = "FAIL" if mismatches else ("PASS" if "gpu_name" in current else "WARN")
    return Check("environment", status, {"current": current, "archived": archived, "mismatches": mismatches,
                                         "unrecorded_in_archive": ["driver_version", "cudnn"]})


def check_disk(repo: Path, runs: list[PlannedRun], reserve_gib: float) -> Check:
    need = sum(planned_bytes(r) for r in runs)
    free = shutil.disk_usage(repo).free
    ok = free - need >= reserve_gib * _GIB
    return Check("disk", "PASS" if ok else "FAIL", {
        "free_gib": round(free / _GIB, 1), "planned_gib": round(need / _GIB, 2), "reserve_gib": reserve_gib,
        "per_run_gib": {r.name: round(planned_bytes(r) / _GIB, 2) for r in runs}})


def run_preflight(repo: Path, names: list[str], reserve_gib: float = 100.0,
                  preservation_root: Path | None = None) -> list[Check]:
    runs = [PLANNED[n] for n in names]
    root = preservation_root or Path(os.environ.get("FEDTRACE_PRESERVATION_ROOT") or Path(repo.resolve().anchor) / "FedTrace_Preservation")
    checks = [check_git(repo, runs), check_configs(repo, runs)]
    checks.append(check_outputs(repo, runs) if checks[1].status == "PASS" else Check("outputs", "FAIL", {"reason": "configs invalid"}))
    checks += [check_archive(repo, root), check_dataset(repo), check_environment(repo), check_disk(repo, runs, reserve_gib)]
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU-only Stage B preflight.")
    parser.add_argument("--runs", nargs="+", default=list(STAGE_B), choices=list(PLANNED))
    parser.add_argument("--reserve-gib", type=float, default=100.0)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    repo = Path.cwd()
    checks = run_preflight(repo, args.runs, args.reserve_gib)
    for c in checks:
        print(f"{c.name:12s} {c.status:5s} {json.dumps(c.detail, default=str)}")
    gpu_hours = sum(PLANNED[n].gpu_hours for n in args.runs)
    print(f"planned runs: {args.runs}; estimated GPU time {gpu_hours:.2f} h (measured 1.33 h per 50-round K=5 run)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps([asdict(c) for c in checks], indent=2, default=str) + "\n", encoding="utf-8")
    return 0 if all(c.status in ("PASS", "WARN") for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
