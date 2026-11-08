"""Write the frozen evidence manifest (sha256sum format) for the C4AI4C workshop paper (read-only on all inputs).

    python scripts/build_evidence_manifest.py paper_plan/31_evidence_manifest.sha256

Covers: the 21 archived NDSS traces (git-tracked) with their summaries; every file of the 14 rev run directories
(X* development, W* confirmation); every update-log manifest and seal; the analysis outputs cited by the paper.
"""

import glob
import hashlib
import sys
from pathlib import Path

PATTERNS = [
    "results/federated/*/client_round_trace.csv", "results/federated/*/final_test_metrics.json",
    "results_ndss/*/*/client_round_trace.csv", "results_ndss/*/*/final_test_metrics.json",
    "results_rev/X[1-4]_*/*", "results_rev/W[12]_*/*",
    "logs_rev/X[1-4]_*/update_log/manifest.jsonl", "logs_rev/X[1-4]_*/update_log/SEAL.json",
    "logs_rev/W[12]_*/update_log/manifest.jsonl", "logs_rev/W[12]_*/update_log/SEAL.json",
    "results_rev/pilot/*__vs__*/summary.json", "results_rev/pilot/*__vs__*/prefix.json",
    "results_rev/pilot/*/*__vs__*/summary.json", "results_rev/pilot/*/*__vs__*/prefix.json",
    "results_rev/pilot/phase_s/*.json", "results_rev/pilot/benign_drift/*.json",
    "results_rev/pilot/evidence_matrix/*.json", "results_rev/pilot/evidence_matrix_confirmation/*.json",
    "results_rev/pilot/contamination/*.json", "results_rev/pilot/unified_outcomes/*.json",
    "results_rev/pilot/source_attribution/*.json", "results_rev/pilot/label_shift/*.json",
]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


files = sorted({Path(f).as_posix() for pat in PATTERNS for f in glob.glob(pat) if Path(f).is_file()})
lines = [f"{sha(Path(f))} *{f}" for f in files]
Path(sys.argv[1]).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print(len(lines), "files")
