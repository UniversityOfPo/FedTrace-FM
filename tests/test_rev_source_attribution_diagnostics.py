"""Post-hoc D1 mechanism checks (L-74): synthetic data only."""

from __future__ import annotations

import pytest

from src.rev import source_attribution as sa
from src.rev.source_attribution_diagnostics import leakage
from tests.test_rev_source_attribution import BASE, trace


def test_leakage_decomposition_sums_to_influence() -> None:
    rows = trace(BASE, change={"client_002": (20, 0.60)})
    lk = leakage(rows, "client_002")
    scores = sa.analyse(rows, ("MED",))["MED"]["B"]["at_eval_scores"]
    for j, v in lk.items():
        assert v["I"] == pytest.approx(v["from_named_client_term"] + v["from_other_terms"])
        assert v["I"] == pytest.approx(scores[j])
    assert lk["client_002"]["from_named_client_term"] == 0.0  # own term never enters its own influence
