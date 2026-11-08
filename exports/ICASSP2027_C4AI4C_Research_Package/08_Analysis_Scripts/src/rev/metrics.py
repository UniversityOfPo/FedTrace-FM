"""Run-level forensic outcomes and seed-level statistics.

Statistical unit: one independent run (one seed). Client-rounds are never treated
as independent samples: they appear only as a secondary, pooled rate next to the
per-run counts. Misses are never dropped: every attacked run contributes to the
detection and attribution denominators, onset error is reported together with
detected / missed / wrongly-attributed counts, and the delay of a missed run is
right-censored at the last observed round.

PROVISIONAL thresholds (``ProvisionalThresholds``) are placeholders pending PI
pre-registration; they are recorded with every summary that uses them.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from scipy.stats import beta
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.rev.online import OnlineDetection


@dataclass(frozen=True)
class ProvisionalThresholds:
    attack_success_min: float = 0.5
    kappa: float = 2.5
    provisional: bool = True


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float] | None:
    if n <= 0:
        return None
    if not 0 <= k <= n:
        raise ValueError("require 0 <= k <= n")
    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lower, upper


@dataclass(frozen=True)
class Proportion:
    k: int
    n: int
    rate: float | None
    ci95: tuple[float, float] | None

    @classmethod
    def of(cls, k: int, n: int) -> Proportion:
        return cls(k, n, k / n if n else None, clopper_pearson(k, n))


# ----------------------------------------------------------------------
# Attack effect
# ----------------------------------------------------------------------
def attack_success_counts(predictions: Sequence[int], target_class: int) -> Proportion:
    """Predictions on triggered non-target-class samples; success = predicted target class."""
    k = sum(1 for p in predictions if int(p) == target_class)
    return Proportion.of(k, len(predictions))


@torch.no_grad()
def predict(model: nn.Module, dataset: Dataset, device: torch.device | str = "cpu", batch_size: int = 256) -> list[int]:
    """Deterministic predictions (no shuffle, no workers, eval mode, global RNG untouched).

    A ``DataLoader`` draws a base seed from its generator on every iteration even
    without shuffling; a private generator keeps the global torch RNG unchanged.
    """
    was_training = model.training
    model.eval()
    out: list[int] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(0))
    for images, _ in loader:
        out.extend(int(p) for p in model(images.to(device)).argmax(dim=1).cpu())
    model.train(was_training)
    return out


@dataclass(frozen=True)
class PairedImpact:
    per_seed: dict[int, float]
    mean: float | None
    n_pairs: int
    unpaired_attacked: tuple[int, ...]
    unpaired_clean: tuple[int, ...]


def paired_impact(attacked: Mapping[int, float], clean: Mapping[int, float]) -> PairedImpact:
    """Per-seed attacked - clean differences; seeds without a matched control are listed, not dropped silently."""
    seeds = sorted(set(attacked) & set(clean))
    diffs = {s: float(attacked[s]) - float(clean[s]) for s in seeds}
    return PairedImpact(diffs, statistics.mean(diffs.values()) if diffs else None, len(diffs),
                        tuple(sorted(set(attacked) - set(clean))), tuple(sorted(set(clean) - set(attacked))))


# ----------------------------------------------------------------------
# Forensic outcomes of one run
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    seed: int
    malicious_client: str | None
    true_onset: int | None
    last_round: int
    detected: bool | None
    attributed_client: str | None
    attribution_correct: bool | None
    alarm_round: int | None
    actionable_round: int | None
    estimated_onset: int | None
    onset_abs_error: int | None
    delay: int | None
    delay_censored: bool | None
    pre_onset_attacker_alarm: bool | None
    honest_alarm_clients: tuple[str, ...]
    honest_alarm_client_rounds: int
    eligible_honest_client_rounds: int
    retrospective_onset: int | None


def run_outcome(run_id: str, seed: int, detection: OnlineDetection, malicious_client: str | None,
                true_onset: int | None, per_round_alarm_state: Mapping[str, Sequence[int]] | None = None) -> RunOutcome:
    """Classify one run from its online detection.

    ``detected`` = the malicious client raised an alarm at or after the true onset.
    ``attribution_correct`` = the attributed (earliest-alarm) client is the malicious one.
    An alarm on the malicious client before the true onset is a false alarm
    (``pre_onset_attacker_alarm``) and does not count as detection.
    ``per_round_alarm_state``: optional client -> rounds in which the client was in an
    alarm state (for the client-round rate); defaults to the alarm round only.
    """
    cfg = detection.config
    eligible_rounds = [r for r in range(cfg.earliest_scoring_round, detection.last_round + 1)]
    honest = [c for c in detection.per_client if c != malicious_client]
    honest_alarms = tuple(sorted(c for c in honest if detection.per_client[c].alarm_round is not None))
    alarm_rounds = {
        c: (list(per_round_alarm_state.get(c, [])) if per_round_alarm_state is not None
            else ([detection.per_client[c].alarm_round] if detection.per_client[c].alarm_round is not None else []))
        for c in honest
    }
    base = dict(
        run_id=run_id, seed=seed, malicious_client=malicious_client, true_onset=true_onset,
        last_round=detection.last_round, attributed_client=detection.attributed_client,
        honest_alarm_clients=honest_alarms,
        honest_alarm_client_rounds=sum(len(v) for v in alarm_rounds.values()),
        eligible_honest_client_rounds=len(honest) * len(eligible_rounds),
    )
    if malicious_client is None:
        return RunOutcome(**base, detected=None, attribution_correct=None, alarm_round=None, actionable_round=None,
                          estimated_onset=None, onset_abs_error=None, delay=None, delay_censored=None,
                          pre_onset_attacker_alarm=None, retrospective_onset=None)
    if true_onset is None:
        raise ValueError("true_onset is required for an attacked run")
    events = detection.per_client[malicious_client]
    pre_onset = events.alarm_round is not None and events.alarm_round < true_onset
    detected = events.alarm_round is not None and not pre_onset
    correct = detection.attributed_client == malicious_client and detected
    return RunOutcome(
        **base,
        detected=detected,
        attribution_correct=correct,
        alarm_round=events.alarm_round if detected else None,
        actionable_round=events.actionable_round if detected else None,
        estimated_onset=events.estimated_onset if detected else None,
        onset_abs_error=abs(events.estimated_onset - true_onset) if detected else None,
        delay=(events.alarm_round - true_onset) if detected else (detection.last_round - true_onset),
        delay_censored=not detected,
        pre_onset_attacker_alarm=pre_onset,
        retrospective_onset=detection.retrospective_onset.get(malicious_client),
    )


# ----------------------------------------------------------------------
# Seed-level summary
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class OutcomeSummary:
    n_runs: int
    detection: Proportion | None
    attribution: Proportion | None
    n_missed: int
    n_wrong_attribution: int
    onset_mae: float | None
    onset_mae_n: int
    onset_errors: tuple[int, ...]
    delays_observed: tuple[int, ...]
    delays_censored_at_least: tuple[int, ...]
    runs_with_honest_alarm: Proportion
    honest_alarms_per_1000_client_rounds: float | None
    per_run_honest_alarm_clients: dict[str, int]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize(outcomes: Sequence[RunOutcome]) -> OutcomeSummary:
    """Summarise runs of ONE condition (all attacked, or all clean); one run per seed."""
    seeds = [o.seed for o in outcomes]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds in one condition (pseudo-replication)")
    attacked = [o for o in outcomes if o.malicious_client is not None]
    if attacked and len(attacked) != len(outcomes):
        raise ValueError("summarize one condition at a time (mixed clean and attacked runs)")
    notes = []
    detection = attribution = None
    missed = wrong = 0
    errors: list[int] = []
    observed: list[int] = []
    censored: list[int] = []
    if attacked:
        detection = Proportion.of(sum(o.detected for o in attacked), len(attacked))
        attribution = Proportion.of(sum(o.attribution_correct for o in attacked), len(attacked))
        missed = sum(not o.detected for o in attacked)
        wrong = sum(o.detected and not o.attribution_correct for o in attacked)
        errors = [o.onset_abs_error for o in attacked if o.detected]
        observed = [o.delay for o in attacked if o.detected]
        censored = [o.delay for o in attacked if not o.detected]
        if missed:
            notes.append(f"{missed} missed run(s): onset error undefined for them; delay censored")
    eligible = sum(o.eligible_honest_client_rounds for o in outcomes)
    alarms = sum(o.honest_alarm_client_rounds for o in outcomes)
    if outcomes:
        notes.append("client-round alarm rate pools client-rounds and is secondary to the per-run counts")
    return OutcomeSummary(
        n_runs=len(outcomes),
        detection=detection,
        attribution=attribution,
        n_missed=missed,
        n_wrong_attribution=wrong,
        onset_mae=statistics.mean(errors) if errors else None,
        onset_mae_n=len(errors),
        onset_errors=tuple(errors),
        delays_observed=tuple(observed),
        delays_censored_at_least=tuple(censored),
        runs_with_honest_alarm=Proportion.of(sum(bool(o.honest_alarm_clients) for o in outcomes), len(outcomes)),
        honest_alarms_per_1000_client_rounds=(1000.0 * alarms / eligible) if eligible else None,
        per_run_honest_alarm_clients={o.run_id: len(o.honest_alarm_clients) for o in outcomes},
        notes=tuple(notes),
    )
