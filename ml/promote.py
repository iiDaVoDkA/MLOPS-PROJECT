"""Promotion rule: decide whether a candidate model replaces the champion.

The thresholds live in ``ml/gates.yaml`` and are fixed before any candidate is
evaluated. ``should_promote`` is a pure function: it reads nothing and changes
nothing, so every outcome can be tested directly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

GATES_PATH = Path(__file__).with_name("gates.yaml")

# Absorbs floating-point noise so that an improvement of exactly
# ``min_improvement`` (e.g. 0.905 vs 0.900) counts as reaching the threshold.
_TOLERANCE = 1e-9


def load_gates(path: Path = GATES_PATH) -> dict[str, Any]:
    gates = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = ("primary_metric", "greater_is_better", "quality_floor", "min_improvement")
    missing = [key for key in required if key not in (gates or {})]
    if missing:
        raise ValueError(f"{path}: missing gate(s) {missing}")
    if gates["min_improvement"] < 0:
        raise ValueError(f"{path}: min_improvement must not be negative")
    return gates


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_guardrails(guardrails: Mapping[str, float], metrics: Mapping[str, float]) -> str | None:
    """Return the first guardrail violation, or None when all of them hold."""
    for name, bound in guardrails.items():
        metric, _, kind = name.rpartition("_")
        if kind not in ("min", "max") or not metric:
            return f"guardrail {name} is not of the form <metric>_min or <metric>_max"
        value = metrics.get(metric)
        if not _is_number(value):
            return f"guardrail {name} cannot be checked: candidate has no valid {metric}"
        if kind == "min" and value < bound:
            return f"candidate {metric} {value:.4f} is below the guardrail minimum {bound}"
        if kind == "max" and value > bound:
            return f"candidate {metric} {value:.4f} is above the guardrail maximum {bound}"
    return None


def should_promote(
    candidate_score: float,
    champion_score: float | None,
    gates: Mapping[str, Any],
    candidate_metrics: Mapping[str, float] | None = None,
) -> tuple[bool, str]:
    """Decide whether the candidate becomes the champion.

    ``candidate_score`` and ``champion_score`` are the primary metric of each
    model on the same held-out observations. ``champion_score`` is None when no
    champion exists yet. ``candidate_metrics`` holds the candidate's other
    metrics, needed only when the gates define guardrails. Anything that cannot
    be verified is refused: the champion is retained.
    """
    metric = gates["primary_metric"]
    higher = bool(gates["greater_is_better"])
    floor = gates["quality_floor"]
    min_improvement = gates["min_improvement"]

    if not _is_number(candidate_score):
        return False, f"candidate {metric} is not a valid number"

    below_floor = candidate_score < floor if higher else candidate_score > floor
    if below_floor:
        side = "below" if higher else "above"
        return (
            False,
            f"candidate {metric} {candidate_score:.4f} is {side} the quality floor {floor}",
        )

    guardrails = gates.get("guardrails") or {}
    if guardrails:
        metrics = {metric: candidate_score, **(candidate_metrics or {})}
        violation = _check_guardrails(guardrails, metrics)
        if violation:
            return False, violation

    if champion_score is None:
        return (
            True,
            f"no champion yet and candidate {metric} {candidate_score:.4f} passes the gates",
        )

    if not _is_number(champion_score):
        return False, f"champion {metric} is not a valid number"

    improvement = candidate_score - champion_score if higher else champion_score - candidate_score
    if improvement + _TOLERANCE < min_improvement:
        return False, (
            f"candidate {metric} {candidate_score:.4f} does not improve on the champion "
            f"{champion_score:.4f} by at least {min_improvement} (improvement {improvement:+.4f})"
        )

    return True, (
        f"candidate {metric} {candidate_score:.4f} improves on the champion "
        f"{champion_score:.4f} by {improvement:+.4f} (minimum {min_improvement})"
    )
