"""The promotion rule must accept only candidates that pass every gate."""

from __future__ import annotations

import math

import pytest

from ml.promote import GATES_PATH, load_gates, should_promote

GATES = {
    "primary_metric": "f1_spam",
    "greater_is_better": True,
    "quality_floor": 0.85,
    "min_improvement": 0.005,
    "guardrails": {"precision_spam_min": 0.90},
}
GOOD_PRECISION = {"precision_spam": 0.95}


def test_committed_gates_file_is_valid():
    gates = load_gates(GATES_PATH)
    assert gates["primary_metric"] == "f1_spam"
    assert gates["greater_is_better"] is True


# --- first release: no champion yet -------------------------------------------


def test_first_candidate_passing_gates_is_promoted():
    promote, reason = should_promote(0.91, None, GATES, GOOD_PRECISION)
    assert promote
    assert "no champion" in reason


@pytest.mark.parametrize("score", [0.0, 0.80, 0.8499])
def test_first_candidate_below_floor_is_rejected(score):
    promote, reason = should_promote(score, None, GATES, GOOD_PRECISION)
    assert not promote
    assert "quality floor" in reason


def test_score_exactly_at_floor_passes():
    promote, _ = should_promote(0.85, None, GATES, GOOD_PRECISION)
    assert promote


# --- with a champion -----------------------------------------------------------


@pytest.mark.parametrize(
    ("candidate", "champion"),
    [(0.93, 0.91), (0.915, 0.91)],  # clear win, and an improvement of exactly 0.005
)
def test_candidate_improving_enough_is_promoted(candidate, champion):
    promote, reason = should_promote(candidate, champion, GATES, GOOD_PRECISION)
    assert promote
    assert "improves" in reason


@pytest.mark.parametrize(
    ("candidate", "champion"),
    [
        (0.91, 0.91),  # exact tie
        (0.914, 0.91),  # improves, but by less than min_improvement
        (0.90, 0.91),  # worse than the champion
    ],
)
def test_candidate_not_improving_enough_is_rejected(candidate, champion):
    promote, reason = should_promote(candidate, champion, GATES, GOOD_PRECISION)
    assert not promote
    assert "does not improve" in reason


def test_floor_applies_even_when_champion_is_worse():
    promote, reason = should_promote(0.84, 0.70, GATES, GOOD_PRECISION)
    assert not promote
    assert "quality floor" in reason


# --- guardrails ----------------------------------------------------------------


def test_low_precision_is_rejected_whatever_the_f1():
    promote, reason = should_promote(0.99, 0.91, GATES, {"precision_spam": 0.89})
    assert not promote
    assert "guardrail" in reason


def test_missing_guardrail_metric_is_rejected():
    promote, reason = should_promote(0.95, None, GATES, {})
    assert not promote
    assert "cannot be checked" in reason


# --- invalid inputs: anything unverifiable keeps the champion ------------------


@pytest.mark.parametrize("bad", [float("nan"), math.inf, None, "0.95", True])
def test_invalid_candidate_score_is_rejected(bad):
    promote, _ = should_promote(bad, 0.91, GATES, GOOD_PRECISION)
    assert not promote


def test_invalid_champion_score_is_rejected():
    promote, reason = should_promote(0.95, float("nan"), GATES, GOOD_PRECISION)
    assert not promote
    assert "champion" in reason


# --- lower-is-better metrics -----------------------------------------------------


def test_direction_is_read_from_the_gates():
    gates = {
        "primary_metric": "log_loss",
        "greater_is_better": False,
        "quality_floor": 0.30,
        "min_improvement": 0.01,
    }
    assert should_promote(0.20, 0.25, gates)[0]  # lower loss, improved by 0.05
    assert not should_promote(0.245, 0.25, gates)[0]  # improved by only 0.005
    assert not should_promote(0.35, None, gates)[0]  # above the floor (worse)


# --- gates file validation -------------------------------------------------------


def test_incomplete_gates_file_is_refused(tmp_path):
    path = tmp_path / "gates.yaml"
    path.write_text("primary_metric: f1_spam\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_gates(path)


def test_negative_min_improvement_is_refused(tmp_path):
    path = tmp_path / "gates.yaml"
    path.write_text(
        "primary_metric: f1_spam\ngreater_is_better: true\n"
        "quality_floor: 0.85\nmin_improvement: -0.1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="negative"):
        load_gates(path)
