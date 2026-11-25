from __future__ import annotations

import json

import pytest
from mlflow import MlflowClient

from ml.pipeline import PipelineError, main, move_champion, needs_retraining
from ml.promote import load_gates
from scripts.make_test_registry import build

GATES = load_gates()


@pytest.mark.parametrize(
    ("dataset_version", "tags", "score", "expected"),
    [
        ("v2", {"dataset_version": "v1", "f1_spam": "0.9129"}, 0.9129, True),
        ("v1", {"dataset_version": "v1", "f1_spam": "0.9129"}, 0.80, True),
        ("v1", {"dataset_version": "v1", "f1_spam": "0.9500"}, 0.9100, True),
        ("v1", {"dataset_version": "v1", "f1_spam": "0.9400"}, 0.9100, True),
        ("v1", {"dataset_version": "v1", "f1_spam": "0.9300"}, 0.9100, False),
        ("v1", {"dataset_version": "v1", "f1_spam": "0.9129"}, 0.9129, False),
        ("v1", {"dataset_version": "v2", "f1_spam": "0.9129"}, 0.9129, False),
        ("v1", {"dataset_version": "v1"}, 0.9129, False),
        ("v2", {}, 0.9129, True),
    ],
    ids=[
        "new-data",
        "below-floor",
        "drop-over-limit",
        "drop-at-limit",
        "drop-under-limit",
        "healthy-same-data",
        "older-data",
        "no-approved-score",
        "no-dataset-tag",
    ],
)
def test_retrain_rule(dataset_version, tags, score, expected):
    retrain, reason = needs_retraining(dataset_version, tags, score, GATES)

    assert retrain is expected
    assert reason


@pytest.fixture
def registry(tmp_path, monkeypatch):
    summary = build(tmp_path / "registry")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", summary["tracking_uri"])
    monkeypatch.delenv("MODEL_ALIAS", raising=False)
    return summary


@pytest.fixture
def client(registry):
    return MlflowClient(registry["tracking_uri"])


def run(registry, tmp_path, *args):
    results = tmp_path / "results"
    code = main([*args, "--model-name", registry["model_name"], "--results-dir", str(results)])
    return code, json.loads((results / "decision.json").read_text())


def champion_of(client, registry):
    return str(client.get_model_version_by_alias(registry["model_name"], "champion").version)


def test_rollback_to_an_approved_version(registry, client, tmp_path):
    name = registry["model_name"]
    client.set_model_version_tag(name, "2", "approved", "true")
    client.set_registered_model_alias(name, "champion", "2")

    code, decision = run(registry, tmp_path, "--rollback", "1")

    assert code == 0
    assert decision["decision"] == "rolled_back"
    assert decision["champion_before"] == "2"
    assert decision["champion_after"] == "1"
    assert decision["validation"] == "skipped"
    assert champion_of(client, registry) == "1"


def test_rollback_to_the_current_champion_changes_nothing(registry, client, tmp_path):
    code, decision = run(registry, tmp_path, "--rollback", "1")

    assert code == 0
    assert decision["decision"] == "rollback_unchanged"
    assert champion_of(client, registry) == "1"


@pytest.mark.parametrize(("target", "message"), [("2", "never approved"), ("99", "no version 99")])
def test_rollback_refused_without_moving_the_alias(registry, client, tmp_path, target, message):
    code, decision = run(registry, tmp_path, "--rollback", target)

    assert code == 1
    assert decision["decision"] == "failed"
    assert message in decision["reason"]
    assert decision["champion_before"] == decision["champion_after"] == "1"
    assert champion_of(client, registry) == "1"


def test_decision_is_written_when_validation_fails(registry, client, tmp_path):
    empty_data_root = tmp_path / "data"
    empty_data_root.mkdir()

    code, decision = run(
        registry, tmp_path, "--dataset-version", "v9", "--data-root", str(empty_data_root)
    )

    assert code == 1
    assert decision["decision"] == "failed"
    assert decision["validation"] == "failed"
    assert decision["retrained"] is False
    assert decision["dataset_version"] == "v9"
    assert decision["champion_before"] == decision["champion_after"] == "1"
    assert champion_of(client, registry) == "1"


def test_unapproved_champion_is_refused(registry, client, tmp_path):
    client.set_registered_model_alias(registry["model_name"], "champion", "2")

    code, decision = run(registry, tmp_path, "--dataset-version", "v1")

    assert code == 1
    assert decision["decision"] == "failed"
    assert "not approved" in decision["reason"]
    assert decision["validation"] == "not_run"
    assert champion_of(client, registry) == "2"


def test_move_champion_refuses_if_the_champion_changed(registry, client):
    with pytest.raises(PipelineError, match="changed during the run"):
        move_champion(client, registry["model_name"], "champion", "2", expected_version="2")

    assert champion_of(client, registry) == "1"


def test_decision_is_written_on_an_unexpected_error(registry, tmp_path, monkeypatch):
    def crash(*_):
        raise ZeroDivisionError("boom")

    monkeypatch.setattr("ml.pipeline.roll_back", crash)

    with pytest.raises(ZeroDivisionError):
        main(["--rollback", "1", "--results-dir", str(tmp_path / "results")])

    decision = json.loads((tmp_path / "results" / "decision.json").read_text())
    assert decision["decision"] == "failed"
    assert decision["champion_after"] == "1"


def test_dataset_version_and_rollback_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--dataset-version", "v2", "--rollback", "1", "--results-dir", str(tmp_path)])

    assert error.value.code == 2
