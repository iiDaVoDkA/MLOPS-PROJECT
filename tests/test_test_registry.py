"""The CI test registry must hold an approved champion and a newer, unapproved version."""

from __future__ import annotations

import json

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from scripts.make_test_registry import CHAMPION_ALIAS, DEFAULT_MODEL_NAME, build


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    root = tmp_path_factory.mktemp("ci-registry")
    summary = build(root)
    return root, summary


def test_champion_is_version_1_and_a_newer_version_exists(registry):
    root, summary = registry
    assert summary["champion_version"] == "1"
    assert summary["newer_candidate_version"] == "2"
    assert json.loads((root / "registry.json").read_text()) == summary

    uri = summary["tracking_uri"]
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    champion = client.get_model_version_by_alias(DEFAULT_MODEL_NAME, CHAMPION_ALIAS)
    assert str(champion.version) == "1"
    assert champion.tags["approved"] == "true"

    newer = client.get_model_version(DEFAULT_MODEL_NAME, "2")
    assert newer.tags["approved"] == "false"
    assert not newer.aliases


def test_champion_predicts_both_labels(registry):
    _, summary = registry
    mlflow.set_tracking_uri(summary["tracking_uri"])
    mlflow.set_registry_uri(summary["tracking_uri"])
    model = mlflow.pyfunc.load_model(f"models:/{DEFAULT_MODEL_NAME}/1")
    frame = pd.DataFrame(
        {"text": ["Are we still meeting for lunch tomorrow?", "WINNER! Text WIN to 80086 now."]}
    )
    labels = [str(label) for label in model.predict(frame)]
    assert set(labels) <= {"ham", "spam"}
    assert labels == ["ham", "spam"]


def test_refuses_to_overwrite_without_force(registry):
    root, _ = registry
    with pytest.raises(SystemExit):
        build(root)
