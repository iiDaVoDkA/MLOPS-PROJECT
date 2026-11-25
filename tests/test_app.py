"""Application tests, run offline against a throwaway MLflow registry.

The registry (scripts/make_test_registry.py) holds:
    version 1  approved, alias ``champion``  -> what the app must serve
    version 2  newer, not approved, no alias -> what the app must ignore
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from app.main import create_app
from app.schemas import MAX_TEXT_LENGTH
from scripts.make_test_registry import build

HAM = "Are we still meeting for lunch tomorrow?"
SPAM = "WINNER! You have won a free prize. Text WIN to 80086 to claim now."
CONFIG_VARS = ("MLFLOW_TRACKING_URI", "MODEL_NAME", "MODEL_ALIAS")


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    return build(tmp_path_factory.mktemp("registry"))


def make_client(monkeypatch, **env: str):
    """Start the app with exactly the given configuration and load the model first."""
    for name in CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return TestClient(create_app(load_in_background=False))


@pytest.fixture
def client(registry, monkeypatch):
    with make_client(
        monkeypatch,
        MLFLOW_TRACKING_URI=registry["tracking_uri"],
        MODEL_NAME=registry["model_name"],
        MODEL_ALIAS="champion",
    ) as c:
        yield c


# 1. Predictions and their output format ------------------------------------------------


@pytest.mark.parametrize(("text", "expected"), [(HAM, "ham"), (SPAM, "spam")])
def test_predict_returns_label_probability_and_version(client, registry, text, expected):
    response = client.post("/predict", json={"text": text})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "spam_probability", "model_version"}
    assert body["prediction"] == expected
    assert 0.0 <= body["spam_probability"] <= 1.0
    assert (body["spam_probability"] > 0.5) == (expected == "spam")
    assert body["model_version"] == registry["champion_version"] == "1"


@pytest.mark.parametrize(
    "text",
    [
        "x",  # shortest valid message
        "a" * MAX_TEXT_LENGTH,  # longest valid message
        "  " + HAM + "  ",  # surrounding whitespace is trimmed, not rejected
        "!!! ??? 😀 — ünïcödé",  # no word the model knows
    ],
    ids=["one-char", "max-length", "padded", "unknown-tokens"],
)
def test_predict_accepts_edge_inputs(client, text):
    response = client.post("/predict", json={"text": text})

    assert response.status_code == 200
    assert response.json()["prediction"] in {"ham", "spam"}
    assert response.json()["model_version"] == "1"


# 2. Malformed or out-of-range inputs: always 422, never 500 ----------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"json": {}},
        {"json": {"text": ""}},
        {"json": {"text": "   \n\t "}},
        {"json": {"text": "a" * (MAX_TEXT_LENGTH + 1)}},
        {"json": {"text": 12345}},
        {"json": {"text": None}},
        {"json": {"text": ["a list"]}},
        {"json": {"text": "hello", "unexpected": 1}},
        {"json": ["hello"]},
        {"content": b"", "headers": {"content-type": "application/json"}},
        {"content": b"{not json", "headers": {"content-type": "application/json"}},
    ],
    ids=[
        "missing-text",
        "empty-string",
        "whitespace-only",
        "over-max-length",
        "number",
        "null",
        "list",
        "extra-field",
        "array-body",
        "empty-body",
        "invalid-json",
    ],
)
def test_bad_input_is_rejected_with_422(client, kwargs):
    response = client.post("/predict", **kwargs)

    assert response.status_code == 422
    assert "detail" in response.json()


# 3. The approved model is loaded even though a newer candidate exists ------------------


def test_model_info_reports_champion_not_newer_candidate(client, registry):
    assert registry["newer_candidate_version"] == "2"  # the trap really exists

    response = client.get("/model-info")

    assert response.status_code == 200
    info = response.json()
    assert set(info) == {
        "name",
        "alias",
        "version",
        "run_id",
        "dataset_version",
        "git_sha",
        "loaded_at",
    }
    assert info["name"] == registry["model_name"]
    assert info["alias"] == "champion"
    assert info["version"] == "1"
    assert info["dataset_version"] == "ci-fixture"
    assert info["git_sha"] == "ci-fixture"
    assert info["run_id"]
    assert info["loaded_at"]

    # The run behind the served version is the champion's, not the candidate's.
    champion = MlflowClient(registry["tracking_uri"]).get_model_version(
        registry["model_name"], registry["champion_version"]
    )
    assert info["run_id"] == champion.run_id


def test_alias_is_resolved_at_startup_not_hard_coded(registry, monkeypatch):
    """Moving the alias to version 2 makes a fresh app serve version 2."""
    registry_client = MlflowClient(registry["tracking_uri"])
    name = registry["model_name"]
    registry_client.set_registered_model_alias(name, "champion", "2")
    try:
        with make_client(
            monkeypatch,
            MLFLOW_TRACKING_URI=registry["tracking_uri"],
            MODEL_NAME=name,
        ) as c:  # MODEL_ALIAS unset: defaults to "champion"
            assert c.get("/model-info").json()["version"] == "2"
            assert c.post("/predict", json={"text": HAM}).json()["model_version"] == "2"
    finally:
        registry_client.set_registered_model_alias(name, "champion", "1")


# 4. Liveness and readiness ------------------------------------------------------------


def test_ready_once_the_model_has_predicted(client):
    assert client.get("/health/live").status_code == 200

    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model_version": "1"}


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"MODEL_NAME": "sms-spam-classifier"},
        {"MLFLOW_TRACKING_URI": "REGISTRY", "MODEL_NAME": "no-such-model"},
        {"MLFLOW_TRACKING_URI": "REGISTRY", "MODEL_NAME": "MODEL", "MODEL_ALIAS": "no-such-alias"},
    ],
    ids=["no-config", "no-tracking-uri", "unknown-model", "unknown-alias"],
)
def test_not_ready_when_the_model_cannot_be_loaded(registry, monkeypatch, env):
    replacements = {"REGISTRY": registry["tracking_uri"], "MODEL": registry["model_name"]}
    env = {key: replacements.get(value, value) for key, value in env.items()}

    with make_client(monkeypatch, **env) as c:
        assert c.get("/health/live").status_code == 200

        ready = c.get("/health/ready")
        assert ready.status_code == 503
        assert ready.json()["status"] == "not_ready"
        assert ready.json()["reason"]

        # Every model-backed route answers 503, not 500, while not ready.
        assert c.get("/model-info").status_code == 503
        assert c.post("/predict", json={"text": HAM}).status_code == 503
        # Input validation still runs first.
        assert c.post("/predict", json={}).status_code == 422


def test_index_page_offers_sample_inputs(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.text.count('class="sample"') == 4
