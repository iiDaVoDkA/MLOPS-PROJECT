from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from app.main import create_app
from scripts.make_test_registry import build

PRIVATE_TEXT = "Hi Sam, the door code is 4417, see you at the flat tonight"
METRIC_NAMES = (
    "http_requests_total",
    "http_request_duration_seconds",
    "predictions_total",
    "model_info",
    "spam_prediction_share",
    "input_length_mean",
)
CONFIG_VARS = ("MLFLOW_TRACKING_URI", "MODEL_NAME", "MODEL_ALIAS")


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    return build(tmp_path_factory.mktemp("registry"))


def start_app(monkeypatch, **env: str) -> TestClient:
    for name in CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return TestClient(create_app(load_in_background=False))


@pytest.fixture
def client(registry, monkeypatch):
    with start_app(
        monkeypatch,
        MLFLOW_TRACKING_URI=registry["tracking_uri"],
        MODEL_NAME=registry["model_name"],
        MODEL_ALIAS="champion",
    ) as test_client:
        yield test_client


def samples(client, metric_name):
    families = text_string_to_metric_families(client.get("/metrics").text)
    return [
        sample for family in families for sample in family.samples if sample.name == metric_name
    ]


def predictions_served(client, model_version):
    return sum(
        sample.value
        for sample in samples(client, "predictions_total")
        if sample.labels["model_version"] == model_version
    )


def log_lines(output):
    return [json.loads(line) for line in output.splitlines() if line.strip()]


def test_metrics_lists_every_metric_after_a_prediction(client):
    client.post("/predict", json={"text": "Are we still meeting for lunch tomorrow?"})

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    for name in METRIC_NAMES:
        assert name in response.text
    assert samples(client, "model_info")[0].labels == {"model_version": "1"}


def test_prediction_counter_increases(client):
    before = predictions_served(client, "1")

    client.post("/predict", json={"text": "WINNER! Text WIN to 80086 to claim your prize"})

    assert predictions_served(client, "1") == before + 1


def test_metrics_scrapes_are_not_counted(client):
    client.get("/metrics")

    paths = {sample.labels["path"] for sample in samples(client, "http_requests_total")}
    assert "/metrics" not in paths


def test_request_id_is_echoed_when_sent(client):
    response = client.get("/health/live", headers={"X-Request-ID": "demo-42"})

    assert response.headers["X-Request-ID"] == "demo-42"


def test_request_id_is_generated_when_missing(client):
    first = client.get("/health/live").headers["X-Request-ID"]
    second = client.get("/health/live").headers["X-Request-ID"]

    assert len(first) == 12
    int(first, 16)
    assert first != second


def test_message_text_is_never_logged(capsys, client):
    client.post("/predict", json={"text": PRIVATE_TEXT}, headers={"X-Request-ID": "trace-1"})
    client.post("/predict", json={"text": PRIVATE_TEXT, "extra": 1})

    output = capsys.readouterr().out
    lines = log_lines(output)

    assert PRIVATE_TEXT not in output
    assert "4417" not in output
    prediction = next(line for line in lines if line["event"] == "prediction_made")
    assert prediction["request_id"] == "trace-1"
    assert prediction["text_length"] == len(PRIVATE_TEXT)
    assert prediction["model_version"] == "1"
    assert {"prediction", "spam_probability", "latency_ms", "level", "timestamp"} <= set(prediction)
    statuses = [line["status"] for line in lines if line["event"] == "request"]
    assert statuses == [200, 422]


def test_startup_logs_the_loaded_model(capsys, registry, monkeypatch):
    with start_app(
        monkeypatch,
        MLFLOW_TRACKING_URI=registry["tracking_uri"],
        MODEL_NAME=registry["model_name"],
    ):
        pass

    loaded = next(
        line for line in log_lines(capsys.readouterr().out) if line["event"] == "model_loaded"
    )
    assert loaded["model_name"] == registry["model_name"]
    assert loaded["model_version"] == "1"
    assert loaded["dataset_version"] == "ci-fixture"


def test_server_errors_are_logged_as_errors(capsys, registry, monkeypatch):
    with start_app(
        monkeypatch, MLFLOW_TRACKING_URI=registry["tracking_uri"], MODEL_NAME="no-such-model"
    ) as test_client:
        assert test_client.post("/predict", json={"text": "hello"}).status_code == 503

    request = next(
        line for line in log_lines(capsys.readouterr().out) if line["event"] == "request"
    )
    assert request["status"] == 503
    assert request["level"] == "error"
    assert request["path"] == "/predict"
