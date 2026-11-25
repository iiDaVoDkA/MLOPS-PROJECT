"""The smoke test must pass against a healthy app and report what is wrong otherwise.

It runs here against the FastAPI TestClient, so CI covers the script without Docker.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from scripts import smoke_test
from scripts.make_test_registry import build

CONFIG_VARS = ("MLFLOW_TRACKING_URI", "MODEL_NAME", "MODEL_ALIAS", "GIT_SHA")
URL = "http://testserver"


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    return build(tmp_path_factory.mktemp("registry"))


def client_http(client: TestClient):
    def http(method, url, body):
        response = client.request(method, url, json=body)
        return response.status_code, response.json()

    return http


def start(monkeypatch, **env: str) -> TestClient:
    for name in CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return TestClient(create_app(load_in_background=False))


@pytest.fixture
def healthy(registry, monkeypatch):
    with start(
        monkeypatch,
        MLFLOW_TRACKING_URI=registry["tracking_uri"],
        MODEL_NAME=registry["model_name"],
        GIT_SHA="abc123",
    ) as client:
        yield client_http(client)


def test_passes_against_a_healthy_app(healthy):
    summary = smoke_test.run_checks(healthy, URL, expect_version="1", expect_app_sha="abc123")

    assert summary["failures"] == []
    assert summary["predictions"] == {"ham": "ham", "spam": "spam"}
    assert summary["model_version"] == "1"
    assert summary["empty_text_status"] == 422


def test_reports_wrong_model_version(healthy):
    summary = smoke_test.run_checks(healthy, URL, expect_version="2", expect_app_sha="abc123")

    failures = " ".join(summary["failures"])
    assert "model_version '1', expected '2'" in failures
    assert "/model-info: version '1', expected '2'" in failures


def old_then_new_image(new_after: int):
    """Fake live URL: ready, but serves the old image for the first ``new_after`` polls."""
    polls = {"model-info": 0}

    def http(method, url, body):
        if url.endswith("/health/ready"):
            return 200, {"status": "ready", "model_version": "1"}
        if url.endswith("/model-info"):
            polls["model-info"] += 1
            sha = "new" if polls["model-info"] > new_after else "old"
            return 200, {"version": "1", "app_git_sha": sha}
        if body == {"text": ""}:
            return 422, {"detail": "too short"}
        return 200, {"prediction": "ham", "model_version": "1"}

    return http, polls


def test_waits_until_the_new_image_serves():
    http, polls = old_then_new_image(new_after=3)
    sleeps = []

    summary = smoke_test.run_checks(
        http, URL, expect_version="1", expect_app_sha="new", timeout=60, sleep=sleeps.append
    )

    assert summary["failures"] == []
    assert summary["app_git_sha"] == "new"
    assert sleeps == [smoke_test.SHA_POLL_SECONDS] * 3  # three polls saw the old image
    assert polls["model-info"] == 5  # four waiting polls, then the identity check


def test_fails_clearly_when_the_new_image_never_serves():
    http, _ = old_then_new_image(new_after=10**9)

    summary = smoke_test.run_checks(
        http, URL, expect_app_sha="new", timeout=0, sleep=lambda _: None
    )

    assert summary["failures"] == [
        "/model-info: never served app_git_sha 'new' (last answer: app_git_sha 'old')"
    ]
    assert "predictions" in summary and summary["predictions"] == {}  # stopped before predicting


def test_reports_wrong_commit_against_the_real_app(healthy):
    summary = smoke_test.run_checks(
        healthy, URL, expect_app_sha="other", timeout=0, sleep=lambda _: None
    )

    assert len(summary["failures"]) == 1
    assert "never served app_git_sha 'other'" in summary["failures"][0]
    assert "app_git_sha 'abc123'" in summary["failures"][0]


def test_fails_when_never_ready(monkeypatch):
    with start(monkeypatch) as client:  # no configuration: never ready
        summary = smoke_test.run_checks(client_http(client), URL, timeout=0, sleep=lambda _: None)

    assert len(summary["failures"]) == 1
    assert "not ready after 0s" in summary["failures"][0]


def test_main_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(smoke_test, "run_checks", lambda *a, **k: {"failures": []})
    assert smoke_test.main(["--url", URL]) == 0

    monkeypatch.setattr(smoke_test, "run_checks", lambda *a, **k: {"failures": ["boom"]})
    assert smoke_test.main(["--url", URL]) == 1
    assert "FAILED: boom" in capsys.readouterr().err
