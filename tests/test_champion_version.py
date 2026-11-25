"""champion_version must print the version behind the alias, and fail clearly otherwise."""

from __future__ import annotations

import pytest

from scripts import champion_version
from scripts.make_test_registry import build

CONFIG_VARS = ("MLFLOW_TRACKING_URI", "MODEL_NAME", "MODEL_ALIAS")


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    return build(tmp_path_factory.mktemp("registry"))


@pytest.fixture
def env(registry, monkeypatch):
    for name in CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", registry["tracking_uri"])
    monkeypatch.setenv("MODEL_NAME", registry["model_name"])
    return monkeypatch


def test_prints_the_champion_version(env, capsys):
    assert champion_version.main() == 0
    assert capsys.readouterr().out.strip() == "1"  # not the newer candidate, version 2


def test_missing_alias_exits_1_with_a_message(env, capsys):
    env.setenv("MODEL_ALIAS", "no-such-alias")

    assert champion_version.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "'no-such-alias'" in captured.err


def test_missing_configuration_exits_1(env, capsys):
    env.delenv("MODEL_NAME")

    assert champion_version.main() == 1
    assert "MODEL_NAME" in capsys.readouterr().err
