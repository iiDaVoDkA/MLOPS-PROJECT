"""Load the approved model from the MLflow Model Registry.

The model is identified only by environment variables: MLFLOW_TRACKING_URI
(with MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD, which MLflow reads
itself), MODEL_NAME and MODEL_ALIAS. Nothing here is tied to a registry host.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd

logger = logging.getLogger("app.model")

LABELS = ("ham", "spam")
POSITIVE_LABEL = "spam"
DEFAULT_ALIAS = "champion"

# A harmless message used to prove the loaded model can predict.
READINESS_SAMPLE = "Are we still meeting for lunch tomorrow?"
RETRY_SECONDS = 15


class ConfigurationError(RuntimeError):
    """The environment does not say which model to load."""


class ModelNotReady(RuntimeError):
    """No model has passed its test prediction yet."""


@dataclass(frozen=True)
class Settings:
    tracking_uri: str
    model_name: str
    model_alias: str

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        missing = [key for key in ("MLFLOW_TRACKING_URI", "MODEL_NAME") if not env.get(key)]
        if missing:
            raise ConfigurationError(f"missing environment variable(s): {', '.join(missing)}")
        return cls(
            tracking_uri=env["MLFLOW_TRACKING_URI"],
            model_name=env["MODEL_NAME"],
            model_alias=env.get("MODEL_ALIAS") or DEFAULT_ALIAS,
        )


@dataclass(frozen=True)
class LoadedModel:
    name: str
    alias: str
    version: str
    run_id: str | None
    dataset_version: str | None
    git_sha: str | None
    loaded_at: datetime
    model: Any
    raw_model: Any = None

    def predict(self, text: str) -> tuple[str, float | None]:
        """Return (label, spam probability or None) for one message."""
        frame = pd.DataFrame({"text": [text]})
        output = self.model.predict(frame)
        first = output.iloc[0] if hasattr(output, "iloc") else output[0]
        if hasattr(first, "iloc"):  # a one-column DataFrame row
            first = first.iloc[0]
        label = str(first)
        if label not in LABELS:
            raise ValueError(f"model returned an unknown label: {label!r}")
        return label, self._spam_probability(frame)

    def _spam_probability(self, frame: pd.DataFrame) -> float | None:
        """Best effort: not every model flavour can give a probability."""
        raw = self.raw_model
        if raw is None or not hasattr(raw, "predict_proba"):
            return None
        try:
            classes = [str(c) for c in raw.classes_]
            return float(raw.predict_proba(frame)[0][classes.index(POSITIVE_LABEL)])
        except Exception:
            logger.debug("Could not compute a spam probability", exc_info=True)
            return None


def resolve_and_load(settings: Settings) -> LoadedModel:
    """Resolve the alias to a concrete version and load exactly that version."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_registry_uri(settings.tracking_uri)
    client = MlflowClient()

    version = client.get_model_version_by_alias(settings.model_name, settings.model_alias)
    number = str(version.version)
    # Load by version number, not by alias: if the alias moves while we are
    # downloading, the version we report is still the one we serve.
    model = mlflow.pyfunc.load_model(f"models:/{settings.model_name}/{number}")

    tags = dict(version.tags or {})
    if version.run_id and not ("dataset_version" in tags and "git_sha" in tags):
        try:
            run_tags = client.get_run(version.run_id).data.tags
            tags = {**run_tags, **tags}
        except Exception:
            logger.warning("Could not read tags of run %s", version.run_id, exc_info=True)

    raw_model = None
    try:
        raw_model = model.get_raw_model()
    except Exception:
        logger.debug("Model exposes no raw model", exc_info=True)

    return LoadedModel(
        name=settings.model_name,
        alias=settings.model_alias,
        version=number,
        run_id=version.run_id or None,
        dataset_version=tags.get("dataset_version"),
        git_sha=tags.get("git_sha"),
        loaded_at=datetime.now(UTC),
        model=model,
        raw_model=raw_model,
    )


class ModelService:
    """Holds the model being served and knows whether it is ready."""

    def __init__(self) -> None:
        self._loaded: LoadedModel | None = None
        self._reason = "model is loading"
        self._retryable = True

    @property
    def loaded(self) -> LoadedModel | None:
        return self._loaded

    @property
    def not_ready_reason(self) -> str:
        return self._reason

    def require(self) -> LoadedModel:
        loaded = self._loaded
        if loaded is None:
            raise ModelNotReady(self._reason)
        return loaded

    def load(self) -> bool:
        """Load the approved model and prove it predicts. Never raises.

        The service becomes ready only after a test prediction succeeds. On
        failure the previously loaded model, if any, keeps serving.
        """
        try:
            settings = Settings.from_env()
        except ConfigurationError as error:
            self._reason, self._retryable = f"configuration error: {error}", False
            logger.error("Cannot load a model: %s", error)
            return False

        try:
            candidate = resolve_and_load(settings)
            candidate.predict(READINESS_SAMPLE)
        except Exception as error:
            # Details go to the log only: the health endpoint is public.
            self._reason = f"model could not be loaded ({type(error).__name__})"
            self._retryable = True
            logger.exception("Loading %s@%s failed", settings.model_name, settings.model_alias)
            return False

        self._loaded = candidate
        logger.info(
            "Serving %s version %s (alias %s, run %s, dataset %s, git %s)",
            candidate.name,
            candidate.version,
            candidate.alias,
            candidate.run_id,
            candidate.dataset_version,
            candidate.git_sha,
        )
        return True

    def load_until_ready(self, stop: threading.Event) -> None:
        """Keep trying while the failure may be temporary (registry unreachable, ...)."""
        while not stop.is_set():
            if self.load() or not self._retryable:
                return
            stop.wait(RETRY_SECONDS)
