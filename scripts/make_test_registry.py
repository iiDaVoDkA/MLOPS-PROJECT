"""Create a throwaway MLflow registry holding a tiny spam model, for CI and tests.

It trains on a handful of hard-coded messages (offline, a few seconds) and
registers two versions of the model:

    version 1  approved, carries the ``champion`` alias  -> what the app must serve
    version 2  newer, NOT approved, no alias             -> what the app must ignore

The registry is a SQLite file plus an artifact folder under ``--root``. Paths are
absolute, so mount the folder at the same path inside a container:

    python -m scripts.make_test_registry --root /tmp/ci-registry --force
    docker run -v /tmp/ci-registry:/tmp/ci-registry \\
        -e MLFLOW_TRACKING_URI=sqlite:////tmp/ci-registry/mlflow.db \\
        -e MODEL_NAME=sms-spam-classifier ...
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

from ml.train import INPUT_EXAMPLE, MODEL_SIGNATURE, build_pipeline

DEFAULT_MODEL_NAME = "sms-spam-classifier"
CHAMPION_ALIAS = "champion"

MESSAGES = [
    ("ham", "Are we still meeting for lunch tomorrow?"),
    ("ham", "Sorry, running late. Be there in ten minutes."),
    ("ham", "Can you pick up some milk on your way home?"),
    ("ham", "Happy birthday! Hope you have a lovely day."),
    ("ham", "I left my keys at your place, I'll grab them tonight."),
    ("ham", "The meeting moved to 3pm, see you there."),
    ("spam", "WINNER! You have won a free prize. Text WIN to 80086 to claim now."),
    ("spam", "URGENT! Your mobile number has won a 2000 cash award. Call 09061701461."),
    ("spam", "FREE entry in our weekly competition. Text WIN to 80086 now."),
    ("spam", "Congratulations! Claim your free ringtone, reply YES. Std txt rate."),
    ("spam", "You have been selected for a cash prize. Call now to claim, valid 12 hours."),
    ("spam", "Free entry to win a holiday! Txt CLAIM to 87121 now. T&Cs apply."),
]

PARAMS = {"C": 10.0, "class_weight": None, "ngram_max": 1, "min_df": 1}


def tracking_uri_for(root: Path) -> str:
    return f"sqlite:///{root.as_posix()}/mlflow.db"


def _log_model(params: dict) -> str:
    train = pd.DataFrame(MESSAGES, columns=["label", "text"])
    pipeline = build_pipeline(params).fit(train[["text"]], train["label"])
    with mlflow.start_run():
        mlflow.log_params({k: str(v) for k, v in params.items()})
        info = mlflow.sklearn.log_model(
            pipeline, name="model", signature=MODEL_SIGNATURE, input_example=INPUT_EXAMPLE
        )
    return info.model_uri


def build(root: Path, model_name: str = DEFAULT_MODEL_NAME, force: bool = False) -> dict:
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        if not force:
            raise SystemExit(f"{root} is not empty. Pass --force to replace it.")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    uri = tracking_uri_for(root)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient()
    experiment_id = client.create_experiment(
        "ci-test-registry", artifact_location=(root / "artifacts").as_uri()
    )
    mlflow.set_experiment(experiment_id=experiment_id)

    common = {"dataset_version": "ci-fixture", "git_sha": "ci-fixture"}
    champion = mlflow.register_model(
        _log_model(PARAMS), model_name, tags={**common, "approved": "true"}
    )
    client.set_registered_model_alias(model_name, CHAMPION_ALIAS, champion.version)

    # Registered after the champion, so it is the newest version, but never approved.
    candidate = mlflow.register_model(
        _log_model({**PARAMS, "C": 1.0}), model_name, tags={**common, "approved": "false"}
    )

    summary = {
        "tracking_uri": uri,
        "model_name": model_name,
        "model_alias": CHAMPION_ALIAS,
        "champion_version": str(champion.version),
        "newer_candidate_version": str(candidate.version),
    }
    (root / "registry.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="folder for the registry")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--force", action="store_true", help="replace an existing registry")
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.model_name, args.force), indent=2))


if __name__ == "__main__":
    main()
