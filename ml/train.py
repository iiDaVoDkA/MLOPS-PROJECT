"""Train a spam classifier, log it to MLflow and apply the promotion gates.

The run records everything needed to reproduce it: Git commit, dataset version
and DVC hash, hyperparameters, metrics on the held-out set, the model and the
evaluation artifacts. The candidate is then compared with the current champion
on the same held-out messages. It is registered and given the champion alias
only if it passes ``ml/gates.yaml``; otherwise the champion is retained and the
reason is logged.

Configuration comes from the environment: MLFLOW_TRACKING_URI,
MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD, MODEL_NAME, MODEL_ALIAS.

    python -m ml.train --data-version v1 --run-name baseline-v1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from ml.data_contract import POSITIVE_LABEL
from ml.evaluate import (
    FEATURE_COLUMNS,
    evaluate,
    load_validated,
    save_confusion_matrix,
    save_errors,
)
from ml.promote import GATES_PATH, load_gates, should_promote

DEFAULT_EXPERIMENT = "sms-spam"
SEED = 42
CV_FOLDS = 5

INPUT_EXAMPLE = pd.DataFrame(
    {
        "text": [
            "Are we still meeting for lunch tomorrow?",
            "WINNER! You have won a free prize. Text WIN to 80086 to claim now.",
        ]
    }
)
# One required string column in, one label ("ham" or "spam") per row out.
MODEL_SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec("string", "text")]),
    outputs=Schema([ColSpec("string")]),
)


def build_pipeline(params: dict[str, Any]) -> Pipeline:
    """TF-IDF over the ``text`` column followed by logistic regression.

    The pipeline takes a DataFrame with a ``text`` column and returns labels,
    so the logged model can be served without any extra preprocessing code.
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, params["ngram_max"]),
        min_df=params["min_df"],
        sublinear_tf=True,
    )
    classifier = LogisticRegression(
        C=params["C"],
        class_weight=params["class_weight"],
        max_iter=1000,
        random_state=SEED,
    )
    return Pipeline(
        [
            ("features", ColumnTransformer([("tfidf", vectorizer, "text")])),
            ("classifier", classifier),
        ]
    )


def git_state() -> tuple[str, bool]:
    """Return (commit SHA, whether the working tree differs from that commit).

    New untracked files count as changes; git-ignored paths (data, caches) do not.
    """
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        return sha, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        sha = os.environ.get("GITHUB_SHA")
        if not sha:
            raise SystemExit("Cannot determine the Git commit: not a repository and no GITHUB_SHA.")
        return sha, False


def dvc_hash(data_version: str) -> str:
    pointer = Path("data") / f"{data_version}.dvc"
    return yaml.safe_load(pointer.read_text(encoding="utf-8"))["outs"][0]["md5"]


def find_champion(client: MlflowClient, name: str, alias: str):
    """Return the model version behind the alias, or None if there is none yet.

    Only an explicit "not found" means "no champion". Any other failure (network,
    credentials) is raised: silently treating it as a first release would
    promote a candidate without comparing it to the real champion.
    """
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        # Servers report a missing model or alias with different error codes,
        # so confirm explicitly that the alias really is absent.
        if _alias_is_set(client, name, alias):
            raise
        return None


def _alias_is_set(client: MlflowClient, name: str, alias: str) -> bool:
    try:
        model = client.get_registered_model(name)
    except MlflowException as error:
        if error.error_code == "RESOURCE_DOES_NOT_EXIST":
            return False
        raise
    return alias in (model.aliases or {})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-version", required=True, help="dataset version, e.g. v1")
    parser.add_argument("--run-name", help="MLflow run name (default: train-<data-version>)")
    parser.add_argument("--C", type=float, default=10.0, help="inverse regularization strength")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--ngram-max", type=int, default=1, help="largest word n-gram")
    parser.add_argument("--min-df", type=int, default=1, help="minimum document frequency")
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="log the run and the decision but leave the registry untouched",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow registering a model trained from uncommitted code",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME")
    model_alias = os.environ.get("MODEL_ALIAS", "champion")
    if not model_name:
        raise SystemExit("MODEL_NAME is not set.")

    sha, dirty = git_state()
    if dirty and not (args.no_register or args.allow_dirty):
        raise SystemExit(
            "The working tree has uncommitted changes, so the logged Git SHA would not describe "
            "this code. Commit first, or pass --no-register or --allow-dirty."
        )

    data_dir = Path("data") / args.data_version
    train, eval_frame = load_validated(data_dir)
    data_hash = dvc_hash(args.data_version)
    gates = load_gates()
    primary = gates["primary_metric"]

    hyperparameters = {
        "C": args.C,
        "class_weight": None if args.class_weight == "none" else args.class_weight,
        "ngram_max": args.ngram_max,
        "min_df": args.min_df,
    }
    provenance = {
        "git_sha": sha,
        "git_dirty": str(dirty).lower(),
        "dataset_version": args.data_version,
        "dvc_hash": data_hash,
    }

    client = MlflowClient()
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT))

    with mlflow.start_run(run_name=args.run_name or f"train-{args.data_version}") as run:
        mlflow.set_tags(provenance)
        mlflow.log_params(
            {
                **{k: str(v) for k, v in hyperparameters.items()},
                "model_type": "tfidf-logistic-regression",
                "seed": SEED,
                "dataset_version": args.data_version,
                "dvc_hash": data_hash,
                "git_sha": sha,
                "train_rows": len(train),
                "eval_rows": len(eval_frame),
            }
        )

        # Cross-validation on the training split only: the number to look at when
        # tuning, so the held-out set is never used to choose hyperparameters.
        cv_scores = cross_val_score(
            build_pipeline(hyperparameters),
            train[FEATURE_COLUMNS],
            train["label"],
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED),
            scoring=make_scorer(f1_score, pos_label=POSITIVE_LABEL),
        )
        mlflow.log_metrics(
            {"cv_f1_spam_mean": float(cv_scores.mean()), "cv_f1_spam_std": float(cv_scores.std())}
        )

        pipeline = build_pipeline(hyperparameters).fit(train[FEATURE_COLUMNS], train["label"])
        metrics, y_pred = evaluate(pipeline, eval_frame)
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            title = f"Held-out set ({args.data_version}, n={len(eval_frame)})"
            save_confusion_matrix(
                list(eval_frame["label"]), y_pred, out / "confusion_matrix.png", title
            )
            save_errors(eval_frame, y_pred, out / "misclassified.csv")
            (out / "eval_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            for path in sorted(out.iterdir()):
                mlflow.log_artifact(str(path), artifact_path="evaluation")
        mlflow.log_artifact(str(data_dir / "manifest.json"), artifact_path="data")
        mlflow.log_artifact(str(GATES_PATH), artifact_path="evaluation")

        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=MODEL_SIGNATURE,
            input_example=INPUT_EXAMPLE,
        )

        # Score the current champion on the very same held-out messages.
        champion = find_champion(client, model_name, model_alias)
        champion_score = None
        if champion is not None:
            champion_model = mlflow.pyfunc.load_model(f"models:/{model_name}/{champion.version}")
            champion_metrics, _ = evaluate(champion_model, eval_frame)
            champion_score = champion_metrics[primary]
            mlflow.log_metrics({f"champion_{k}": v for k, v in champion_metrics.items()})
            mlflow.set_tag("champion_version_before", champion.version)

        promote, reason = should_promote(metrics[primary], champion_score, gates, metrics)

        registered_version = None
        if promote and not args.no_register:
            version = mlflow.register_model(
                model_info.model_uri,
                model_name,
                tags={**provenance, "approved": "true", primary: f"{metrics[primary]:.4f}"},
            )
            client.set_registered_model_alias(model_name, model_alias, version.version)
            registered_version = str(version.version)
            mlflow.set_tag("registered_version", registered_version)
            decision = "promoted"
        elif promote:
            decision = "would_promote"
        else:
            decision = "rejected"
        mlflow.set_tags({"promotion_decision": decision, "promotion_reason": reason})

        summary = {
            "run_id": run.info.run_id,
            "model_name": model_name,
            "dataset_version": args.data_version,
            "dvc_hash": data_hash,
            "git_sha": sha,
            "metrics": metrics,
            "champion_version_before": champion.version if champion is not None else None,
            "champion_score": champion_score,
            "decision": decision,
            "reason": reason,
            "registered_version": registered_version,
        }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
