"""Score a model on the held-out evaluation set.

Champion and candidate are always scored with this module on the same
``eval.csv``, so their numbers are comparable.

    python -m ml.evaluate --data-version v1 --model-uri "models:/$MODEL_NAME@champion"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ml.data_contract import LABELS, POSITIVE_LABEL, load_split, validate_dir

FEATURE_COLUMNS = ["text"]


def predict_labels(model: Any, frame: pd.DataFrame) -> list[str]:
    """Predict with a scikit-learn pipeline or an MLflow pyfunc model alike."""
    return [str(label) for label in model.predict(frame[FEATURE_COLUMNS])]


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[POSITIVE_LABEL], average=None, zero_division=0
    )
    return {
        "f1_spam": float(f1[0]),
        "precision_spam": float(precision[0]),
        "recall_spam": float(recall[0]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def evaluate(model: Any, eval_frame: pd.DataFrame) -> tuple[dict[str, float], list[str]]:
    y_pred = predict_labels(model, eval_frame)
    return compute_metrics(list(eval_frame["label"]), y_pred), y_pred


def save_confusion_matrix(y_true: list[str], y_pred: list[str], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")  # no display in CI
    import matplotlib.pyplot as plt

    matrix = confusion_matrix(y_true, y_pred, labels=list(LABELS))
    figure, axis = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(matrix, display_labels=list(LABELS)).plot(
        ax=axis, cmap="Blues", colorbar=False, values_format="d"
    )
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_errors(eval_frame: pd.DataFrame, y_pred: list[str], path: Path) -> int:
    """Write the misclassified messages, the most useful artifact for error analysis."""
    errors = eval_frame.assign(predicted=y_pred)
    errors = errors[errors["label"] != errors["predicted"]]
    errors[["id", "label", "predicted", "text"]].to_csv(path, index=False)
    return len(errors)


def load_validated(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, eval), refusing data that breaks the contract."""
    violations = validate_dir(data_dir)
    if violations:
        raise SystemExit(f"Data contract failed for {data_dir}:\n  - " + "\n  - ".join(violations))
    return load_split(data_dir / "train.csv"), load_split(data_dir / "eval.csv")


def main() -> None:
    import mlflow

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-version", required=True, help="dataset version, e.g. v1")
    parser.add_argument("--model-uri", required=True, help="e.g. models:/<name>@champion")
    args = parser.parse_args()

    _, eval_frame = load_validated(Path("data") / args.data_version)
    model = mlflow.pyfunc.load_model(args.model_uri)
    metrics, _ = evaluate(model, eval_frame)
    print(json.dumps({"model_uri": args.model_uri, "rows": len(eval_frame), **metrics}, indent=2))


if __name__ == "__main__":
    main()
