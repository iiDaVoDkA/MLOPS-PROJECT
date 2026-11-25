from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from ml.data_contract import LABELS, load_split, validate_dir
from ml.evaluate import FEATURE_COLUMNS, evaluate, save_confusion_matrix, save_errors
from ml.promote import GATES_PATH, load_gates, should_promote
from ml.train import INPUT_EXAMPLE, MODEL_SIGNATURE, build_pipeline, find_champion, git_state

DEFAULT_MODEL_NAME = "sms-spam-classifier"
DEFAULT_EXPERIMENT = "sms-spam"
TRAINING_PARAMS = {"C": 10.0, "class_weight": None, "ngram_max": 1, "min_df": 1}
LOAD_TEST_TEXT = "Are we still meeting for lunch tomorrow?"
DATASET_VERSION_PATTERN = re.compile(r"v(\d+)")


class PipelineError(RuntimeError):
    pass


EXPECTED_FAILURES = (PipelineError, MlflowException, OSError, ValueError, KeyError)


def version_number(dataset_version: str | None) -> int | None:
    match = DATASET_VERSION_PATTERN.fullmatch(dataset_version or "")
    return int(match.group(1)) if match else None


def needs_retraining(
    dataset_version: str, champion_tags: dict, champion_score: float, gates: dict
) -> tuple[bool, str]:
    metric = gates["primary_metric"]
    higher = gates["greater_is_better"]
    floor = gates["quality_floor"]
    max_drop = gates["retrain"]["max_f1_drop"]

    champion_data = champion_tags.get("dataset_version")
    new_number = version_number(dataset_version)
    champion_number = version_number(champion_data)
    if champion_number is None:
        return True, f"champion has no comparable dataset version ({champion_data!r})"
    if new_number > champion_number:
        return True, f"{dataset_version} is newer than the champion's data {champion_data}"

    below_floor = champion_score < floor if higher else champion_score > floor
    if below_floor:
        return True, f"champion {metric} {champion_score:.4f} is past the quality floor {floor}"

    approved_score = champion_tags.get(metric)
    if approved_score is not None:
        approved_score = float(approved_score)
        drop = approved_score - champion_score if higher else champion_score - approved_score
        if drop >= max_drop - 1e-9:
            return True, (
                f"champion {metric} fell from {approved_score:.4f} to {champion_score:.4f} "
                f"(drop {drop:.4f}, limit {max_drop})"
            )

    return False, (
        f"champion {metric} {champion_score:.4f} on {dataset_version} is healthy "
        f"and was trained on {champion_data}"
    )


def get_champion(client: MlflowClient, name: str, alias: str):
    champion = find_champion(client, name, alias)
    if champion is not None and champion.tags.get("approved") != "true":
        raise PipelineError(
            f"{name}@{alias} points to version {champion.version}, which is not approved"
        )
    return champion


def champion_version(client: MlflowClient, name: str, alias: str) -> str | None:
    champion = find_champion(client, name, alias)
    return str(champion.version) if champion is not None else None


def move_champion(
    client: MlflowClient, name: str, alias: str, new_version: str, expected_version: str | None
) -> None:
    current = champion_version(client, name, alias)
    if current != expected_version:
        raise PipelineError(
            f"{name}@{alias} changed during the run: expected version {expected_version}, found {current}"
        )
    client.set_registered_model_alias(name, alias, new_version)
    after = champion_version(client, name, alias)
    if after != str(new_version):
        raise PipelineError(
            f"{name}@{alias} should point to version {new_version} but points to {after}"
        )


def load_test(name: str, version: str) -> None:
    model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    prediction = str(next(iter(model.predict(pd.DataFrame({"text": [LOAD_TEST_TEXT]})))))
    if prediction not in LABELS:
        raise PipelineError(f"version {version} returned an unknown label {prediction!r}")


def roll_back(client: MlflowClient, name: str, alias: str, target: str) -> tuple[str, str]:
    try:
        target_version = client.get_model_version(name, target)
    except MlflowException as error:
        if error.error_code == "RESOURCE_DOES_NOT_EXIST":
            raise PipelineError(f"{name} has no version {target}") from error
        raise
    if target_version.tags.get("approved") != "true":
        raise PipelineError(f"{name} version {target} was never approved")

    current = champion_version(client, name, alias)
    if current == target:
        return "rollback_unchanged", f"{name}@{alias} already points to version {target}"

    load_test(name, target)
    move_champion(client, name, alias, target, current)
    return "rolled_back", f"{name}@{alias} moved from version {current} to version {target}"


def read_dvc_hash(data_root: Path, dataset_version: str) -> str | None:
    pointer = data_root / f"{dataset_version}.dvc"
    if not pointer.is_file():
        return None
    return yaml.safe_load(pointer.read_text(encoding="utf-8"))["outs"][0]["md5"]


def current_git_sha() -> tuple[str, bool]:
    try:
        return git_state()
    except SystemExit:
        return "unknown", False


def log_evaluation(eval_frame: pd.DataFrame, predictions: list[str], dataset_version: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        title = f"Held-out set ({dataset_version}, n={len(eval_frame)})"
        save_confusion_matrix(
            list(eval_frame["label"]), predictions, out / "confusion_matrix.png", title
        )
        save_errors(eval_frame, predictions, out / "misclassified.csv")
        for path in sorted(out.iterdir()):
            mlflow.log_artifact(str(path), artifact_path="evaluation")


def run_training_cycle(
    args: argparse.Namespace, client: MlflowClient, gates: dict, outcome: dict
) -> None:
    name, alias, dataset_version = args.model_name, args.alias, args.dataset_version
    metric = gates["primary_metric"]

    champion = get_champion(client, name, alias)
    outcome["champion_before"] = str(champion.version) if champion is not None else None

    data_root = Path(args.data_root)
    violations = validate_dir(data_root / dataset_version)
    if violations:
        outcome["validation"] = "failed"
        raise PipelineError(f"data contract failed for {dataset_version}: " + "; ".join(violations))
    outcome["validation"] = "passed"
    outcome["dvc_hash"] = read_dvc_hash(data_root, dataset_version)
    train = load_split(data_root / dataset_version / "train.csv")
    eval_frame = load_split(data_root / dataset_version / "eval.csv")

    sha, dirty = current_git_sha()
    provenance = {
        "git_sha": sha,
        "git_dirty": str(dirty).lower(),
        "dataset_version": dataset_version,
        "dvc_hash": outcome["dvc_hash"] or "unknown",
    }

    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT))
    with mlflow.start_run(run_name=f"ct-{dataset_version}") as run:
        outcome["run_id"] = run.info.run_id
        mlflow.set_tags(
            {
                **provenance,
                "ct_model_name": name,
                "champion_before": str(outcome["champion_before"]),
            }
        )

        champion_score = None
        if champion is None:
            retrain, reason = True, "no champion yet: training the initial model"
        else:
            champion_model = mlflow.pyfunc.load_model(f"models:/{name}/{champion.version}")
            champion_metrics, _ = evaluate(champion_model, eval_frame)
            champion_score = champion_metrics[metric]
            mlflow.log_metrics(
                {f"champion_{key}": value for key, value in champion_metrics.items()}
            )
            retrain, reason = needs_retraining(
                dataset_version, champion.tags, champion_score, gates
            )
        outcome["champion_score"] = champion_score
        mlflow.set_tags({"retrain": str(retrain).lower(), "retrain_reason": reason})

        if not retrain:
            outcome["decision"], outcome["reason"] = "retain_no_retraining", reason
            mlflow.set_tags({"ct_decision": outcome["decision"], "ct_reason": reason})
            return

        outcome["retrained"] = True
        mlflow.log_params(
            {
                **{key: str(value) for key, value in TRAINING_PARAMS.items()},
                "train_rows": len(train),
            }
        )
        pipeline = build_pipeline(TRAINING_PARAMS).fit(train[FEATURE_COLUMNS], train["label"])
        metrics, predictions = evaluate(pipeline, eval_frame)
        outcome["candidate_score"] = metrics[metric]
        mlflow.log_metrics(metrics)
        log_evaluation(eval_frame, predictions, dataset_version)
        model_info = mlflow.sklearn.log_model(
            pipeline, name="model", signature=MODEL_SIGNATURE, input_example=INPUT_EXAMPLE
        )
        candidate = mlflow.register_model(
            model_info.model_uri,
            name,
            tags={**provenance, "approved": "false", metric: f"{metrics[metric]:.4f}"},
        )
        candidate_version = str(candidate.version)
        outcome["candidate_version"] = candidate_version

        promote, promote_reason = should_promote(metrics[metric], champion_score, gates, metrics)
        if promote:
            client.set_model_version_tag(name, candidate_version, "approved", "true")
            try:
                move_champion(client, name, alias, candidate_version, outcome["champion_before"])
            except Exception:
                client.set_model_version_tag(name, candidate_version, "approved", "false")
                raise
            decision = "bootstrapped" if champion is None else "promoted"
        else:
            decision = "retain_rejected"

        outcome["decision"], outcome["reason"] = decision, f"{reason}; {promote_reason}"
        mlflow.set_tags(
            {
                "ct_decision": decision,
                "ct_reason": outcome["reason"],
                "candidate_version": candidate_version,
            }
        )


def run_rollback(args: argparse.Namespace, client: MlflowClient, outcome: dict) -> None:
    outcome["champion_before"] = champion_version(client, args.model_name, args.alias)
    decision, reason = roll_back(client, args.model_name, args.alias, str(args.rollback))
    outcome["decision"], outcome["reason"] = decision, reason
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT))
    with mlflow.start_run(run_name=f"rollback-{args.rollback}") as run:
        outcome["run_id"] = run.info.run_id
        mlflow.set_tags(
            {
                "ct_model_name": args.model_name,
                "ct_decision": decision,
                "ct_reason": reason,
                "champion_before": str(outcome["champion_before"]),
                "rollback_target": str(args.rollback),
            }
        )


def dataset_version_arg(value: str) -> str:
    if version_number(value) is None:
        raise argparse.ArgumentTypeError(f"expected a dataset version like v2, got {value!r}")
    return value


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("a model version is a positive integer")
    return number


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuous training: retrain, promote or roll back."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dataset-version", type=dataset_version_arg)
    action.add_argument("--rollback", type=positive_int, metavar="VERSION")
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME") or DEFAULT_MODEL_NAME)
    parser.add_argument("--alias", default=os.environ.get("MODEL_ALIAS") or "champion")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--gates", type=Path, default=GATES_PATH)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outcome = {
        "decision": None,
        "reason": None,
        "model_name": args.model_name,
        "alias": args.alias,
        "validation": "skipped" if args.rollback else "not_run",
        "retrained": False,
        "champion_before": None,
        "champion_after": None,
        "candidate_version": None,
        "champion_score": None,
        "candidate_score": None,
        "run_id": None,
        "dataset_version": args.dataset_version,
        "dvc_hash": None,
        "rollback_target": str(args.rollback) if args.rollback else None,
    }

    client = None
    try:
        if os.environ.get("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
            mlflow.set_registry_uri(os.environ["MLFLOW_TRACKING_URI"])
        client = MlflowClient()
        if args.rollback:
            run_rollback(args, client, outcome)
        else:
            run_training_cycle(args, client, load_gates(args.gates), outcome)
    except EXPECTED_FAILURES as error:
        outcome["decision"] = "failed"
        outcome["reason"] = f"{type(error).__name__}: {error}"
    finally:
        if outcome["decision"] is None:
            outcome["decision"] = "failed"
            outcome["reason"] = "unexpected error, see the traceback in the log"
        finish(outcome, client, args)

    return 1 if outcome["decision"] == "failed" else 0


def finish(outcome: dict, client: MlflowClient | None, args: argparse.Namespace) -> None:
    if client is not None:
        try:
            outcome["champion_after"] = champion_version(client, args.model_name, args.alias)
        except EXPECTED_FAILURES as error:
            outcome["reason"] = f"{outcome['reason']} (champion could not be re-read: {error})"

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "decision.json").write_text(
        json.dumps(outcome, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    sys.exit(main())
