"""Data contract for the SMS spam dataset.

A dataset version is a directory holding ``train.csv`` and ``eval.csv``, both with
the columns ``id,label,text``. The contract is checked before any training or
evaluation; a single violation must stop the run.

    python -m ml.data_contract data/v1
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

COLUMNS = ["id", "label", "text"]
LABELS = ("ham", "spam")
POSITIVE_LABEL = "spam"

# Longest message in the source corpus is 910 characters. The app's input
# schema must use the same bound so serving and training agree.
MIN_TEXT_LENGTH = 1
MAX_TEXT_LENGTH = 1000

MIN_ROWS = {"train": 500, "eval": 200}
# The corpus is ~13% spam. A split far outside this band means a broken
# extraction or a labelling problem, not a new data delivery.
SPAM_SHARE_RANGE = (0.05, 0.40)

_ID_PATTERN = re.compile(r"[0-9a-f]{16}")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Canonical form used to decide whether two messages are the same."""
    return _WHITESPACE.sub(" ", text).strip().lower()


def message_id(text: str) -> str:
    """Stable identifier of a message, derived from its normalized text."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()[:16]


def load_split(path: Path) -> pd.DataFrame:
    """Read a split without letting pandas turn texts like "NA" into NaN."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8")


def _check_split(name: str, df: pd.DataFrame) -> list[str]:
    if list(df.columns) != COLUMNS:
        return [f"{name}: columns are {list(df.columns)}, expected {COLUMNS}"]

    errors: list[str] = []

    min_rows = MIN_ROWS.get(name, 1)
    if len(df) < min_rows:
        errors.append(f"{name}: {len(df)} rows, expected at least {min_rows}")

    for column in COLUMNS:
        values = df[column]
        missing = int(values.isna().sum())
        if missing:
            errors.append(f"{name}.{column}: {missing} missing value(s)")
        not_str = int((~values.dropna().map(lambda v: isinstance(v, str))).sum())
        if not_str:
            errors.append(f"{name}.{column}: {not_str} value(s) are not strings")
    if errors:
        # The remaining checks assume complete string columns.
        return errors

    blank = int((df["text"].str.strip() == "").sum())
    if blank:
        errors.append(f"{name}.text: {blank} blank message(s)")

    lengths = df["text"].str.len()
    out_of_range = int(((lengths < MIN_TEXT_LENGTH) | (lengths > MAX_TEXT_LENGTH)).sum())
    if out_of_range:
        errors.append(
            f"{name}.text: {out_of_range} message(s) outside "
            f"{MIN_TEXT_LENGTH}..{MAX_TEXT_LENGTH} characters"
        )

    invalid_labels = sorted(set(df["label"]) - set(LABELS))
    if invalid_labels:
        errors.append(f"{name}.label: invalid label(s) {invalid_labels}, expected {list(LABELS)}")
    else:
        absent = sorted(set(LABELS) - set(df["label"]))
        if absent:
            errors.append(f"{name}.label: no rows for label(s) {absent}")
        elif len(df):
            share = float((df["label"] == POSITIVE_LABEL).mean())
            low, high = SPAM_SHARE_RANGE
            if not low <= share <= high:
                errors.append(f"{name}.label: spam share {share:.3f} outside {low}..{high}")

    malformed = int((~df["id"].map(lambda v: bool(_ID_PATTERN.fullmatch(v)))).sum())
    if malformed:
        errors.append(f"{name}.id: {malformed} id(s) are not 16 lowercase hex characters")

    mismatched = int((df["id"] != df["text"].map(message_id)).sum())
    if mismatched:
        errors.append(f"{name}.id: {mismatched} id(s) do not match their text")

    duplicated = int(df["id"].duplicated().sum())
    if duplicated:
        errors.append(f"{name}.id: {duplicated} duplicated id(s)")

    duplicated_text = int(df["text"].map(normalize_text).duplicated().sum())
    if duplicated_text:
        errors.append(f"{name}.text: {duplicated_text} duplicated message(s)")

    return errors


def validate_frames(train: pd.DataFrame, eval_: pd.DataFrame) -> list[str]:
    """Return every contract violation found. An empty list means the data is valid."""
    errors = _check_split("train", train) + _check_split("eval", eval_)

    if list(train.columns) == COLUMNS and list(eval_.columns) == COLUMNS:
        shared_ids = set(train["id"].dropna()) & set(eval_["id"].dropna())
        if shared_ids:
            errors.append(f"train/eval overlap: {len(shared_ids)} shared id(s)")

        def texts(df: pd.DataFrame) -> set[str]:
            return {normalize_text(t) for t in df["text"] if isinstance(t, str)}

        shared_texts = texts(train) & texts(eval_)
        if shared_texts:
            errors.append(f"train/eval overlap: {len(shared_texts)} shared message(s)")

    return errors


def validate_dir(data_dir: Path) -> list[str]:
    """Validate the dataset version stored in ``data_dir``."""
    missing = [name for name in ("train.csv", "eval.csv") if not (data_dir / name).is_file()]
    if missing:
        return [f"{data_dir}: missing file(s) {missing}"]
    return validate_frames(load_split(data_dir / "train.csv"), load_split(data_dir / "eval.csv"))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m ml.data_contract <data_dir>", file=sys.stderr)
        return 2

    data_dir = Path(args[0])
    errors = validate_dir(data_dir)
    if errors:
        print(f"Data contract FAILED for {data_dir} ({len(errors)} violation(s)):", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    train = load_split(data_dir / "train.csv")
    eval_ = load_split(data_dir / "eval.csv")
    print(f"Data contract OK for {data_dir}: train={len(train)} rows, eval={len(eval_)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
