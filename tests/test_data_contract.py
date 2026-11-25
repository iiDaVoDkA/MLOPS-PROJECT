"""The data contract must accept a clean dataset and name every kind of violation."""

from __future__ import annotations

import pandas as pd
import pytest

from ml import data_contract
from ml.data_contract import COLUMNS, MAX_TEXT_LENGTH, message_id, validate_dir, validate_frames


def make_split(prefix: str, rows: int, spam_every: int = 5) -> pd.DataFrame:
    records = []
    for i in range(rows):
        text = f"{prefix} message number {i}"
        label = "spam" if i % spam_every == 0 else "ham"
        records.append({"id": message_id(text), "label": label, "text": text})
    return pd.DataFrame(records, columns=COLUMNS)


@pytest.fixture(autouse=True)
def small_minimums(monkeypatch):
    monkeypatch.setattr(data_contract, "MIN_ROWS", {"train": 20, "eval": 10})


@pytest.fixture
def train() -> pd.DataFrame:
    return make_split("train", 40)


@pytest.fixture
def eval_() -> pd.DataFrame:
    return make_split("eval", 20)


def test_clean_dataset_passes(train, eval_):
    assert validate_frames(train, eval_) == []


def test_wrong_columns(train, eval_):
    errors = validate_frames(train.rename(columns={"text": "message"}), eval_)
    assert any("columns" in e for e in errors)

    errors = validate_frames(train, eval_.drop(columns=["id"]))
    assert any(e.startswith("eval: columns") for e in errors)


def test_wrong_type(train, eval_):
    train["text"] = train["text"].astype(object)
    train.loc[0, "text"] = 12345
    errors = validate_frames(train, eval_)
    assert any("not strings" in e for e in errors)


def test_missing_and_blank_values(train, eval_):
    with_null = train.copy()
    with_null.loc[3, "label"] = None
    assert any("missing" in e for e in validate_frames(with_null, eval_))

    with_blank = train.copy()
    with_blank.loc[3, "text"] = "   "
    assert any("blank" in e for e in validate_frames(with_blank, eval_))


def test_text_length_out_of_range(train, eval_):
    too_long = "x" * (MAX_TEXT_LENGTH + 1)
    train.loc[0, ["id", "text"]] = [message_id(too_long), too_long]
    assert any("outside" in e and "characters" in e for e in validate_frames(train, eval_))

    at_limit = "y" * MAX_TEXT_LENGTH
    train.loc[0, ["id", "text"]] = [message_id(at_limit), at_limit]
    assert validate_frames(train, eval_) == []


def test_invalid_and_absent_labels(train, eval_):
    invalid = train.copy()
    invalid.loc[1, "label"] = "SPAM"
    assert any("invalid label" in e for e in validate_frames(invalid, eval_))

    only_ham = train.copy()
    only_ham["label"] = "ham"
    assert any("no rows for label" in e for e in validate_frames(only_ham, eval_))

    mostly_spam = train.copy()
    mostly_spam["label"] = ["ham"] * 5 + ["spam"] * 35
    assert any("spam share" in e for e in validate_frames(mostly_spam, eval_))


def test_too_few_rows(train, eval_):
    assert any("at least" in e for e in validate_frames(train.head(10), eval_))


def test_ids_must_match_text_and_be_unique(train, eval_):
    tampered = train.copy()
    tampered.loc[2, "text"] = "a different message"
    assert any("do not match" in e for e in validate_frames(tampered, eval_))

    malformed = train.copy()
    malformed.loc[2, "id"] = "not-an-id"
    assert any("16 lowercase hex" in e for e in validate_frames(malformed, eval_))

    duplicated = pd.concat([train, train.head(1)], ignore_index=True)
    assert any("duplicated" in e for e in validate_frames(duplicated, eval_))


def test_train_eval_overlap(train, eval_):
    leaked = pd.concat([train, eval_.head(2)], ignore_index=True)
    errors = validate_frames(leaked, eval_)
    assert any("shared id" in e for e in errors)
    assert any("shared message" in e for e in errors)


def test_overlap_ignores_case_and_spacing(train, eval_):
    # Same message with different casing and spacing is still a leak, even if
    # someone recomputed nothing and gave it a fresh-looking id.
    text = eval_.loc[0, "text"].upper().replace(" ", "   ")
    leak = pd.DataFrame([{"id": message_id(text), "label": eval_.loc[0, "label"], "text": text}])
    errors = validate_frames(pd.concat([train, leak], ignore_index=True), eval_)
    assert any("overlap" in e for e in errors)


def test_validate_dir_roundtrip(tmp_path, train, eval_):
    assert any("missing file" in e for e in validate_dir(tmp_path))

    # "NA" and "null" are real messages, not missing values.
    for i, text in enumerate(["NA", "null"]):
        train.loc[i, ["id", "text"]] = [message_id(text), text]
    train.to_csv(tmp_path / "train.csv", index=False)
    eval_.to_csv(tmp_path / "eval.csv", index=False)
    assert validate_dir(tmp_path) == []
    assert data_contract.main([str(tmp_path)]) == 0

    eval_.head(3).to_csv(tmp_path / "eval.csv", index=False)
    assert data_contract.main([str(tmp_path)]) == 1
