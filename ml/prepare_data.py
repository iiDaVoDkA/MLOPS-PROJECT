"""Build a dataset version from the UCI SMS Spam Collection.

The source is static, so "new data arriving" is simulated with reproducible
batches. Every unique message is assigned a bucket in 0..99 from a hash of its
normalized text, which makes the assignment independent of row order, random
seeds and library versions:

    buckets  0..19  held-out evaluation set (identical in every version)
    buckets 20..49  training batch 1 (v1)
    buckets 50..99  training batch 2 (added in v2)

    python -m ml.prepare_data --version v1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from ml.data_contract import COLUMNS, LABELS, message_id, normalize_text

SOURCE_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SOURCE_SHA256 = "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3"
SOURCE_MEMBER = "SMSSpamCollection"

EVAL_BUCKETS = range(20)
TRAIN_BUCKETS = {
    "v1": range(20, 50),
    "v2": range(20, 100),
}


def bucket_of(identifier: str) -> int:
    return int(identifier[:8], 16) % 100


def read_source(source: Path | None) -> bytes:
    """Return the source archive, checking it is the exact snapshot we documented."""
    if source is None:
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            payload = response.read()
    else:
        payload = source.read_bytes()

    digest = hashlib.sha256(payload).hexdigest()
    if digest != SOURCE_SHA256:
        raise SystemExit(
            f"Source archive sha256 is {digest}, expected {SOURCE_SHA256}. "
            "The upstream file changed; review it before building a dataset from it."
        )
    return payload


def parse_messages(archive: bytes) -> dict[str, tuple[str, str]]:
    """Return ``{id: (label, text)}`` with one entry per unique message."""
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        raw = zf.read(SOURCE_MEMBER).decode("utf-8")

    messages: dict[str, tuple[str, str]] = {}
    labels_seen: dict[str, set[str]] = {}
    for line_number, line in enumerate(raw.split("\n"), start=1):
        if not line.strip():
            continue
        label, separator, text = line.partition("\t")
        text = text.strip()
        if not separator or label not in LABELS or not normalize_text(text):
            raise SystemExit(f"Unexpected source line {line_number}: {line[:80]!r}")
        identifier = message_id(text)
        labels_seen.setdefault(identifier, set()).add(label)
        # Keep the first occurrence of a duplicated message.
        messages.setdefault(identifier, (label, text))

    # A message labelled both ways has no trustworthy label.
    for identifier, labels in labels_seen.items():
        if len(labels) > 1:
            del messages[identifier]
    return messages


def write_split(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(rows)


def build(version: str, out_dir: Path, source: Path | None = None) -> dict:
    messages = parse_messages(read_source(source))

    splits: dict[str, list[tuple[str, str, str]]] = {"train": [], "eval": []}
    for identifier in sorted(messages):
        label, text = messages[identifier]
        bucket = bucket_of(identifier)
        if bucket in EVAL_BUCKETS:
            splits["eval"].append((identifier, label, text))
        elif bucket in TRAIN_BUCKETS[version]:
            splits["train"].append((identifier, label, text))

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "dataset": "sms-spam-collection",
        "version": version,
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "eval_buckets": [EVAL_BUCKETS.start, EVAL_BUCKETS.stop - 1],
        "train_buckets": [TRAIN_BUCKETS[version].start, TRAIN_BUCKETS[version].stop - 1],
        "splits": {},
    }
    for name, rows in splits.items():
        write_split(out_dir / f"{name}.csv", rows)
        counts = Counter(label for _, label, _ in rows)
        manifest["splits"][name] = {"rows": len(rows), **{label: counts[label] for label in LABELS}}

    # No timestamp: the same inputs must always produce the same DVC hash.
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", required=True, choices=sorted(TRAIN_BUCKETS))
    parser.add_argument("--out", type=Path, help="output directory (default: data/<version>)")
    parser.add_argument(
        "--source", type=Path, help="local copy of the source zip (default: download)"
    )
    args = parser.parse_args()

    out_dir = args.out or Path("data") / args.version
    manifest = build(args.version, out_dir, args.source)
    print(f"Wrote {out_dir}: {json.dumps(manifest['splits'])}")


if __name__ == "__main__":
    main()
