from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path


def load_texts(csv_path: Path, spammy: bool) -> list[str]:
    with csv_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if spammy:
        rows = [row for row in rows if row.get("label") == "spam"]
    return [row["text"] for row in rows]


def send(url: str, payload: dict) -> tuple[str, dict | None]:
    request = urllib.request.Request(
        f"{url.rstrip('/')}/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return str(response.status), json.loads(response.read())
    except urllib.error.HTTPError as error:
        return str(error.code), None
    except urllib.error.URLError:
        return "unreachable", None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay CSV messages against /predict")
    parser.add_argument("--csv", type=Path, default=Path("data/v1/eval.csv"))
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--rate", type=float, default=5.0, help="requests per second")
    parser.add_argument("--limit", type=int, default=100, help="number of rows to send")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--broken", action="store_true", help="send a wrong field to get 422s")
    parser.add_argument("--spammy", action="store_true", help="only send spam rows")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    texts = load_texts(args.csv, args.spammy)
    if args.shuffle:
        random.shuffle(texts)
    texts = texts[: args.limit]

    statuses: Counter[str] = Counter()
    predictions: Counter[str] = Counter()
    interval = 1 / args.rate
    next_send = time.monotonic()
    for text in texts:
        time.sleep(max(0.0, next_send - time.monotonic()))
        next_send += interval
        payload = {"msg": text} if args.broken else {"text": text}
        status, body = send(args.url, payload)
        statuses[status] += 1
        if body:
            predictions[body["prediction"]] += 1

    print(f"sent {len(texts)} requests to {args.url}/predict")
    for status, count in sorted(statuses.items()):
        print(f"  status {status}: {count}")
    for label, count in sorted(predictions.items()):
        print(f"  predicted {label}: {count}")


if __name__ == "__main__":
    main()
