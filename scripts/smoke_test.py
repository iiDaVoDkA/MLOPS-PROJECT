"""Smoke-test a running instance of the app: readiness, predictions, model identity.

Standard library only, so it runs anywhere (CI, a laptop, against the live URL).
With --expect-app-sha it first waits until the URL serves that image: right after a
deploy, Render keeps answering with the old instance until the new one is ready.

    python -m scripts.smoke_test --url http://localhost:8000 --expect-version 1 \\
        --expect-app-sha "$(git rev-parse HEAD)" --timeout 120

Exits 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

HAM = "Are we still meeting for lunch tomorrow?"
SPAM = "WINNER! You have won a free prize. Text WIN to 80086 to claim now."
POLL_SECONDS = 3.0
SHA_POLL_SECONDS = 5.0

# (method, url, json body or None) -> (status code, parsed JSON body or None)
Http = Callable[[str, str, Any], tuple[int, Any]]


def urllib_http(method: str, url: str, body: Any = None, timeout: float = 10.0) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    try:
        return status, json.loads(raw) if raw else None
    except ValueError:
        return status, None


def wait_until_ready(
    http: Http, url: str, timeout: float, sleep: Callable[[float], None] = time.sleep
) -> str | None:
    """Poll /health/ready. Return None once ready, or the reason it never became ready."""
    deadline = time.monotonic() + timeout
    last = "no response"
    while True:
        try:
            status, body = http("GET", f"{url}/health/ready", None)
            if status == 200:
                return None
            last = f"HTTP {status}: {body}"
        except (OSError, urllib.error.URLError) as error:
            last = f"{type(error).__name__}: {error}"
        if time.monotonic() >= deadline:
            return f"not ready after {timeout:.0f}s (last answer: {last})"
        sleep(POLL_SECONDS)


def wait_for_app_sha(
    http: Http,
    url: str,
    expected: str,
    deadline: float,
    sleep: Callable[[float], None] = time.sleep,
) -> str | None:
    """Poll /model-info until it reports ``expected`` as app_git_sha.

    Return None once it does, or the reason it never did by ``deadline``
    (a time.monotonic() value).
    """
    last = "no response"
    while True:
        try:
            status, info = http("GET", f"{url}/model-info", None)
            if status == 200 and isinstance(info, dict):
                if info.get("app_git_sha") == expected:
                    return None
                last = f"app_git_sha {info.get('app_git_sha')!r}"
            else:
                last = f"HTTP {status}: {info}"
        except (OSError, urllib.error.URLError) as error:
            last = f"{type(error).__name__}: {error}"
        if time.monotonic() >= deadline:
            return f"never served app_git_sha {expected!r} (last answer: {last})"
        sleep(SHA_POLL_SECONDS)


def run_checks(
    http: Http,
    url: str,
    expect_version: str | None = None,
    expect_app_sha: str | None = None,
    timeout: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run every check and return a summary; ``summary["failures"]`` is empty on success."""
    url = url.rstrip("/")
    failures: list[str] = []
    summary: dict[str, Any] = {"url": url, "predictions": {}, "failures": failures}

    deadline = time.monotonic() + timeout
    not_ready = wait_until_ready(http, url, timeout, sleep)
    if not_ready:
        failures.append(f"/health/ready: {not_ready}")
        return summary
    summary["ready"] = True

    if expect_app_sha is not None:
        wrong_image = wait_for_app_sha(http, url, expect_app_sha, deadline, sleep)
        if wrong_image:
            failures.append(f"/model-info: {wrong_image}")
            return summary

    for name, text in (("ham", HAM), ("spam", SPAM)):
        status, body = http("POST", f"{url}/predict", {"text": text})
        if status != 200 or not isinstance(body, dict):
            failures.append(f"/predict ({name} message): HTTP {status}: {body}")
            continue
        summary["predictions"][name] = body.get("prediction")
        if body.get("prediction") not in {"ham", "spam"}:
            failures.append(f"/predict ({name} message): unexpected prediction {body}")
        if expect_version is not None and body.get("model_version") != expect_version:
            failures.append(
                f"/predict ({name} message): model_version {body.get('model_version')!r}, "
                f"expected {expect_version!r}"
            )

    status, info = http("GET", f"{url}/model-info", None)
    if status != 200 or not isinstance(info, dict):
        failures.append(f"/model-info: HTTP {status}: {info}")
    else:
        summary["model_version"] = info.get("version")
        summary["app_git_sha"] = info.get("app_git_sha")
        if expect_version is not None and info.get("version") != expect_version:
            failures.append(
                f"/model-info: version {info.get('version')!r}, expected {expect_version!r}"
            )
        if expect_app_sha is not None and info.get("app_git_sha") != expect_app_sha:
            failures.append(
                f"/model-info: app_git_sha {info.get('app_git_sha')!r}, expected {expect_app_sha!r}"
            )

    status, body = http("POST", f"{url}/predict", {"text": ""})
    summary["empty_text_status"] = status
    if status != 422:
        failures.append(f"/predict (empty text): HTTP {status}, expected 422")

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", required=True, help="base URL, e.g. http://localhost:8000")
    parser.add_argument("--expect-version", help="registry version the app must serve")
    parser.add_argument("--expect-app-sha", help="commit the image must report (waits for it)")
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="seconds to wait for ready and for the expected image",
    )
    args = parser.parse_args(argv)

    summary = run_checks(
        urllib_http, args.url, args.expect_version, args.expect_app_sha, args.timeout
    )
    summary["ok"] = not summary["failures"]
    print(json.dumps(summary, indent=2))
    for failure in summary["failures"]:
        print(f"FAILED: {failure}", file=sys.stderr)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
