from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
import uuid
from collections import deque

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

DRIFT_WINDOW_SIZE = 500
METRICS_PATH = "/metrics"
REQUEST_ID_HEADER = "X-Request-ID"
VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")

HTTP_REQUESTS = Counter(
    "http_requests_total", "HTTP requests handled", ["method", "path", "status"]
)
HTTP_DURATION = Histogram("http_request_duration_seconds", "HTTP request duration", ["path"])
PREDICTIONS = Counter("predictions_total", "Predictions served", ["prediction", "model_version"])
MODEL_INFO = Gauge("model_info", "Model version currently served", ["model_version"])
SPAM_SHARE = Gauge(
    "spam_prediction_share", f"Share of spam in the last {DRIFT_WINDOW_SIZE} predictions"
)
INPUT_LENGTH_MEAN = Gauge(
    "input_length_mean", f"Mean message length in the last {DRIFT_WINDOW_SIZE} predictions"
)

log = structlog.get_logger("app.http")


class CurrentStdoutHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout
        super().emit(record)


def configure_logging() -> None:
    os.environ.setdefault("MLFLOW_CONFIGURE_LOGGING", "false")
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = logging.getLevelNamesMapping().get(level_name, logging.INFO)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
    )
    handler = CurrentStdoutHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "mlflow"):
        library_logger = logging.getLogger(name)
        library_logger.handlers = []
        library_logger.propagate = True
    logging.getLogger("uvicorn.access").disabled = True
    logging.captureWarnings(True)


class DriftWindow:
    def __init__(self, size: int = DRIFT_WINDOW_SIZE) -> None:
        self._predictions: deque[tuple[bool, int]] = deque(maxlen=size)
        self._lock = threading.Lock()

    def add(self, is_spam: bool, text_length: int) -> None:
        with self._lock:
            self._predictions.append((is_spam, text_length))
            count = len(self._predictions)
            SPAM_SHARE.set(sum(spam for spam, _ in self._predictions) / count)
            INPUT_LENGTH_MEAN.set(sum(length for _, length in self._predictions) / count)


drift_window = DriftWindow()


def record_prediction(prediction: str, model_version: str, text_length: int) -> None:
    PREDICTIONS.labels(prediction=prediction, model_version=model_version).inc()
    drift_window.add(prediction == "spam", text_length)


def metrics_response(model_version: str | None) -> Response:
    MODEL_INFO.clear()
    if model_version is not None:
        MODEL_INFO.labels(model_version=model_version).set(1)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def request_id_for(request: Request) -> str:
    incoming = request.headers.get(REQUEST_ID_HEADER, "")
    if VALID_REQUEST_ID.fullmatch(incoming):
        return incoming
    return uuid.uuid4().hex[:12]


def route_path(request: Request) -> str:
    return getattr(request.scope.get("route"), "path", "unmatched")


def record_request(request: Request, status: int, seconds: float) -> None:
    path = route_path(request)
    if path == METRICS_PATH:
        return
    HTTP_REQUESTS.labels(method=request.method, path=path, status=str(status)).inc()
    HTTP_DURATION.labels(path=path).observe(seconds)
    write = log.error if status >= 500 else log.info
    write(
        "request",
        method=request.method,
        path=request.url.path,
        status=status,
        duration_ms=round(seconds * 1000, 2),
    )


def add_request_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def observe_request(request: Request, call_next):
        request_id = request_id_for(request)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            record_request(request, status, time.perf_counter() - start)
            structlog.contextvars.clear_contextvars()
