# SMS spam detector: FastAPI app that serves the approved model from the MLflow registry.
# No secrets or data in the image. Configure at runtime with MLFLOW_TRACKING_URI,
# MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD, MODEL_NAME, MODEL_ALIAS and PORT.

FROM python:3.11-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

FROM python:3.11-slim AS runtime
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

ARG GIT_SHA=unknown
ENV GIT_SHA=$GIT_SHA
LABEL org.opencontainers.image.revision=$GIT_SHA

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app
USER app
WORKDIR /app
COPY --chown=app:app app/ ./app/

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import os, sys, urllib.request; url = f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health/ready'; sys.exit(0 if urllib.request.urlopen(url, timeout=4).status == 200 else 1)"]
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
