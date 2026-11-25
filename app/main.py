"""SMS spam detector: web form and inference API.

uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.model_loader import ModelNotReady, ModelService
from app.schemas import (
    MAX_TEXT_LENGTH,
    Liveness,
    ModelInfo,
    PredictRequest,
    PredictResponse,
    Readiness,
)


def configure_logging() -> None:
    """Send the app's own logs to stdout as an event stream.

    The ``app`` logger gets its own handler instead of relying on the root
    logger, which libraries may reconfigure after start-up.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    app_logger = logging.getLogger("app")
    app_logger.handlers = [handler]
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False


configure_logging()

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

SAMPLE_INPUTS = [
    {
        "label": "Lunch plans",
        "text": "Are we still meeting for lunch tomorrow? I can book a table for 1pm.",
    },
    {
        "label": "Running late",
        "text": "Sorry, my train is delayed. I'll be home around 8, start dinner without me.",
    },
    {
        "label": "Prize draw",
        "text": "WINNER!! You have been selected to receive a £1000 cash prize. "
        "To claim call 09061701461 now. Claim code KL341. Valid 12 hours only.",
    },
    {
        "label": "Free ringtone",
        "text": "FREE entry in our weekly competition! Text WIN to 80086 to get a free ringtone. "
        "Std txt rate. T&Cs apply. To stop reply STOP.",
    },
]


def create_app(load_in_background: bool = True) -> FastAPI:
    """Build the application.

    In production the model loads in a background thread, so the process is
    live immediately and becomes ready once a test prediction has succeeded.
    Tests pass ``load_in_background=False`` to load before the first request.
    """
    service = ModelService()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        stop = threading.Event()
        if load_in_background:
            threading.Thread(
                target=service.load_until_ready, args=(stop,), name="model-loader", daemon=True
            ).start()
        else:
            service.load()
        yield
        stop.set()

    app = FastAPI(title="SMS spam detector", lifespan=lifespan)
    app.state.model_service = service

    @app.exception_handler(ModelNotReady)
    async def model_not_ready(_: Request, error: ModelNotReady) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": f"Model not ready: {error}"})

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "index.html", {"samples": SAMPLE_INPUTS, "max_length": MAX_TEXT_LENGTH}
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest) -> PredictResponse:
        loaded = service.require()
        try:
            label, probability = loaded.predict(body.text)
        except Exception as error:
            logging.getLogger("app").exception("Prediction failed")
            raise HTTPException(status_code=503, detail="The model could not answer.") from error
        return PredictResponse(
            prediction=label, spam_probability=probability, model_version=loaded.version
        )

    @app.get("/model-info", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        loaded = service.require()
        return ModelInfo(
            name=loaded.name,
            alias=loaded.alias,
            version=loaded.version,
            run_id=loaded.run_id,
            dataset_version=loaded.dataset_version,
            git_sha=loaded.git_sha,
            loaded_at=loaded.loaded_at,
            app_git_sha=os.environ.get("GIT_SHA") or "unknown",
        )

    @app.get("/health/live", response_model=Liveness)
    def live() -> Liveness:
        return Liveness(status="alive")

    @app.get("/health/ready", response_model=Readiness, response_model_exclude_none=True)
    def ready():
        loaded = service.loaded
        if loaded is None:
            body = Readiness(status="not_ready", reason=service.not_ready_reason)
            return JSONResponse(status_code=503, content=body.model_dump(exclude_none=True))
        return Readiness(status="ready", model_version=loaded.version)

    return app


app = create_app()
