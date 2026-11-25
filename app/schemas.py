"""Request and response schemas of the inference API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Same bound as the data contract (ml/data_contract.py), so the app never sends
# the model a message longer than anything it could have been trained on.
MAX_TEXT_LENGTH = 1000


class PredictRequest(BaseModel):
    """One SMS message to classify."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="The message, 1 to 1000 characters after trimming whitespace.",
        examples=["Are we still meeting for lunch tomorrow?"],
    )


class PredictResponse(BaseModel):
    prediction: Literal["ham", "spam"]
    spam_probability: float | None = Field(
        description="Probability that the message is spam, when the model provides one."
    )
    model_version: str = Field(description="Registry version of the model that answered.")


class ModelInfo(BaseModel):
    """Identity of the model currently loaded. Field names are part of the deploy contract."""

    name: str
    alias: str
    version: str
    run_id: str | None
    dataset_version: str | None
    git_sha: str | None
    loaded_at: datetime


class Liveness(BaseModel):
    status: Literal["alive"]


class Readiness(BaseModel):
    status: Literal["ready", "not_ready"]
    model_version: str | None = None
    reason: str | None = None
