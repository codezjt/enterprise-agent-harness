from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class SpanType(str, Enum):
    AGENT = "AGENT"
    PLANNER = "PLANNER"
    LLM = "LLM"
    TOOL = "TOOL"
    RAG = "RAG"
    MEMORY = "MEMORY"
    REPLAN = "REPLAN"
    APPROVAL = "APPROVAL"


class SpanStatus(str, Enum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class TraceSpan(BaseModel):
    span_id: str = Field(default_factory=lambda: str(uuid4()))

    run_id: str
    parent_span_id: str | None = None

    span_type: SpanType
    name: str

    status: SpanStatus = SpanStatus.RUNNING

    input: Any | None = None
    output: Any | None = None
    error: str | None = None

    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    ended_at: datetime | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        if self.ended_at is None:
            return None

        return (
            self.ended_at - self.started_at
        ).total_seconds() * 1000