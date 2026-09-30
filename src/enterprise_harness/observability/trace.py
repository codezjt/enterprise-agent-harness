from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class SpanStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class SpanType(str, Enum):
    AGENT = "agent"
    PLANNER = "planner"
    LLM = "llm"
    TOOL = "tool"
    RAG = "rag"
    MEMORY = "memory"
    REPLAN = "replan"
    APPROVAL = "approval"


@dataclass
class TraceSpan:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    run_id: str

    component: SpanType
    name: str = ""

    start_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    end_time: datetime | None = None

    status: SpanStatus = SpanStatus.RUNNING

    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def finish(
        self,
        *,
        status: SpanStatus = SpanStatus.SUCCESS,
        output: Any = None,
        error: str | None = None,
    ) -> None:
        self.status = status
        self.output = output
        self.end_time = datetime.now(timezone.utc)
        if error is not None:
            self.metadata["error"] = error

    @property
    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "run_id": self.run_id,
            "component": self.component.value,
            "name": self.name,
            "start_time": self.start_time.isoformat(),
            "end_time": (
                self.end_time.isoformat()
                if self.end_time is not None
                else None
            ),
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
        }
