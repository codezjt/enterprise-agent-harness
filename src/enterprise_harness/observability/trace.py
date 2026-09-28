from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4
from contextlib import contextmanager
from collections.abc import Iterator


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

    start_time: datetime
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
    ) -> None:
        self.status = status
        self.output = output
        self.end_time = datetime.now(timezone.utc)

    @property
    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None

        return (
            self.end_time - self.start_time
        ).total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        data["component"] = self.component.value
        data["status"] = self.status.value
        data["start_time"] = self.start_time.isoformat()

        if self.end_time is not None:
            data["end_time"] = self.end_time.isoformat()

        data["duration_ms"] = self.duration_ms

        return data


class TraceManager:
    def __init__(self) -> None:
        self._spans: dict[str, TraceSpan] = {}
        self._span_stack: dict[str, list[TraceSpan]] = {}

    def start_span(
        self,
        *,
        run_id: str,
        component: SpanType,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceSpan:
        if trace_id is None:
            trace_id = str(uuid4())

        span = TraceSpan(
            trace_id=trace_id,
            span_id=str(uuid4()),
            parent_span_id=parent_span_id,
            run_id=run_id,
            component=component,
            start_time=datetime.now(timezone.utc),
            input=input,
            metadata=metadata or {},
        )

        self._spans[span.span_id] = span

        return span

    def finish_span(
        self,
        span_id: str,
        *,
        status: SpanStatus = SpanStatus.SUCCESS,
        output: Any = None,
    ) -> TraceSpan:
        span = self.get_span(span_id)

        span.finish(
            status=status,
            output=output,
        )

        return span

    def get_span(self, span_id: str) -> TraceSpan:
        try:
            return self._spans[span_id]
        except KeyError as exc:
            raise KeyError(
                f"Trace span not found: {span_id}"
            ) from exc

    def get_trace(self, trace_id: str) -> list[TraceSpan]:
        return [
            span
            for span in self._spans.values()
            if span.trace_id == trace_id
        ]

    def get_run_spans(self, run_id: str) -> list[TraceSpan]:
        return [
            span
            for span in self._spans.values()
            if span.run_id == run_id
        ]

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            span.to_dict()
            for span in self._spans.values()
        ]

    def clear(self) -> None:
        self._spans.clear()

    @contextmanager
    def span(
            self,
            *,
            run_id: str,
            component: SpanType,
            trace_id: str | None = None,
            parent_span_id: str | None = None,
            input: Any = None,
            metadata: dict[str, Any] | None = None,
    ) -> Iterator[TraceSpan]:
        if parent_span_id is None:
            parent = self.current_span(run_id)
            if parent is not None:
                parent_span_id = parent.span_id
                trace_id = parent.trace_id

        if trace_id is None:
            trace_id = str(uuid4())

        current = self.start_span(
            run_id=run_id,
            component=component,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            input=input,
            metadata=metadata,
        )

        self._span_stack.setdefault(run_id, []).append(current)

        try:
            yield current
        except Exception as exc:
            self.finish_span(
                current.span_id,
                status=SpanStatus.FAILED,
                output={"error": str(exc)},
            )
            raise
        else:
            if current.status == SpanStatus.RUNNING:
                self.finish_span(
                    current.span_id,
                    status=SpanStatus.SUCCESS,
                )
        finally:
            stack = self._span_stack.get(run_id)

            if stack:
                stack.pop()

                if not stack:
                    self._span_stack.pop(run_id, None)

    def current_span(self, run_id: str) -> TraceSpan | None:
        stack = self._span_stack.get(run_id)

        if not stack:
            return None

        return stack[-1]