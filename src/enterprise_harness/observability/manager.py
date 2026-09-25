from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .trace import SpanStatus, SpanType, TraceSpan


class TraceManager:

    def __init__(self):
        self._spans: dict[str, TraceSpan] = {}

    def start_run_span(
        self,
        run_id: str,
        agent_name: str,
        *,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceSpan:

        return self.start_span(
            run_id=run_id,
            span_type=SpanType.AGENT,
            name=agent_name,
            input=input,
            metadata=metadata,
        )

    def start_span(
        self,
        run_id: str,
        span_type: SpanType,
        name: str,
        *,
        parent_span_id: str | None = None,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceSpan:

        span = TraceSpan(
            run_id=run_id,
            parent_span_id=parent_span_id,
            span_type=span_type,
            name=name,
            input=input,
            metadata=metadata or {},
        )

        self._spans[span.span_id] = span

        return span

    def finish_span(
        self,
        span_id: str,
        *,
        output: Any | None = None,
        status: SpanStatus = SpanStatus.SUCCESS,
        error: str | None = None,
    ) -> TraceSpan:

        span = self.get_span(span_id)

        span.output = output
        span.status = status
        span.error = error
        span.ended_at = datetime.now(timezone.utc)

        return span

    def fail_span(
        self,
        span_id: str,
        error: Exception | str,
    ) -> TraceSpan:

        return self.finish_span(
            span_id,
            status=SpanStatus.FAILED,
            error=str(error),
        )

    def get_span(
        self,
        span_id: str,
    ) -> TraceSpan:

        span = self._spans.get(span_id)

        if span is None:
            raise KeyError(
                f"Trace span not found: {span_id}"
            )

        return span

    def get_run_spans(
        self,
        run_id: str,
    ) -> list[TraceSpan]:

        return [
            span
            for span in self._spans.values()
            if span.run_id == run_id
        ]

    def get_children(
        self,
        parent_span_id: str,
    ) -> list[TraceSpan]:

        return [
            span
            for span in self._spans.values()
            if span.parent_span_id == parent_span_id
        ]

    def clear(self) -> None:
        self._spans.clear()