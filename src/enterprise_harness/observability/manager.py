from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator
from uuid import uuid4

from .trace import SpanStatus, SpanType, TraceSpan
from .metrics import MetricCollector, MetricSnapshot
from .audit import AuditEvent, AuditLogger
from .cost import CostTracker


@dataclass
class ObservabilitySnapshot:
    run_id: str | None = None
    tenant_id: str | None = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    traces: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] | None = None
    audits: list[dict[str, Any]] = field(default_factory=list)
    cost: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "tenant_id": self.tenant_id,
            "timestamp": self.timestamp.isoformat(),
            "traces": self.traces,
            "metrics": self.metrics,
            "audits": self.audits,
            "cost": self.cost,
        }


class ObservabilityManager:

    def __init__(
        self,
        trace_manager: TraceManager | None = None,
        metric_collector: MetricCollector | None = None,
        audit_logger: AuditLogger | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        self.trace = trace_manager or TraceManager()
        self.metrics = metric_collector or MetricCollector()
        self.audit = audit_logger or AuditLogger()
        self.cost = cost_tracker

    def _ensure_dict(self, value: Any) -> dict[str, Any]:
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if isinstance(value, dict):
            return value
        return {"value": str(value)}

    def snapshot(
        self,
        *,
        run_id: str | None = None,
        tenant_id: str | None = None,
    ) -> ObservabilitySnapshot:
        traces = self.trace.get_run_spans(run_id) if run_id else self.trace.snapshot()

        audits = (
            self.audit.get_run_events(run_id) if run_id
            else self.audit.get_events(tenant_id=tenant_id)
        )

        return ObservabilitySnapshot(
            run_id=run_id,
            tenant_id=tenant_id,
            traces=[self._ensure_dict(t) for t in traces],
            metrics=self.metrics.snapshot().to_dict(),
            audits=[a.to_dict() for a in audits],
            cost=self.cost.snapshot() if self.cost else None,
        )

    def clear(self) -> None:
        self.trace.clear()
        self.metrics.reset()
        self.audit.clear()
        if self.cost:
            self.cost.reset()


class TraceManager:

    def __init__(self) -> None:
        self._spans: dict[str, TraceSpan] = {}
        self._span_stack: dict[str, list[TraceSpan]] = {}

    def start_run_span(
        self,
        run_id: str,
        agent_name: str,
        *,
        tenant_id: str | None = None,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceSpan:
        return self.start_span(
            run_id=run_id,
            span_type=SpanType.AGENT,
            name=agent_name,
            tenant_id=tenant_id,
            input=input,
            metadata=metadata,
        )

    def start_span(
        self,
        *,
        run_id: str,
        span_type: SpanType,
        name: str = "",
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        tenant_id: str | None = None,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceSpan:
        if parent_span_id is None:
            parent = self.current_span(run_id)
            if parent is not None:
                parent_span_id = parent.span_id
                if trace_id is None:
                    trace_id = parent.trace_id
                if tenant_id is None:
                    tenant_id = parent.tenant_id
        else:
            if tenant_id is None and parent_span_id in self._spans:
                tenant_id = self._spans[parent_span_id].tenant_id

        if trace_id is None:
            trace_id = str(uuid4())

        span = TraceSpan(
            trace_id=trace_id,
            span_id=str(uuid4()),
            parent_span_id=parent_span_id,
            run_id=run_id,
            tenant_id=tenant_id or "default",
            component=span_type,
            name=name,
            start_time=datetime.now(timezone.utc),
            input=input,
            metadata={
                **(metadata or {}),
            },
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
        span.finish(
            status=status,
            output=output,
            error=error,
        )
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
            raise KeyError(f"Trace span not found: {span_id}")
        return span

    def get_run_spans(
        self,
        run_id: str,
        tenant_id: str | None = None,
    ) -> list[TraceSpan]:
        spans = [
            span
            for span in self._spans.values()
            if span.run_id == run_id
        ]
        if tenant_id is not None:
            spans = [
                s for s in spans
                if s.tenant_id == tenant_id
            ]
        return spans

    def get_tenant_spans(
        self,
        tenant_id: str,
    ) -> list[TraceSpan]:
        return [
            span
            for span in self._spans.values()
            if span.tenant_id == tenant_id
        ]

    def get_trace(
        self,
        trace_id: str,
    ) -> list[TraceSpan]:
        return [
            span
            for span in self._spans.values()
            if span.trace_id == trace_id
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

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            span.to_dict()
            for span in self._spans.values()
        ]

    def clear(self) -> None:
        self._spans.clear()
        self._span_stack.clear()

    @contextmanager
    def span(
        self,
        *,
        run_id: str,
        span_type: SpanType,
        name: str = "",
        parent_span_id: str | None = None,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[TraceSpan]:
        current = self.start_span(
            run_id=run_id,
            span_type=span_type,
            name=name,
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
                error=str(exc),
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
