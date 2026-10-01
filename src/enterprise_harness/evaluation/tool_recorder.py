from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from enterprise_harness.observability.trace import SpanType, TraceSpan


@dataclass
class RecordedToolCall:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None
    output: Any = None

    @property
    def name(self) -> str:
        return self.tool_name


class ToolCallRecorder:

    def __init__(self, trace_manager=None) -> None:
        self._trace_manager = trace_manager
        self._captured: dict[str, list[RecordedToolCall]] = {}
        self._cleared: set[str] = set()

    def record_from_span(self, run_id: str, span: TraceSpan) -> None:
        if span.component != SpanType.TOOL:
            return
        call = RecordedToolCall(
            tool_name=span.name,
            arguments=span.input if isinstance(span.input, dict) else {},
            success=span.status.value != "failed",
            error=span.metadata.get("error"),
            output=span.output,
        )
        self._captured.setdefault(run_id, []).append(call)

    def extract_from_traces(self, run_id: str) -> list[RecordedToolCall]:
        if run_id in self._cleared:
            return []
        if run_id in self._captured:
            return list(self._captured[run_id])

        if self._trace_manager is None:
            return []

        spans = self._trace_manager.get_run_spans(run_id)
        calls: list[RecordedToolCall] = []
        for span in spans:
            if span.component == SpanType.TOOL:
                calls.append(RecordedToolCall(
                    tool_name=span.name,
                    arguments=span.input if isinstance(span.input, dict) else {},
                    success=span.status.value != "failed",
                    error=span.metadata.get("error"),
                    output=span.output,
                ))
        self._captured[run_id] = calls
        return calls

    def get_tool_names(self, run_id: str) -> list[str]:
        calls = self.extract_from_traces(run_id)
        return [c.tool_name for c in calls]

    def get_recorded_calls(self, run_id: str) -> list[RecordedToolCall]:
        return self.extract_from_traces(run_id)

    def clear(self, run_id: str | None = None) -> None:
        if run_id is not None:
            self._captured.pop(run_id, None)
            self._cleared.add(run_id)
        else:
            self._captured.clear()
            self._cleared.clear()