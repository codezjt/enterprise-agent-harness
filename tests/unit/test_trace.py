import pytest

from enterprise_harness.observability import (
    SpanStatus,
    SpanType,
    TraceManager,
)


def test_trace_manager_creates_root_span():
    manager = TraceManager()

    span = manager.start_span(
        run_id="run-001",
        span_type=SpanType.AGENT,
        input={"task": "test"},
    )

    assert span.trace_id
    assert span.span_id
    assert span.parent_span_id is None
    assert span.run_id == "run-001"
    assert span.component == SpanType.AGENT
    assert span.status == SpanStatus.RUNNING
    assert span.end_time is None


def test_trace_manager_creates_child_span():
    manager = TraceManager()

    root = manager.start_span(
        run_id="run-001",
        span_type=SpanType.AGENT,
    )

    child = manager.start_span(
        run_id="run-001",
        trace_id=root.trace_id,
        parent_span_id=root.span_id,
        span_type=SpanType.TOOL,
        input={"tool": "query_order"},
    )

    assert child.trace_id == root.trace_id
    assert child.parent_span_id == root.span_id
    assert child.component == SpanType.TOOL


def test_trace_span_can_be_finished():
    manager = TraceManager()

    span = manager.start_span(
        run_id="run-001",
        span_type=SpanType.LLM,
    )

    result = manager.finish_span(
        span.span_id,
        status=SpanStatus.SUCCESS,
        output={"answer": "ok"},
    )

    assert result is span
    assert span.status == SpanStatus.SUCCESS
    assert span.output == {"answer": "ok"}
    assert span.end_time is not None
    assert span.duration_ms is not None
    assert span.duration_ms >= 0


def test_trace_span_can_be_failed():
    manager = TraceManager()

    span = manager.start_span(
        run_id="run-001",
        span_type=SpanType.TOOL,
    )

    manager.finish_span(
        span.span_id,
        status=SpanStatus.FAILED,
        error="timeout",
    )

    assert span.status == SpanStatus.FAILED
    assert span.metadata["error"] == "timeout"


def test_get_trace_returns_same_trace_spans():
    manager = TraceManager()

    root = manager.start_span(
        run_id="run-001",
        span_type=SpanType.AGENT,
    )

    manager.start_span(
        run_id="run-001",
        trace_id=root.trace_id,
        parent_span_id=root.span_id,
        span_type=SpanType.PLANNER,
    )

    manager.start_span(
        run_id="run-001",
        trace_id=root.trace_id,
        parent_span_id=root.span_id,
        span_type=SpanType.LLM,
    )

    other = manager.start_span(
        run_id="run-002",
        span_type=SpanType.AGENT,
    )

    trace = manager.get_trace(root.trace_id)

    assert len(trace) == 3
    assert all(span.trace_id == root.trace_id for span in trace)
    assert other not in trace


def test_get_run_spans():
    manager = TraceManager()

    manager.start_span(
        run_id="run-001",
        span_type=SpanType.AGENT,
    )

    manager.start_span(
        run_id="run-001",
        span_type=SpanType.TOOL,
    )

    manager.start_span(
        run_id="run-002",
        span_type=SpanType.AGENT,
    )

    spans = manager.get_run_spans("run-001")

    assert len(spans) == 2
    assert all(span.run_id == "run-001" for span in spans)


def test_trace_span_to_dict():
    manager = TraceManager()

    span = manager.start_span(
        run_id="run-001",
        span_type=SpanType.RAG,
        input={"query": "订单异常"},
        metadata={"top_k": 5},
    )

    manager.finish_span(
        span.span_id,
        output={"documents": 3},
    )

    data = span.to_dict()

    assert data["trace_id"] == span.trace_id
    assert data["span_id"] == span.span_id
    assert data["parent_span_id"] is None
    assert data["run_id"] == "run-001"
    assert data["component"] == "rag"
    assert data["status"] == "success"
    assert data["input"] == {"query": "订单异常"}
    assert data["output"] == {"documents": 3}
    assert data["metadata"] == {"top_k": 5}
    assert data["start_time"]
    assert data["end_time"]
    assert data["duration_ms"] >= 0


def test_trace_manager_rejects_unknown_span():
    manager = TraceManager()

    try:
        manager.get_span("not-exists")
        assert False
    except KeyError as exc:
        assert "Trace span not found" in str(exc)


def test_trace_manager_clear():
    manager = TraceManager()

    manager.start_span(
        run_id="run-001",
        span_type=SpanType.AGENT,
    )

    assert len(manager.snapshot()) == 1

    manager.clear()

    assert manager.snapshot() == []


def test_nested_spans_build_parent_relationship():
    manager = TraceManager()

    with manager.span(
        run_id="run-001",
        span_type=SpanType.AGENT,
    ) as agent:

        with manager.span(
            run_id="run-001",
            span_type=SpanType.PLANNER,
        ) as planner:

            with manager.span(
                run_id="run-001",
                span_type=SpanType.LLM,
            ) as llm:

                assert manager.current_span("run-001") is llm

            assert manager.current_span("run-001") is planner

        assert manager.current_span("run-001") is agent

    assert manager.current_span("run-001") is None

    assert planner.parent_span_id == agent.span_id
    assert llm.parent_span_id == planner.span_id

    assert agent.trace_id == planner.trace_id
    assert planner.trace_id == llm.trace_id


def test_nested_span_is_finished_automatically():
    manager = TraceManager()

    with manager.span(
        run_id="run-001",
        span_type=SpanType.TOOL,
    ) as span:
        assert span.status == SpanStatus.RUNNING

    assert span.status == SpanStatus.SUCCESS
    assert span.end_time is not None


def test_nested_span_is_failed_when_exception_occurs():
    manager = TraceManager()

    with pytest.raises(RuntimeError):
        with manager.span(
            run_id="run-001",
            span_type=SpanType.TOOL,
        ) as span:
            raise RuntimeError("tool timeout")

    assert span.status == SpanStatus.FAILED
    assert span.end_time is not None
    assert span.metadata["error"] == "tool timeout"


def test_span_stack_is_isolated_by_run_id():
    manager = TraceManager()

    with manager.span(
        run_id="run-001",
        span_type=SpanType.AGENT,
    ) as first:

        with manager.span(
            run_id="run-002",
            span_type=SpanType.AGENT,
        ) as second:

            assert manager.current_span("run-001") is first
            assert manager.current_span("run-002") is second

            assert second.parent_span_id is None

        assert manager.current_span("run-001") is first
        assert manager.current_span("run-002") is None

    assert manager.current_span("run-001") is None