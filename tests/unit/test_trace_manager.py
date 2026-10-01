from enterprise_harness.observability import (
    SpanStatus,
    SpanType,
    TraceManager,
)


def test_start_and_finish_span():

    manager = TraceManager()

    span = manager.start_span(
        run_id="run-001",
        span_type=SpanType.TOOL,
        name="query_order",
        input={
            "order_id": "ORD001",
        },
    )

    assert span.run_id == "run-001"
    assert span.component == SpanType.TOOL
    assert span.name == "query_order"
    assert span.status == SpanStatus.RUNNING
    assert span.end_time is None

    finished = manager.finish_span(
        span.span_id,
        output={
            "order_id": "ORD001",
            "status": "PAID",
        },
    )

    assert finished.status == SpanStatus.SUCCESS
    assert finished.output["status"] == "PAID"
    assert finished.end_time is not None
    assert finished.duration_ms is not None


def test_failed_span():

    manager = TraceManager()

    span = manager.start_span(
        run_id="run-001",
        span_type=SpanType.TOOL,
        name="update_order",
    )

    manager.fail_span(
        span.span_id,
        RuntimeError("database error"),
    )

    result = manager.get_span(span.span_id)

    assert result.status == SpanStatus.FAILED
    assert result.metadata["error"] == "database error"


def test_parent_child_span():

    manager = TraceManager()

    parent = manager.start_span(
        run_id="run-001",
        span_type=SpanType.AGENT,
        name="order_agent",
    )

    child = manager.start_span(
        run_id="run-001",
        span_type=SpanType.TOOL,
        name="query_order",
        parent_span_id=parent.span_id,
    )

    children = manager.get_children(
        parent.span_id
    )

    assert len(children) == 1
    assert children[0].span_id == child.span_id


def test_get_run_spans():

    manager = TraceManager()

    manager.start_span(
        run_id="run-001",
        span_type=SpanType.AGENT,
        name="agent",
    )

    manager.start_span(
        run_id="run-001",
        span_type=SpanType.TOOL,
        name="query_order",
    )

    manager.start_span(
        run_id="run-002",
        span_type=SpanType.AGENT,
        name="other-agent",
    )

    spans = manager.get_run_spans("run-001")

    assert len(spans) == 2