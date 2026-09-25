import pytest

from enterprise_harness.gateway import (
    ToolDefinition,
    ToolExecutor,
    ToolGateway,
    ToolRegistry,
    ToolRouter,
    ToolValidator,
)
from enterprise_harness.observability import (
    SpanStatus,
    SpanType,
    TraceManager,
)
from enterprise_harness.policy import PolicyEngine


@pytest.mark.asyncio
async def test_tool_gateway_creates_success_trace():

    async def query_order(order_id: str):
        return {
            "order_id": order_id,
            "status": "PAID",
        }

    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="query_order",
            description="Query order",
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                    },
                },
                "required": [
                    "order_id",
                ],
            },
            handler=query_order,
        )
    )

    trace_manager = TraceManager()

    gateway = ToolGateway(
        registry=registry,
        router=ToolRouter(registry),
        validator=ToolValidator(),
        executor=ToolExecutor(),
        policy_engine=PolicyEngine(),
        trace_manager=trace_manager,
    )

    result = await gateway.execute(
        tool_name="query_order",
        arguments={
            "order_id": "ORD001",
        },
        run_id="run-001",
    )

    assert result["order_id"] == "ORD001"

    spans = trace_manager.get_run_spans(
        "run-001"
    )

    assert len(spans) == 1

    span = spans[0]

    assert span.span_type == SpanType.TOOL
    assert span.name == "query_order"
    assert span.status == SpanStatus.SUCCESS
    assert span.output["order_id"] == "ORD001"


@pytest.mark.asyncio
async def test_tool_gateway_creates_failed_trace():

    async def query_order(order_id: str):
        raise RuntimeError("database unavailable")

    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="query_order",
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                    },
                },
                "required": [
                    "order_id",
                ],
            },
            handler=query_order,
        )
    )

    trace_manager = TraceManager()

    gateway = ToolGateway(
        registry=registry,
        router=ToolRouter(registry),
        validator=ToolValidator(),
        executor=ToolExecutor(),
        policy_engine=PolicyEngine(),
        trace_manager=trace_manager,
    )

    with pytest.raises(RuntimeError):

        await gateway.execute(
            tool_name="query_order",
            arguments={
                "order_id": "ORD001",
            },
            run_id="run-001",
        )

    spans = trace_manager.get_run_spans(
        "run-001"
    )

    assert len(spans) == 1

    span = spans[0]

    assert span.status == SpanStatus.FAILED
    assert span.error == "database unavailable"