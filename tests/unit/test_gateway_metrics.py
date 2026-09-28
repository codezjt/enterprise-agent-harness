import pytest

from enterprise_harness.gateway import (
    ToolDefinition,
    ToolGateway,
    ToolRegistry,
)
from enterprise_harness.observability import MetricCollector
from enterprise_harness.policy import (
    PolicyDecision,
    PolicyEngine,
    PolicyRule,
)


def query_order(order_id: str):
    return {
        "order_id": order_id,
        "status": "CREATED",
    }


def failed_tool(order_id: str):
    raise RuntimeError("tool execution failed")


class FakeRouter:

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def route(self, tool_name: str) -> ToolDefinition:
        return self.registry.get(tool_name)


class FakeValidator:

    def validate(
        self,
        tool: ToolDefinition,
        arguments: dict,
    ) -> dict:
        return arguments


class FakeExecutor:

    async def execute(
        self,
        tool: ToolDefinition,
        arguments: dict,
    ):
        return tool.handler(**arguments)


def create_gateway(
    policy_engine: PolicyEngine | None = None,
):
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="query_order",
            description="查询订单",
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                    },
                },
                "required": ["order_id"],
            },
            handler=query_order,
        )
    )

    registry.register(
        ToolDefinition(
            name="failed_tool",
            description="执行失败工具",
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                    },
                },
                "required": ["order_id"],
            },
            handler=failed_tool,
        )
    )

    metric_collector = MetricCollector()

    gateway = ToolGateway(
        registry=registry,
        router=FakeRouter(registry),
        validator=FakeValidator(),
        executor=FakeExecutor(),
        policy_engine=policy_engine or PolicyEngine(),
        metric_collector=metric_collector,
    )

    return gateway, metric_collector


@pytest.mark.asyncio
async def test_gateway_metrics_success():
    gateway, metric_collector = create_gateway()

    result = await gateway.execute(
        tool_name="query_order",
        arguments={
            "order_id": "1001",
        },
    )

    assert result == {
        "order_id": "1001",
        "status": "CREATED",
    }

    snapshot = metric_collector.snapshot()

    assert snapshot.tool_calls == 1
    assert snapshot.tool_successes == 1
    assert snapshot.tool_success_rate == 1.0
    assert snapshot.policy_denied == 0


@pytest.mark.asyncio
async def test_gateway_metrics_tool_failure():
    gateway, metric_collector = create_gateway()

    with pytest.raises(
        RuntimeError,
        match="tool execution failed",
    ):
        await gateway.execute(
            tool_name="failed_tool",
            arguments={
                "order_id": "1001",
            },
        )

    snapshot = metric_collector.snapshot()

    assert snapshot.tool_calls == 1
    assert snapshot.tool_successes == 0
    assert snapshot.tool_success_rate == 0.0
    assert snapshot.policy_denied == 0


@pytest.mark.asyncio
async def test_gateway_metrics_policy_denied():
    policy_engine = PolicyEngine(
        rules=[
            PolicyRule(
                tool_name="query_order",
                decision=PolicyDecision.DENY,
            )
        ]
    )

    gateway, metric_collector = create_gateway(
        policy_engine=policy_engine,
    )

    with pytest.raises(
        PermissionError,
        match="denied",
    ):
        await gateway.execute(
            tool_name="query_order",
            arguments={
                "order_id": "1001",
            },
        )

    snapshot = metric_collector.snapshot()

    assert snapshot.tool_calls == 0
    assert snapshot.tool_successes == 0
    assert snapshot.tool_success_rate == 0.0
    assert snapshot.policy_denied == 1