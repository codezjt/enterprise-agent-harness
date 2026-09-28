import pytest

from enterprise_harness.gateway import (
    ToolDefinition,
    ToolGateway,
    ToolRegistry,
)
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


class FakeAuditLogger:

    def __init__(self):
        self.records = []

    def record(
        self,
        *,
        event,
        user_id,
        agent_id,
        run_id,
        tool,
        arguments,
        decision,
        approver,
        result,
    ):
        self.records.append(
            {
                "event": event,
                "user_id": user_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "tool": tool,
                "arguments": arguments,
                "decision": decision,
                "approver": approver,
                "result": result,
            }
        )


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

    audit_logger = FakeAuditLogger()

    gateway = ToolGateway(
        registry=registry,
        router=FakeRouter(registry),
        validator=FakeValidator(),
        executor=FakeExecutor(),
        policy_engine=policy_engine or PolicyEngine(),
        audit_logger=audit_logger,
    )

    return gateway, audit_logger


@pytest.mark.asyncio
async def test_gateway_audit_success():
    gateway, audit_logger = create_gateway()

    result = await gateway.execute(
        tool_name="query_order",
        arguments={
            "order_id": "1001",
        },
        context={
            "agent_id": "test-agent",
        },
        run_id="run-001",
    )

    assert result == {
        "order_id": "1001",
        "status": "CREATED",
    }

    assert len(audit_logger.records) == 1

    record = audit_logger.records[0]

    assert record["event"] == "tool_execution"
    assert record["user_id"] == "anonymous"
    assert record["agent_id"] == "test-agent"
    assert record["run_id"] == "run-001"
    assert record["tool"] == "query_order"
    assert record["arguments"] == {
        "order_id": "1001",
    }
    assert record["decision"] == PolicyDecision.ALLOW.value
    assert record["result"]["success"] is True
    assert record["result"]["output"] == {
        "order_id": "1001",
        "status": "CREATED",
    }


@pytest.mark.asyncio
async def test_gateway_audit_policy_denied():
    policy_engine = PolicyEngine(
        rules=[
            PolicyRule(
                tool_name="query_order",
                decision=PolicyDecision.DENY,
            )
        ]
    )

    gateway, audit_logger = create_gateway(
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
            run_id="run-002",
        )

    assert len(audit_logger.records) == 1

    record = audit_logger.records[0]

    assert record["event"] == "tool_execution"
    assert record["run_id"] == "run-002"
    assert record["tool"] == "query_order"
    assert record["decision"] == PolicyDecision.DENY.value
    assert record["result"] == {
        "success": False,
        "error": "policy_denied",
    }


@pytest.mark.asyncio
async def test_gateway_audit_tool_failure():
    gateway, audit_logger = create_gateway()

    with pytest.raises(
        RuntimeError,
        match="tool execution failed",
    ):
        await gateway.execute(
            tool_name="failed_tool",
            arguments={
                "order_id": "1001",
            },
            context={
                "agent_id": "test-agent",
            },
            run_id="run-003",
        )

    assert len(audit_logger.records) == 1

    record = audit_logger.records[0]

    assert record["event"] == "tool_execution"
    assert record["agent_id"] == "test-agent"
    assert record["run_id"] == "run-003"
    assert record["tool"] == "failed_tool"
    assert record["decision"] == PolicyDecision.ALLOW.value
    assert record["result"]["success"] is False
    assert record["result"]["error"] == "tool execution failed"


@pytest.mark.asyncio
async def test_gateway_audit_require_approval():
    policy_engine = PolicyEngine(
        rules=[
            PolicyRule(
                tool_name="query_order",
                decision=PolicyDecision.REQUIRE_APPROVAL,
            )
        ]
    )

    gateway, audit_logger = create_gateway(
        policy_engine=policy_engine,
    )

    with pytest.raises(
        PermissionError,
        match="requires approval",
    ):
        await gateway.execute(
            tool_name="query_order",
            arguments={
                "order_id": "1001",
            },
            run_id="run-004",
        )

    assert len(audit_logger.records) == 1

    record = audit_logger.records[0]

    assert record["event"] == "tool_execution"
    assert record["run_id"] == "run-004"
    assert record["tool"] == "query_order"
    assert record["decision"] == (
        PolicyDecision.REQUIRE_APPROVAL.value
    )
    assert record["result"] == {
        "success": False,
        "error": "approval_required",
    }