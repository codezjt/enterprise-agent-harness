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

from enterprise_harness.gateway.exceptions import (
    ApprovalRequiredError,
)


def query_order(order_id: str):
    return {
        "order_id": order_id,
        "status": "CREATED",
    }


def update_order(order_id: str, quantity: int):
    return {
        "order_id": order_id,
        "quantity": quantity,
        "updated": True,
    }


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
) -> ToolGateway:
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
            name="update_order",
            description="修改订单",
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                    },
                    "quantity": {
                        "type": "integer",
                    },
                },
                "required": [
                    "order_id",
                    "quantity",
                ],
            },
            handler=update_order,
        )
    )

    return ToolGateway(
        registry=registry,
        router=FakeRouter(registry),
        validator=FakeValidator(),
        executor=FakeExecutor(),
        policy_engine=policy_engine or PolicyEngine(),
    )


@pytest.mark.asyncio
async def test_gateway_execute():
    gateway = create_gateway()

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


@pytest.mark.asyncio
async def test_gateway_unknown_tool():
    gateway = create_gateway()

    with pytest.raises(KeyError):
        await gateway.execute(
            tool_name="unknown_tool",
            arguments={},
        )


@pytest.mark.asyncio
async def test_gateway_policy_allow():
    policy_engine = PolicyEngine(
        rules=[
            PolicyRule(
                tool_name="query_order",
                decision=PolicyDecision.ALLOW,
            )
        ]
    )

    gateway = create_gateway(policy_engine)

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


@pytest.mark.asyncio
async def test_gateway_policy_deny():
    policy_engine = PolicyEngine(
        rules=[
            PolicyRule(
                tool_name="update_order",
                decision=PolicyDecision.DENY,
            )
        ]
    )

    gateway = create_gateway(policy_engine)

    with pytest.raises(
        PermissionError,
        match="denied",
    ):
        await gateway.execute(
            tool_name="update_order",
            arguments={
                "order_id": "1001",
                "quantity": 80,
            },
        )


@pytest.mark.asyncio
async def test_gateway_policy_require_approval():

    policy_engine = PolicyEngine(
        rules=[
            PolicyRule(
                tool_name="update_order",
                decision=PolicyDecision.REQUIRE_APPROVAL,
            )
        ]
    )

    gateway = create_gateway(policy_engine)

    with pytest.raises(
        ApprovalRequiredError,
        match="requires approval",
    ):
        await gateway.execute(
            tool_name="update_order",
            arguments={
                "order_id": "1001",
                "quantity": 80,
            },
            run_id="run-001",
        )

@pytest.mark.asyncio
async def test_gateway_executes_after_approval():

    policy_engine = PolicyEngine(
        rules=[
            PolicyRule(
                tool_name="update_order",
                decision=PolicyDecision.REQUIRE_APPROVAL,
            )
        ]
    )

    gateway = create_gateway(policy_engine)

    arguments = {
        "order_id": "1001",
        "quantity": 80,
    }

    with pytest.raises(ApprovalRequiredError) as exc_info:

        await gateway.execute(
            tool_name="update_order",
            arguments=arguments,
            run_id="run-001",
        )

    approval_id = exc_info.value.approval_id

    approval = await gateway.approval_manager.approve(
        approval_id=approval_id,
        comment="approved",
    )

    assert approval.approval_id == approval_id

    result = await gateway.execute(
        tool_name="update_order",
        arguments=arguments,
        run_id="run-001",
        approval_id=approval_id,
    )

    assert result is not None

@pytest.mark.asyncio
async def test_gateway_rejects_unknown_approval():

    policy_engine = PolicyEngine(
        rules=[
            PolicyRule(
                tool_name="update_order",
                decision=PolicyDecision.REQUIRE_APPROVAL,
            )
        ]
    )

    gateway = create_gateway(policy_engine)

    arguments = {
        "order_id": "1001",
        "quantity": 80,
    }

    # 1. 不存在的 approval_id
    with pytest.raises(KeyError):
        await gateway.execute(
            tool_name="update_order",
            arguments=arguments,
            run_id="run-001",
            approval_id="approval-not-exist",
        )

    # 2. 创建真实审批请求
    with pytest.raises(ApprovalRequiredError) as exc_info:
        await gateway.execute(
            tool_name="update_order",
            arguments=arguments,
            run_id="run-001",
        )

    approval_id = exc_info.value.approval_id

    # 3. 审批通过
    await gateway.approval_manager.approve(
        approval_id
    )

    # 4. 把同一个 approval 用到其他 Run
    with pytest.raises(
        PermissionError,
        match="does not belong to this run",
    ):
        await gateway.execute(
            tool_name="update_order",
            arguments=arguments,
            run_id="run-002",
            approval_id=approval_id,
        )