from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from enterprise_harness.agent import (
    AgentConfig,
    DeepAgentRuntime,
)
from enterprise_harness.gateway import (
    ToolDefinition,
    ToolExecutor,
    ToolGateway,
    ToolRegistry,
    ToolRouter,
    ToolValidator,
)
from enterprise_harness.policy import (
    DEFAULT_ROLES,
    PolicyEngine,
    Principal,
    RBAC,
)


async def query_order(
    order_id: str,
):
    return {
        "order_id": order_id,
        "status": "PROCESSING",
    }


def create_gateway() -> ToolGateway:
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
                "required": [
                    "order_id",
                ],
            },
            permissions=[
                "order:read",
            ],
            handler=query_order,
        )
    )

    rbac = RBAC(
        roles=DEFAULT_ROLES,
    )

    return ToolGateway(
        registry=registry,
        router=ToolRouter(registry),
        validator=ToolValidator(),
        executor=ToolExecutor(),
        policy_engine=PolicyEngine(
            rbac=rbac,
        ),
    )


@pytest.mark.asyncio
async def test_deepagent_runtime_builds_gateway_tools():
    gateway = create_gateway()

    config = AgentConfig(
        agent_id="order-agent",
        name="Order Agent",
        model="test-model",
        system_prompt="You are an order assistant.",
        tools=[
            "query_order",
        ],
    )

    fake_agent = MagicMock()

    fake_agent.ainvoke = AsyncMock(
        return_value={
            "messages": [],
        }
    )

    with patch(
        "enterprise_harness.agent.deepagent_runtime.create_deep_agent",
        return_value=fake_agent,
    ):
        runtime = DeepAgentRuntime(
            config=config,
            tool_gateway=gateway,
            principal=Principal(
                principal_id="user-001",
                role="viewer",
            ),
        )

    assert runtime.agent is fake_agent

    fake_agent.ainvoke.assert_not_called()