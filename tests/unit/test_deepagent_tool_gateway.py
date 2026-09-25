import pytest

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
        "quantity": 100,
        "status": "PROCESSING",
    }


def create_gateway() -> ToolGateway:
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="query_order",
            description="查询订单信息",
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

    router = ToolRouter(
        registry=registry,
    )

    validator = ToolValidator()

    executor = ToolExecutor()

    rbac = RBAC(
        roles=DEFAULT_ROLES,
    )

    policy_engine = PolicyEngine(
        rbac=rbac,
    )

    return ToolGateway(
        registry=registry,
        router=router,
        validator=validator,
        executor=executor,
        policy_engine=policy_engine,
    )


@pytest.mark.asyncio
async def test_query_order_through_gateway():
    gateway = create_gateway()

    principal = Principal(
        principal_id="user-001",
        role="viewer",
    )

    result = await gateway.execute(
        tool_name="query_order",
        arguments={
            "order_id": "O1001",
        },
        principal=principal,
    )

    assert result == {
        "order_id": "O1001",
        "quantity": 100,
        "status": "PROCESSING",
    }


@pytest.mark.asyncio
async def test_query_order_denied_without_permission():
    gateway = create_gateway()

    principal = Principal(
        principal_id="user-001",
        role="operator",
    )

    # operator 实际拥有 order:read，
    # 因此这里应该正常执行。
    result = await gateway.execute(
        tool_name="query_order",
        arguments={
            "order_id": "O1001",
        },
        principal=principal,
    )

    assert result["order_id"] == "O1001"


@pytest.mark.asyncio
async def test_query_order_denied_without_principal():
    gateway = create_gateway()

    with pytest.raises(PermissionError):
        await gateway.execute(
            tool_name="query_order",
            arguments={
                "order_id": "O1001",
            },
        )


@pytest.mark.asyncio
async def test_query_order_invalid_arguments():
    gateway = create_gateway()

    principal = Principal(
        principal_id="user-001",
        role="viewer",
    )

    with pytest.raises(ValueError):
        await gateway.execute(
            tool_name="query_order",
            arguments={},
            principal=principal,
        )