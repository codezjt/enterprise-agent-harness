import pytest

from enterprise_harness.gateway import (
    ToolDefinition,
    ToolGateway,
    ToolRegistry,
    ToolValidationError,
)


def query_order(order_id: str):
    return {
        "order_id": order_id,
        "status": "CREATED",
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
                "required": ["order_id"],
            },
            handler=query_order,
        )
    )

    return ToolGateway(registry)


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
async def test_gateway_invalid_arguments():
    gateway = create_gateway()

    with pytest.raises(ToolValidationError):
        await gateway.execute(
            tool_name="query_order",
            arguments={},
        )


@pytest.mark.asyncio
async def test_gateway_invalid_argument_type():
    gateway = create_gateway()

    with pytest.raises(ToolValidationError):
        await gateway.execute(
            tool_name="query_order",
            arguments={
                "order_id": 1001,
            },
        )