# import pytest
#
# from enterprise_harness.gateway import (
#     ToolDefinition,
#     ToolGateway,
#     ToolRegistry,
#     ToolValidationError,
# )
# from enterprise_harness.policy import (
#     PolicyDecision,
#     PolicyEngine,
#     PolicyRule,
# )
#
#
# def query_order(order_id: str):
#     return {
#         "order_id": order_id,
#         "status": "CREATED",
#     }
#
#
# def update_order(order_id: str, quantity: int):
#     return {
#         "order_id": order_id,
#         "quantity": quantity,
#         "updated": True,
#     }
#
#
# def create_gateway(
#     policy_engine: PolicyEngine | None = None,
# ) -> ToolGateway:
#     registry = ToolRegistry()
#
#     registry.register(
#         ToolDefinition(
#             name="query_order",
#             description="查询订单",
#             input_schema={
#                 "type": "object",
#                 "properties": {
#                     "order_id": {
#                         "type": "string",
#                     },
#                 },
#                 "required": ["order_id"],
#             },
#             handler=query_order,
#         )
#     )
#
#     registry.register(
#         ToolDefinition(
#             name="update_order",
#             description="修改订单",
#             input_schema={
#                 "type": "object",
#                 "properties": {
#                     "order_id": {
#                         "type": "string",
#                     },
#                     "quantity": {
#                         "type": "integer",
#                     },
#                 },
#                 "required": [
#                     "order_id",
#                     "quantity",
#                 ],
#             },
#             handler=update_order,
#         )
#     )
#
#     return ToolGateway(
#         registry=registry,
#         policy_engine=policy_engine,
#     )
#
#
# @pytest.mark.asyncio
# async def test_gateway_execute():
#     gateway = create_gateway()
#
#     result = await gateway.execute(
#         tool_name="query_order",
#         arguments={
#             "order_id": "1001",
#         },
#     )
#
#     assert result == {
#         "order_id": "1001",
#         "status": "CREATED",
#     }
#
#
# @pytest.mark.asyncio
# async def test_gateway_unknown_tool():
#     gateway = create_gateway()
#
#     with pytest.raises(KeyError):
#         await gateway.execute(
#             tool_name="unknown_tool",
#             arguments={},
#         )
#
#
# @pytest.mark.asyncio
# async def test_gateway_invalid_arguments():
#     gateway = create_gateway()
#
#     with pytest.raises(ToolValidationError):
#         await gateway.execute(
#             tool_name="query_order",
#             arguments={},
#         )
#
#
# @pytest.mark.asyncio
# async def test_gateway_invalid_argument_type():
#     gateway = create_gateway()
#
#     with pytest.raises(ToolValidationError):
#         await gateway.execute(
#             tool_name="query_order",
#             arguments={
#                 "order_id": 1001,
#             },
#         )
#
#
# @pytest.mark.asyncio
# async def test_gateway_policy_allow():
#     policy_engine = PolicyEngine(
#         rules=[
#             PolicyRule(
#                 tool_name="query_order",
#                 decision=PolicyDecision.ALLOW,
#             )
#         ]
#     )
#
#     gateway = create_gateway(policy_engine)
#
#     result = await gateway.execute(
#         tool_name="query_order",
#         arguments={
#             "order_id": "1001",
#         },
#     )
#
#     assert result == {
#         "order_id": "1001",
#         "status": "CREATED",
#     }
#
#
# @pytest.mark.asyncio
# async def test_gateway_policy_deny():
#     policy_engine = PolicyEngine(
#         rules=[
#             PolicyRule(
#                 tool_name="update_order",
#                 decision=PolicyDecision.DENY,
#             )
#         ]
#     )
#
#     gateway = create_gateway(policy_engine)
#
#     with pytest.raises(PermissionError, match="denied"):
#         await gateway.execute(
#             tool_name="update_order",
#             arguments={
#                 "order_id": "1001",
#                 "quantity": 80,
#             },
#         )
#
#
# @pytest.mark.asyncio
# async def test_gateway_policy_require_approval():
#     policy_engine = PolicyEngine(
#         rules=[
#             PolicyRule(
#                 tool_name="update_order",
#                 decision=PolicyDecision.REQUIRE_APPROVAL,
#             )
#         ]
#     )
#
#     gateway = create_gateway(policy_engine)
#
#     with pytest.raises(
#         PermissionError,
#         match="requires approval",
#     ):
#         await gateway.execute(
#             tool_name="update_order",
#             arguments={
#                 "order_id": "1001",
#                 "quantity": 80,
#             },
#         )