# import pytest
#
# from enterprise_harness.gateway import ToolDefinition, ToolExecutor
#
#
# def query_order(order_id: str):
#     return {
#         "order_id": order_id,
#         "status": "CREATED",
#     }
#
#
# async def query_inventory(material_id: str):
#     return {
#         "material_id": material_id,
#         "quantity": 100,
#     }
#
#
# @pytest.mark.asyncio
# async def test_execute_sync_tool():
#     tool = ToolDefinition(
#         name="query_order",
#         handler=query_order,
#     )
#
#     executor = ToolExecutor()
#
#     result = await executor.execute(
#         tool,
#         {
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
# async def test_execute_async_tool():
#     tool = ToolDefinition(
#         name="query_inventory",
#         handler=query_inventory,
#     )
#
#     executor = ToolExecutor()
#
#     result = await executor.execute(
#         tool,
#         {
#             "material_id": "M001",
#         },
#     )
#
#     assert result == {
#         "material_id": "M001",
#         "quantity": 100,
#     }
#
#
# @pytest.mark.asyncio
# async def test_execute_tool_without_handler():
#     tool = ToolDefinition(
#         name="query_order",
#     )
#
#     executor = ToolExecutor()
#
#     with pytest.raises(ValueError):
#         await executor.execute(
#             tool,
#             {
#                 "order_id": "1001",
#             },
#         )