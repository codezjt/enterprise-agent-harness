# import pytest
#
# from enterprise_harness.gateway import (
#     ToolDefinition,
#     ToolRegistry,
#     ToolRouter,
# )
#
#
# def query_order(order_id: str):
#     return {
#         "order_id": order_id,
#     }
#
#
# def test_route_tool():
#     registry = ToolRegistry()
#
#     tool = ToolDefinition(
#         name="query_order",
#         description="查询订单",
#         handler=query_order,
#     )
#
#     registry.register(tool)
#
#     router = ToolRouter(registry)
#
#     result = router.route("query_order")
#
#     assert result is tool
#     assert result.name == "query_order"
#
#
# def test_route_unknown_tool():
#     registry = ToolRegistry()
#     router = ToolRouter(registry)
#
#     with pytest.raises(KeyError):
#         router.route("query_order")