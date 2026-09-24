# import pytest
#
# from enterprise_harness.gateway import (
#     ToolDefinition,
#     ToolRegistry,
# )
#
#
# def query_order(order_id: str):
#     return {
#         "order_id": order_id,
#         "quantity": 100,
#     }
#
#
# def test_register_and_get_tool():
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
#     result = registry.get("query_order")
#
#     assert result is tool
#     assert result.name == "query_order"
#     assert result.description == "查询订单"
#     assert result.handler is query_order
#
#
# def test_tool_exists():
#     registry = ToolRegistry()
#
#     tool = ToolDefinition(
#         name="query_order",
#         handler=query_order,
#     )
#
#     registry.register(tool)
#
#     assert registry.exists("query_order")
#     assert not registry.exists("update_order")
#
#
# def test_duplicate_tool_registration():
#     registry = ToolRegistry()
#
#     tool = ToolDefinition(
#         name="query_order",
#         handler=query_order,
#     )
#
#     registry.register(tool)
#
#     with pytest.raises(ValueError):
#         registry.register(tool)
#
#
# def test_get_unknown_tool():
#     registry = ToolRegistry()
#
#     with pytest.raises(KeyError):
#         registry.get("query_order")
#
#
# def test_unregister_tool():
#     registry = ToolRegistry()
#
#     tool = ToolDefinition(
#         name="query_order",
#         handler=query_order,
#     )
#
#     registry.register(tool)
#
#     registry.unregister("query_order")
#
#     assert not registry.exists("query_order")
#
#
# def test_list_tools():
#     registry = ToolRegistry()
#
#     tool1 = ToolDefinition(
#         name="query_order",
#         handler=query_order,
#     )
#
#     tool2 = ToolDefinition(
#         name="query_inventory",
#     )
#
#     registry.register(tool1)
#     registry.register(tool2)
#
#     tools = registry.list_tools()
#
#     assert len(tools) == 2
#     assert tools[0].name == "query_order"
#     assert tools[1].name == "query_inventory"