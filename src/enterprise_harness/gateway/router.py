from .models import ToolDefinition
from .registry import ToolRegistry


class ToolRouter:
    """
    根据 Tool 名称路由到 ToolDefinition。
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def route(self, tool_name: str) -> ToolDefinition:
        return self.registry.get(tool_name)