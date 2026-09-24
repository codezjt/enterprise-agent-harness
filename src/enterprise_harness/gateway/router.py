from .models import ToolDefinition
from .registry import ToolRegistry


class ToolRouter:
    """负责根据 Tool 名称找到对应的 Tool 定义。"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def route(self, tool_name: str) -> ToolDefinition:
        return self.registry.get(tool_name)