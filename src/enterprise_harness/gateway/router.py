from .exceptions import ToolNotFoundError
from .models import ToolDefinition
from .registry import ToolRegistry


class ToolRouter:

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def route(self, tool_name: str) -> ToolDefinition:
        try:
            return self.registry.get(tool_name)
        except KeyError:
            raise ToolNotFoundError(tool_name)