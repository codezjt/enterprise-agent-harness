from .models import ToolDefinition


class ToolRegistry:
    """
    Tool 注册中心。
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(
                f"Tool already registered: {tool.name}"
            )

        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        tool = self._tools.get(name)

        if tool is None:
            raise KeyError(
                f"Tool not found: {name}"
            )

        return tool

    def exists(self, name: str) -> bool:
        return name in self._tools

    def unregister(self, name: str) -> None:
        if name not in self._tools:
            raise KeyError(
                f"Tool not found: {name}"
            )

        del self._tools[name]

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())