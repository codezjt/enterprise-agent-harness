import inspect
from typing import Any

from .models import ToolDefinition


class ToolExecutor:
    """负责实际执行 Tool。"""

    async def execute(
        self,
        tool: ToolDefinition,
        arguments: dict[str, Any],
    ) -> Any:
        if tool.handler is None:
            raise ValueError(
                f"Tool '{tool.name}' has no handler"
            )

        result = tool.handler(**arguments)

        if inspect.isawaitable(result):
            result = await result

        return result