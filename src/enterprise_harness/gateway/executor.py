import inspect
from typing import Any

from .models import ToolDefinition


class ToolExecutor:
    """
    Tool 实际执行器。
    """

    async def execute(
        self,
        tool: ToolDefinition,
        arguments: dict[str, Any],
    ) -> Any:
        if tool.handler is None:
            raise RuntimeError(
                f"Tool handler is not configured: {tool.name}"
            )

        result = tool.handler(**arguments)

        if inspect.isawaitable(result):
            return await result

        return result