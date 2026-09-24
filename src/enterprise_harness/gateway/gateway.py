from typing import Any

from .executor import ToolExecutor
from .registry import ToolRegistry
from .router import ToolRouter
from .validator import ToolValidator


class ToolGateway:
    """Tool 的统一执行入口。

    对上层屏蔽 Registry、Router、Validator、Executor 的内部实现。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        router: ToolRouter | None = None,
        validator: ToolValidator | None = None,
        executor: ToolExecutor | None = None,
    ):
        self.registry = registry
        self.router = router or ToolRouter(registry)
        self.validator = validator or ToolValidator()
        self.executor = executor or ToolExecutor()

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        arguments = arguments or {}

        # 1. Router：找到 Tool
        tool = self.router.route(tool_name)

        # 2. Validator：校验参数
        validated_arguments = self.validator.validate(
            tool,
            arguments,
        )

        # 3. Executor：执行 Tool
        return await self.executor.execute(
            tool,
            validated_arguments,
        )