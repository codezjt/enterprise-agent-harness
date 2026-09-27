from abc import ABC, abstractmethod
from typing import Any

from enterprise_harness.runtime.context import RunContext

from .config import AgentConfig


class AgentRuntime(ABC):

    def __init__(
        self,
        config: AgentConfig,
    ):
        self.config = config

    """
            执行一个 Agent Task。

            Args:
                task: 当前任务描述
                context: Runtime 上下文

            Returns:
                Agent 执行结果
            """
    @abstractmethod
    async def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        raise NotImplementedError

    async def run_with_context(
        self,
        run_context: RunContext,
    ) -> Any:

        return await self.run(
            task=run_context.task,
            context=run_context.context,
        )