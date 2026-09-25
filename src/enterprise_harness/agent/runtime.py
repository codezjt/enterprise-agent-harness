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