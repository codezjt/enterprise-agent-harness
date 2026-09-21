from abc import ABC, abstractmethod
from typing import Any

from .config import AgentConfig


class AgentRuntime(ABC):
    """
    Agent Runtime 抽象。

    Enterprise Harness 不直接依赖具体 Agent Framework，
    而是通过 AgentRuntime 调用 Agent。
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    @abstractmethod
    async def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """
        执行一个 Agent Task。

        Args:
            task: 当前任务描述
            context: Runtime 上下文

        Returns:
            Agent 执行结果
        """
        raise NotImplementedError