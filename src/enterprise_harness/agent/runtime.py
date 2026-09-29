from abc import ABC, abstractmethod
from typing import Any

from enterprise_harness.observability import MetricCollector
from enterprise_harness.runtime.context import RunContext

from .config import AgentConfig


class AgentRuntime(ABC):

    def __init__(
        self,
        config: AgentConfig,
        metric_collector: MetricCollector | None = None,
    ):
        self.config = config
        self.metric_collector = metric_collector

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

    async def run_with_context(
        self,
        run_context: RunContext,
    ) -> Any:
        try:
            result = await self.run(
                task=run_context.task,
                context=run_context.context,
            )

            if self.metric_collector is not None:
                self.metric_collector.record_agent_run(
                    success=True
                )

            return result

        except Exception:
            if self.metric_collector is not None:
                self.metric_collector.record_agent_run(
                    success=False
                )

            raise