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
    def build_agent(
        self,
        run_context: RunContext | None = None,
    ) -> Any:
        """
        基于 RunContext 构建 Agent 执行图。

        DeepAgentRuntime 负责 Agent Capability 的组装：
          - AgentConfig
          - Model
          - Tools（通过 ToolGateway）
          - Middleware
          - Skills
          - Memory

        LangGraphRuntime 负责 Durable Execution：
          - checkpoint
          - interrupt
          - resume
          - cancellation

        RunManager 只调用 Runtime.run(context)。
        """
        raise NotImplementedError