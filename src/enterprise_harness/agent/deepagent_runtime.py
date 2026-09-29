from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent

from enterprise_harness.context import (
    ContextBuilder,
    ContextProvider,
)
from enterprise_harness.gateway.deepagent import DeepAgentToolAdapter
from enterprise_harness.gateway.gateway import ToolGateway
from enterprise_harness.observability import MetricCollector
from enterprise_harness.runtime.context import RunContext

from .config import AgentConfig
from .model import resolve_model
from .runtime import AgentRuntime
import time

class DeepAgentRuntime(AgentRuntime):
    """
    基于 DeepAgents 的 AgentRuntime 实现。

    Agent 本身可以复用，
    RunContext 在每次执行时注入。
    """

    def __init__(
        self,
        config: AgentConfig,
        tool_gateway: ToolGateway | None = None,
        tools: list[Any] | None = None,
        principal=None,
        context_builder=None,
        context_provider=None,
        metric_collector: MetricCollector | None = None,
    ):
        super().__init__(
            config,
            metric_collector=metric_collector,
        )

        self.tool_gateway = tool_gateway
        self.tools = tools or []

        model = resolve_model(config.model)

        # 不把 RunContext 绑定到 Agent。
        # Agent 是长期存在的 Runtime 组件。
        self.agent = create_deep_agent(
            model=model,
            tools=self.tools,
            system_prompt=config.system_prompt,
            name=config.name,
        )

        self.principal = principal

        self.context_builder = context_builder or ContextBuilder()

        self.context_provider = (
            context_provider or ContextProvider()
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Any:

        input_data = {
            "messages": [
                {
                    "role": "user",
                    "content": task,
                }
            ]
        }

        if context:
            input_data["context"] = context
        start_time = time.perf_counter()
        try:
            result = await self.agent.ainvoke(
                input_data
            )

            if self.metric_collector is not None:
                self.metric_collector.record_agent_run(
                    success=True
                )
                latency_ms = (
                                     time.perf_counter() - start_time
                             ) * 1000

                self.metric_collector.record_latency(
                    latency_ms
                )

            return result

        except Exception:
            if self.metric_collector is not None:
                self.metric_collector.record_agent_run(
                    success=False
                )

            raise

    async def run_with_context(
        self,
        run_context: RunContext,
    ):
        agent_tools = list(self.tools)

        if (
            self.tool_gateway is not None
            and self.config.tools
        ):
            definitions = [
                self.tool_gateway.registry.get(tool_name)
                for tool_name in self.config.tools
            ]

            principal = (
                run_context.principal
                if run_context.principal is not None
                else self.principal
            )

            adapter = DeepAgentToolAdapter(
                gateway=self.tool_gateway,
                principal=principal,
                context=run_context.context,
                run_id=run_context.run_id,
                parent_span_id=run_context.current_span_id,
            )

            agent_tools.extend(
                adapter.adapt_all(definitions)
            )

        if self.context_provider is not None:
            memory_results = await self.context_provider.get_memory(
                query=run_context.task,
            )

            rag_results = await self.context_provider.get_rag(
                query=run_context.task,
            )

            run_context.context["memory_results"] = memory_results
            run_context.context["rag_results"] = rag_results

        built_context = (
            self.context_builder.build_from_run_context(
                run_context
            )
        )

        run_context.built_context = built_context

        model = resolve_model(self.config.model)

        agent = create_deep_agent(
            model=model,
            tools=agent_tools,
            system_prompt=self.config.system_prompt,
            name=self.config.name,
        )

        input_data = {
            "messages": [
                {
                    "role": "user",
                    "content": run_context.task,
                }
            ],
            "context": {
                "items": [
                    item.model_dump()
                    for item in built_context
                ]
            },
        }
        start_time = time.perf_counter()
        try:
            result = await agent.ainvoke(input_data)

            if self.metric_collector is not None:
                self.metric_collector.record_agent_run(
                    success=True
                )
                latency_ms = (
                                     time.perf_counter() - start_time
                             ) * 1000

                self.metric_collector.record_latency(
                    latency_ms
                )

            return result

        except Exception:
            if self.metric_collector is not None:
                self.metric_collector.record_agent_run(
                    success=False
                )

            raise

    def _format_context(
        self,
        items,
    ) -> str:
        sections = []

        for item in items:
            sections.append(
                f"[{item.source}]\n{item.content}"
            )

        return "\n\n".join(sections)