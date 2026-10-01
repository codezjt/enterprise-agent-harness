from __future__ import annotations

from typing import Any

from deepagents import (
    create_deep_agent,
    FilesystemMiddleware,
    MemoryMiddleware,
)
from langgraph.checkpoint.memory import MemorySaver

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


_FS_TOOL_MAP: dict[str, str] = {
    "ls": "ls",
    "read": "read_file",
    "read_file": "read_file",
    "write": "write_file",
    "write_file": "write_file",
    "edit": "edit_file",
    "edit_file": "edit_file",
    "delete": "delete",
    "glob": "glob",
    "grep": "grep",
    "execute": "execute",
}


def _build_middleware_list(config: AgentConfig) -> list:
    mws: list = []
    for mw_name in config.middleware:
        name = mw_name.lower()
        if name == "filesystem":
            tools: list[str] | None = None
            if config.filesystem_permissions:
                tools = []
                for p in config.filesystem_permissions:
                    mapped = _FS_TOOL_MAP.get(p.lower())
                    if mapped:
                        tools.append(mapped)
            mws.append(FilesystemMiddleware(tools=tools))
        elif name == "memory":
            try:
                mws.append(MemoryMiddleware())
            except TypeError:
                pass
    return mws


class DeepAgentRuntime(AgentRuntime):
    """
    基于 DeepAgents 的 AgentRuntime 实现。

    直接调用 create_deep_agent() 构建 CompiledStateGraph，
    完整接入 middleware / skills / interrupt_on / checkpointer。
    当 Tool Gateway 的 ApprovalRequiredError 被 adapter 捕获后，
    用 langgraph.types.interrupt() 中断 Agent Loop，
    让 LangGraph Runtime 能检测 WAITING_APPROVAL 并 resume。
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
        self.principal = principal
        self.context_builder = context_builder or ContextBuilder()
        self.context_provider = context_provider or ContextProvider()

        self._checkpointer = MemorySaver() if config.use_checkpointer else None
        self._middleware = _build_middleware_list(config)

    def _build_tools_for_run(
        self,
        run_context: RunContext | None = None,
    ) -> list:
        agent_tools = list(self.tools)

        if self.tool_gateway is not None and self.config.tools:
            definitions = [
                self.tool_gateway.registry.get(tool_name)
                for tool_name in self.config.tools
            ]
            definitions = [d for d in definitions if d is not None]

            principal = (
                run_context.principal
                if run_context and run_context.principal is not None
                else self.principal
            )

            adapter = DeepAgentToolAdapter(
                gateway=self.tool_gateway,
                principal=principal,
                context=run_context.context if run_context else {},
                run_id=run_context.run_id if run_context else None,
                parent_span_id=(
                    run_context.current_span_id
                    if run_context
                    else None
                ),
                tenant_id=(
                    run_context.tenant_id
                    if run_context
                    else "default"
                ),
            )

            agent_tools.extend(adapter.adapt_all(definitions))

        return agent_tools

    def build_agent(
        self,
        run_context: RunContext | None = None,
    ):
        model = resolve_model(self.config.model)
        agent_tools = self._build_tools_for_run(run_context)

        kwargs: dict[str, Any] = {
            "model": model,
            "tools": agent_tools,
            "system_prompt": self.config.system_prompt,
            "name": self.config.name,
        }

        if self._middleware:
            kwargs["middleware"] = self._middleware

        if self.config.skills:
            kwargs["skills"] = self.config.skills

        if self.config.interrupt_on:
            kwargs["interrupt_on"] = self.config.interrupt_on

        if self._checkpointer is not None:
            kwargs["checkpointer"] = self._checkpointer

        return create_deep_agent(**kwargs)

    @property
    def checkpointer(self):
        return self._checkpointer

    async def run(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        input_data: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": task,
                }
            ]
        }
        if context:
            input_data["context"] = context

        agent = self.build_agent(run_context=None)
        start_time = time.perf_counter()
        try:
            result = await agent.ainvoke(input_data)
            if self.metric_collector is not None:
                self.metric_collector.record_agent_run(success=True)
                self.metric_collector.record_latency(
                    (time.perf_counter() - start_time) * 1000
                )
            return result
        except Exception:
            if self.metric_collector is not None:
                self.metric_collector.record_agent_run(success=False)
            raise

    async def run_with_context(
        self,
        run_context: RunContext,
    ):
        if self.context_provider is not None:
            memory_results = await self.context_provider.get_memory(
                query=run_context.task,
            )
            rag_results = await self.context_provider.get_rag(
                query=run_context.task,
            )
            run_context.context["memory_results"] = memory_results
            run_context.context["rag_results"] = rag_results

        built_context = self.context_builder.build_from_run_context(run_context)
        run_context.built_context = built_context

        agent = self.build_agent(run_context)

        input_data = {
            "messages": [
                {
                    "role": "user",
                    "content": run_context.task,
                }
            ],
            "context": {
                "items": [item.model_dump() for item in built_context]
            },
        }

        thread_id = run_context.run_id or str(__import__("uuid").uuid4())
        invoke_config = {"configurable": {"thread_id": thread_id}}

        start_time = time.perf_counter()
        try:
            result = await agent.ainvoke(input_data, invoke_config)
            if self.metric_collector is not None:
                self.metric_collector.record_agent_run(success=True)
                self.metric_collector.record_latency(
                    (time.perf_counter() - start_time) * 1000
                )
            return result
        except Exception:
            if self.metric_collector is not None:
                self.metric_collector.record_agent_run(success=False)
            raise
