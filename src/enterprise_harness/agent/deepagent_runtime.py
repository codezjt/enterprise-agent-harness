from __future__ import annotations

from typing import Any

from deepagents import create_deep_agent

from enterprise_harness.gateway.deepagent import DeepAgentToolAdapter
from enterprise_harness.gateway.gateway import ToolGateway
from enterprise_harness.runtime.context import RunContext

from .config import AgentConfig
from .model import resolve_model
from .runtime import AgentRuntime


class DeepAgentRuntime(AgentRuntime):
    """
    基于 DeepAgents 的 AgentRuntime 实现。

    Agent 本身可以复用；
    RunContext 在每次执行时注入。
    """

    def __init__(
        self,
        config: AgentConfig,
        tool_gateway: ToolGateway | None = None,
        tools: list[Any] | None = None,
        principal=None,
    ):
        super().__init__(config)

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

        return await self.agent.ainvoke(
            input_data
        )

    async def run_with_context(
        self,
        run_context: RunContext,
    ) -> Any:

        agent_tools = list(self.tools)

        if (
            self.tool_gateway is not None
            and self.config.tools
        ):

            definitions = [
                self.tool_gateway.registry.get(
                    tool_name
                )
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
                adapter.adapt_all(
                    definitions
                )
            )

        # 每次 Run 根据当前 Context 创建执行 Agent。
        model = resolve_model(
            self.config.model
        )

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
            ]
        }

        if run_context.context:
            input_data["context"] = (
                run_context.context
            )

        return await agent.ainvoke(
            input_data
        )