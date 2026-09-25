from typing import Any

from langchain_core.tools import StructuredTool

from enterprise_harness.policy.rbac import Principal

from .gateway import ToolGateway
from .models import ToolDefinition


class DeepAgentToolAdapter:

    def __init__(
        self,
        gateway: ToolGateway,
        principal: Principal | None = None,
        context: dict[str, Any] | None = None,
        run_id: str | None = None,
        parent_span_id: str | None = None,
    ):
        self.gateway = gateway
        self.principal = principal
        self.context = context or {}
        self.run_id = run_id
        self.parent_span_id = parent_span_id

    def adapt(
        self,
        tool: ToolDefinition,
    ) -> StructuredTool:

        async def invoke(**arguments: Any) -> Any:

            return await self.gateway.execute(
                tool_name=tool.name,
                arguments=arguments,
                context=self.context,
                principal=self.principal,
                run_id=self.run_id,
                parent_span_id=self.parent_span_id,
            )

        return StructuredTool.from_function(
            coroutine=invoke,
            name=tool.name,
            description=tool.description,
            args_schema=None,
        )

    def adapt_all(
        self,
        tools: list[ToolDefinition],
    ) -> list[StructuredTool]:

        return [
            self.adapt(tool)
            for tool in tools
        ]