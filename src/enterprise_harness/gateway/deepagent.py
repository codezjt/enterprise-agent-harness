from typing import Any

from langchain_core.tools import StructuredTool

from enterprise_harness.gateway.exceptions import (
    ApprovalRequiredError,
    ToolApprovalRejectedError,
)
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
        tenant_id: str = "default",
    ):
        self.gateway = gateway
        self.principal = principal
        self.context = context or {}
        self.run_id = run_id
        self.parent_span_id = parent_span_id
        self.tenant_id = tenant_id

    def adapt(
        self,
        tool: ToolDefinition,
    ) -> StructuredTool:
        adapter = self
        _tool_name = tool.name

        async def invoke(**arguments: Any) -> Any:
            try:
                ctx = dict(adapter.context)
                ctx["tenant_id"] = adapter.tenant_id
                return await adapter.gateway.execute(
                    tool_name=_tool_name,
                    arguments=arguments,
                    context=ctx,
                    principal=adapter.principal,
                    run_id=adapter.run_id,
                    parent_span_id=adapter.parent_span_id,
                    tenant_id=adapter.tenant_id,
                )
            except ApprovalRequiredError as exc:
                from langgraph.types import interrupt

                result = interrupt(
                    {
                        "type": "approval_required",
                        "approval_id": exc.approval_id,
                        "tool_name": exc.tool_name or _tool_name,
                        "run_id": exc.run_id or adapter.run_id,
                        "message": str(exc),
                        "principal": (
                            adapter.principal.principal_id
                            if adapter.principal
                            else None
                        ),
                    }
                )

                if isinstance(result, dict) and result.get("approved"):
                    ctx = dict(adapter.context)
                    ctx["tenant_id"] = adapter.tenant_id
                    return await adapter.gateway.execute(
                        tool_name=_tool_name,
                        arguments=arguments,
                        context=ctx,
                        principal=adapter.principal,
                        run_id=adapter.run_id,
                        parent_span_id=adapter.parent_span_id,
                        approval_id=exc.approval_id,
                        tenant_id=adapter.tenant_id,
                    )

                raise ToolApprovalRejectedError(
                    _tool_name,
                    exc.approval_id,
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
