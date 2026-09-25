from __future__ import annotations

from typing import Any

from enterprise_harness.observability import (
    SpanStatus,
    SpanType,
    TraceManager,
)
from enterprise_harness.policy.engine import PolicyEngine
from enterprise_harness.policy.models import PolicyDecision
from enterprise_harness.policy.rbac import Principal

from .executor import ToolExecutor
from .registry import ToolRegistry
from .router import ToolRouter
from .validator import ToolValidator


class ToolGateway:

    def __init__(
        self,
        registry: ToolRegistry,
        router: ToolRouter,
        validator: ToolValidator,
        executor: ToolExecutor,
        policy_engine: PolicyEngine,
        trace_manager: TraceManager | None = None,
    ):
        self.registry = registry
        self.router = router
        self.validator = validator
        self.executor = executor
        self.policy_engine = policy_engine
        self.trace_manager = trace_manager

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        principal: Principal | None = None,
        *,
        run_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> Any:

        arguments = arguments or {}

        span = None

        if self.trace_manager is not None and run_id is not None:

            span = self.trace_manager.start_span(
                run_id=run_id,
                span_type=SpanType.TOOL,
                name=tool_name,
                parent_span_id=parent_span_id,
                input=arguments,
                metadata={
                    "principal_id": (
                        principal.principal_id
                        if principal
                        else None
                    )
                },
            )

        try:

            # 1. Tool Routing
            tool = self.router.route(tool_name)

            # 2. Input Validation
            validated_arguments = self.validator.validate(
                tool,
                arguments,
            )

            # 3. Policy Check
            decision = self.policy_engine.check(
                tool=tool,
                arguments=validated_arguments,
                context=context,
                principal=principal,
            )

            # 4. Policy Deny
            if decision == PolicyDecision.DENY:
                raise PermissionError(
                    f"Tool execution denied by policy: {tool.name}"
                )

            # 5. Approval
            if decision == PolicyDecision.REQUIRE_APPROVAL:
                raise PermissionError(
                    f"Tool execution requires approval: {tool.name}"
                )

            # 6. Execute
            result = await self.executor.execute(
                tool,
                validated_arguments,
            )

            # 7. Trace Success
            if span is not None:

                self.trace_manager.finish_span(
                    span.span_id,
                    output=result,
                    status=SpanStatus.SUCCESS,
                )

            return result

        except Exception as exc:

            # 8. Trace Failure
            if span is not None:

                self.trace_manager.fail_span(
                    span.span_id,
                    exc,
                )

            raise