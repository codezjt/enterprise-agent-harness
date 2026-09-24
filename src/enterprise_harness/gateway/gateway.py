from typing import Any

from enterprise_harness.policy import (
    PolicyDecision,
    PolicyEngine,
    Principal,
)

from .executor import ToolExecutor
from .registry import ToolRegistry
from .router import ToolRouter
from .validator import ToolValidator


class ToolGateway:
    """Tool 的统一执行入口。"""

    def __init__(
        self,
        registry: ToolRegistry,
        router: ToolRouter | None = None,
        validator: ToolValidator | None = None,
        executor: ToolExecutor | None = None,
        policy_engine: PolicyEngine | None = None,
    ):
        self.registry = registry
        self.router = router or ToolRouter(registry)
        self.validator = validator or ToolValidator()
        self.executor = executor or ToolExecutor()
        self.policy_engine = policy_engine or PolicyEngine()

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        principal: Principal | None = None,
    ) -> Any:
        arguments = arguments or {}

        # 1. Registry / Router
        tool = self.router.route(tool_name)

        # 2. 参数校验
        validated_arguments = self.validator.validate(
            tool,
            arguments,
        )

        # 3. RBAC + Policy
        decision = self.policy_engine.check(
            tool=tool,
            arguments=validated_arguments,
            context=context,
            principal=principal,
        )

        # 4. Policy Decision
        if decision == PolicyDecision.DENY:
            raise PermissionError(
                f"Tool execution denied by policy: {tool.name}"
            )

        if decision == PolicyDecision.REQUIRE_APPROVAL:
            raise PermissionError(
                f"Tool execution requires approval: {tool.name}"
            )

        # 5. Executor
        return await self.executor.execute(
            tool,
            validated_arguments,
        )