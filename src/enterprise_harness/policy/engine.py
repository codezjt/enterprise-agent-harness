from typing import Any

from enterprise_harness.gateway import ToolDefinition

from .models import PolicyDecision
from .rbac import Principal, RBAC
from .rules import PolicyRule


class PolicyEngine:
    """负责判断 Tool 是否允许执行。

    决策链：

        Principal
            ↓
        RBAC 权限检查
            ↓
        PolicyRule
            ↓
        ALLOW / DENY / REQUIRE_APPROVAL
    """

    def __init__(
        self,
        rules: list[PolicyRule] | None = None,
        rbac: RBAC | None = None,
    ):
        self.rules = rules or []
        self.rbac = rbac or RBAC()

    def check(
        self,
        tool: ToolDefinition,
        arguments: dict[str, Any],
        context: dict[str, Any] | None = None,
        principal: Principal | None = None,
    ) -> PolicyDecision:
        # 1. 如果 Tool 没有配置权限要求，
        #    则不需要 RBAC 检查。
        if tool.permissions and principal is not None:
            for permission in tool.permissions:
                if not self.rbac.has_permission(
                    principal,
                    permission,
                ):
                    return PolicyDecision.DENY

        # 2. 如果 Tool 要求权限，但调用方没有身份，
        #    企业场景下默认拒绝。
        if tool.permissions and principal is None:
            return PolicyDecision.DENY

        # 3. RBAC 通过后，再检查 PolicyRule。
        for rule in self.rules:
            if rule.tool_name == tool.name:
                return rule.decision

        # 4. 没有特殊策略，默认允许。
        return PolicyDecision.ALLOW