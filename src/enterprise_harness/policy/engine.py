from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .models import PolicyDecision
from .rbac import Principal, RBAC
from .rules import PolicyRule

if TYPE_CHECKING:
    from enterprise_harness.gateway.models import ToolDefinition


RISK_APPROVAL_LEVELS: frozenset[str] = frozenset({
    "HIGH",
    "CRITICAL",
})

RISK_DENY_LEVELS: frozenset[str] = frozenset({
    "BLOCKED",
})


class PolicyEngine:
    """负责判断 Tool 是否允许执行。

    决策链：

        Principal
            ↓
        RBAC 权限检查
            ↓
        PolicyRule 精确匹配
            ↓
        risk_level 自动判断
            ↓
        ALLOW / DENY / REQUIRE_APPROVAL
    """

    def __init__(
        self,
        rules: list[PolicyRule] | None = None,
        rbac: RBAC | None = None,
        approval_levels: frozenset[str] | None = None,
        deny_levels: frozenset[str] | None = None,
    ):
        self.rules = rules or []
        self.rbac = rbac or RBAC()
        self.approval_levels = approval_levels or RISK_APPROVAL_LEVELS
        self.deny_levels = deny_levels or RISK_DENY_LEVELS

    def check(
        self,
        tool: "ToolDefinition",
        arguments: dict[str, Any],
        context: dict[str, Any] | None = None,
        principal: Principal | None = None,
    ) -> PolicyDecision:

        if tool.permissions and principal is not None:
            for permission in tool.permissions:
                if not self.rbac.has_permission(
                    principal,
                    permission,
                ):
                    return PolicyDecision.DENY

        if tool.permissions and principal is None:
            return PolicyDecision.DENY

        for rule in self.rules:
            if rule.tool_name == tool.name:
                return rule.decision

        risk = (tool.risk_level or "LOW").upper()

        if risk in self.deny_levels:
            return PolicyDecision.DENY

        if risk in self.approval_levels:
            return PolicyDecision.REQUIRE_APPROVAL

        return PolicyDecision.ALLOW
