from typing import Any

from enterprise_harness.gateway import ToolDefinition

from .models import PolicyDecision
from .rules import PolicyRule


class PolicyEngine:
    """负责判断 Tool 是否允许执行。"""

    def __init__(
        self,
        rules: list[PolicyRule] | None = None,
    ):
        self.rules = rules or []

    def check(
        self,
        tool: ToolDefinition,
        arguments: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> PolicyDecision:
        for rule in self.rules:
            if rule.tool_name == tool.name:
                return rule.decision

        return PolicyDecision.ALLOW