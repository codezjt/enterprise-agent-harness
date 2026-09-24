from dataclasses import dataclass

from .models import PolicyDecision


@dataclass
class PolicyRule:
    """最基础的 Tool Policy 规则。"""

    tool_name: str
    decision: PolicyDecision