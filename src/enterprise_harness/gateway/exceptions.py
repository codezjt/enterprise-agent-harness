from __future__ import annotations

from typing import Any


class ApprovalRequiredError(PermissionError):
    """Tool 执行需要人工审批。"""

    def __init__(
        self,
        *,
        approval_id: str,
        run_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        message: str | None = None,
    ) -> None:
        self.approval_id = approval_id
        self.run_id = run_id
        self.tool_name = tool_name
        self.arguments = arguments

        super().__init__(
            message
            or (
                "Tool execution requires approval: "
                f"{tool_name}"
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "approval_required",
            "approval_id": self.approval_id,
            "run_id": self.run_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "message": str(self),
        }