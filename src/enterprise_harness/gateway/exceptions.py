from __future__ import annotations

from typing import Any


class ToolExecutionError(RuntimeError):
    def __init__(
        self,
        tool_name: str,
        message: str | None = None,
    ) -> None:
        self.tool_name = tool_name
        super().__init__(
            message
            or f"Tool execution failed: {tool_name}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_execution_error",
            "tool_name": self.tool_name,
            "message": str(self),
        }


class ToolNotFoundError(KeyError):
    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool not found: {tool_name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_not_found",
            "tool_name": self.tool_name,
            "message": str(self),
        }


class ToolValidationError(ValueError):
    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        super().__init__(
            f"Invalid arguments for tool '{tool_name}': {message}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_validation_error",
            "tool_name": self.tool_name,
            "message": str(self),
        }


class ToolPolicyError(PermissionError):
    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(
            f"Tool execution denied by policy: {tool_name}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_policy_error",
            "tool_name": self.tool_name,
            "message": str(self),
        }


class ToolTimeoutError(TimeoutError):
    def __init__(self, tool_name: str, timeout: float) -> None:
        self.tool_name = tool_name
        self.timeout = timeout
        super().__init__(
            f"Tool execution timed out after {timeout}s: {tool_name}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_timeout_error",
            "tool_name": self.tool_name,
            "timeout": self.timeout,
            "message": str(self),
        }


class ToolApprovalRequiredError(PermissionError):
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
                f"{tool_name} (approval_id={approval_id})"
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_approval_required",
            "approval_id": self.approval_id,
            "run_id": self.run_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "message": str(self),
        }


class ToolApprovalRejectedError(PermissionError):
    def __init__(self, tool_name: str, approval_id: str) -> None:
        self.tool_name = tool_name
        self.approval_id = approval_id
        super().__init__(
            f"Approval rejected for tool {tool_name} "
            f"(approval_id={approval_id})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_approval_rejected",
            "tool_name": self.tool_name,
            "approval_id": self.approval_id,
            "message": str(self),
        }


ApprovalRequiredError = ToolApprovalRequiredError


_ERROR_TYPE_MAP: dict[type[Exception] | type[BaseException], str] = {
    ToolNotFoundError: "TOOL_NOT_FOUND",
    ToolValidationError: "TOOL_VALIDATION_FAILED",
    ToolPolicyError: "TOOL_POLICY_DENIED",
    ToolTimeoutError: "TOOL_TIMEOUT",
    ToolApprovalRequiredError: "TOOL_APPROVAL_REQUIRED",
    ToolApprovalRejectedError: "TOOL_APPROVAL_REJECTED",
    ToolExecutionError: "TOOL_EXECUTION_FAILED",
    PermissionError: "PERMISSION_ERROR",
    TimeoutError: "TIMEOUT_ERROR",
    ValueError: "VALIDATION_ERROR",
    KeyError: "RESOURCE_NOT_FOUND",
}


def classify_error(exc: BaseException) -> str:
    for exc_type, code in _ERROR_TYPE_MAP.items():
        if isinstance(exc, exc_type):
            return code
    return "UNKNOWN_ERROR"