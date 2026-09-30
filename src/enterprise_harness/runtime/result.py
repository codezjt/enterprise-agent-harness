from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RuntimeStatus(str, Enum):
    """
    Runtime 的统一执行结果状态。

    RunManager 不应该关心底层到底是：
    DeepAgents / LangGraph / TaskGraph / Custom Runtime。

    所有 Runtime 最终都通过 RuntimeResult 向上层报告执行状态。
    """

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    CANCELLED = "CANCELLED"


@dataclass(slots=True)
class RuntimeResult:
    """
    Runtime 的统一执行结果。

    这是 Runtime Adapter 与 RunManager 之间的协议边界。

    设计原则：

    Runtime
        ↓
    RuntimeResult
        ↓
    RunManager
        ↓
    Run
    """

    status: RuntimeStatus

    result: Any | None = None

    error: str | None = None

    approval_id: str | None = None

    checkpoint_id: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_completed(self) -> bool:
        return self.status == RuntimeStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        return self.status == RuntimeStatus.FAILED

    @property
    def is_waiting_approval(self) -> bool:
        return self.status == RuntimeStatus.WAITING_APPROVAL

    @property
    def is_cancelled(self) -> bool:
        return self.status == RuntimeStatus.CANCELLED

    @classmethod
    def completed(
        cls,
        result: Any = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeResult:
        return cls(
            status=RuntimeStatus.COMPLETED,
            result=result,
            metadata=metadata or {},
        )

    @classmethod
    def failed(
        cls,
        error: str,
        *,
        result: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeResult:
        return cls(
            status=RuntimeStatus.FAILED,
            result=result,
            error=error,
            metadata=metadata or {},
        )

    @classmethod
    def waiting_approval(
        cls,
        *,
        approval_id: str | None = None,
        checkpoint_id: str | None = None,
        result: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeResult:
        return cls(
            status=RuntimeStatus.WAITING_APPROVAL,
            result=result,
            approval_id=approval_id,
            checkpoint_id=checkpoint_id,
            metadata=metadata or {},
        )

    @classmethod
    def cancelled(
        cls,
        *,
        result: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeResult:
        return cls(
            status=RuntimeStatus.CANCELLED,
            result=result,
            metadata=metadata or {},
        )