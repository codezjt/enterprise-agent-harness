from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .result import RuntimeResult


class Runtime(ABC):
    """
    Enterprise Harness 统一 Runtime Contract。

    所有具体 Runtime 都应该通过这个协议接入 RunManager：

        RunManager
            ↓
        Runtime
            ↓
        RuntimeResult

    具体实现可以是：

        DeepAgentRuntime
        LangGraphRuntime
        TaskGraphRuntime
        PlannedRuntime
        CustomRuntime
    """

    @abstractmethod
    async def run(
        self,
        context: Any,
    ) -> RuntimeResult:
        """
        启动一次 Runtime 执行。

        Args:
            context:
                Enterprise Harness 的统一执行上下文。

        Returns:
            RuntimeResult:
                统一 Runtime 执行结果。
        """
        raise NotImplementedError

    async def resume(
        self,
        context: Any,
        value: Any,
    ) -> RuntimeResult:
        """
        恢复一次被 Runtime 暂停的执行。

        默认 Runtime 不支持 Resume。

        LangGraph / Durable Runtime 等需要暂停恢复能力的实现
        可以覆盖此方法。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support resume"
        )

    async def cancel(
        self,
        context: Any,
    ) -> RuntimeResult:
        """
        取消 Runtime 执行。

        默认实现不提供真正的底层取消能力。

        RunManager 当前可以继续维护 Run 生命周期；
        后续如果 Runtime 支持真正的 cancellation，
        再由具体 Runtime 覆盖。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support cancel"
        )

    @property
    def supports_resume(self) -> bool:
        """
        当前 Runtime 是否支持 Resume。
        """
        return False

    @property
    def supports_cancel(self) -> bool:
        """
        当前 Runtime 是否支持底层 Cancel。
        """
        return False