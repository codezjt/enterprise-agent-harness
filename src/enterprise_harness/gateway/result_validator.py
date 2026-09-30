from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ResultValidator(ABC):
    """Tool 执行结果校验器。"""

    @abstractmethod
    def validate(
        self,
        tool_name: str,
        result: Any,
    ) -> bool:
        raise NotImplementedError


class DefaultResultValidator(ResultValidator):
    """默认结果校验：只检查 result 不为 None。"""

    def validate(
        self,
        tool_name: str,
        result: Any,
    ) -> bool:
        return result is not None
