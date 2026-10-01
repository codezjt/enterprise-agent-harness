from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from jsonschema import ValidationError, validate as jsonschema_validate


class ResultValidator(ABC):
    """Tool 执行结果校验器。"""

    @abstractmethod
    def validate(
        self,
        tool_name: str,
        result: Any,
        output_schema: dict[str, Any] | None = None,
    ) -> Any:
        raise NotImplementedError


class DefaultResultValidator(ResultValidator):
    """默认结果校验 —— 非空检查 + JSON Schema 校验。"""

    def validate(
        self,
        tool_name: str,
        result: Any,
        output_schema: dict[str, Any] | None = None,
    ) -> Any:
        if result is None:
            raise ValueError(
                f"Tool '{tool_name}' returned None"
            )

        if output_schema is not None:
            try:
                jsonschema_validate(
                    instance=result,
                    schema=output_schema,
                )
            except ValidationError as exc:
                raise ValueError(
                    f"Tool '{tool_name}' result validation failed: "
                    f"{exc.message}"
                ) from exc

        return result


class NoOpResultValidator(ResultValidator):
    """不做校验，直接透传。"""

    def validate(
        self,
        tool_name: str,
        result: Any,
        output_schema: dict[str, Any] | None = None,
    ) -> Any:
        return result
