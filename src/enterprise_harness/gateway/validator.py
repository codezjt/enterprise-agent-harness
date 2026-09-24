from typing import Any

from jsonschema import ValidationError
from jsonschema import validate

from .models import ToolDefinition


class ToolValidationError(ValueError):
    """Tool 参数校验失败。"""


class ToolValidator:
    """负责按照 ToolDefinition 中的 JSON Schema 校验参数。"""

    def validate(
        self,
        tool: ToolDefinition,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            validate(
                instance=arguments,
                schema=tool.input_schema,
            )
        except ValidationError as exc:
            raise ToolValidationError(
                f"Invalid arguments for tool '{tool.name}': {exc.message}"
            ) from exc

        return arguments