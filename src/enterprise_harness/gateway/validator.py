from typing import Any

from jsonschema import ValidationError, validate

from .exceptions import ToolValidationError
from .models import ToolDefinition


class ToolValidator:

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
                tool.name,
                exc.message,
            )

        return arguments