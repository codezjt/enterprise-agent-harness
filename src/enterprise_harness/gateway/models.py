from typing import Any, Callable

from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    """
    Harness 中统一管理的 Tool 定义。
    """

    name: str

    description: str = ""

    input_schema: dict[str, Any] = Field(
        default_factory=dict
    )

    risk_level: str = "LOW"

    permissions: list[str] = Field(
        default_factory=list
    )

    timeout: int = 30

    retry_policy: dict[str, Any] = Field(
        default_factory=dict
    )

    handler: Callable[..., Any] | None = None