from typing import Any

from pydantic import BaseModel, Field


class ContextItem(BaseModel):
    priority: int
    source: str
    content: Any
    metadata: dict[str, Any] = Field(default_factory=dict)