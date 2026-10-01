from __future__ import annotations

from pydantic import BaseModel, Field


class Identity(BaseModel):
    principal_id: str
    tenant_id: str = "default"
    roles: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)