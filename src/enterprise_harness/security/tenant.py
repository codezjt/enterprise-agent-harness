from __future__ import annotations

from pydantic import BaseModel, Field


class Tenant(BaseModel):
    tenant_id: str
    name: str = ""
    is_active: bool = True
    metadata: dict[str, str] = Field(default_factory=dict)