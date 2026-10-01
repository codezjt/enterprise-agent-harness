from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ApprovalStatus(str, Enum):
    """审批状态。"""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class ApprovalRequest(BaseModel):
    """一次需要人工确认的审批请求。"""

    approval_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    run_id: str
    tool_name: str
    tenant_id: str | None = None
    arguments: dict[str, Any] = Field(
        default_factory=dict
    )

    requester_id: str | None = None

    status: ApprovalStatus = ApprovalStatus.PENDING

    comment: str | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    resolved_at: datetime | None = None