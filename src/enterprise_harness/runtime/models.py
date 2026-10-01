from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"

    @classmethod
    def is_valid_transition(cls, from_status, to_status) -> bool:
        return to_status in _STATUS_TRANSITIONS.get(from_status, frozenset())


_STATUS_TRANSITIONS = {
    RunStatus.CREATED: {RunStatus.RUNNING, RunStatus.PLANNING, RunStatus.CANCELLED},
    RunStatus.RUNNING: {RunStatus.PLANNING, RunStatus.EXECUTING, RunStatus.WAITING_APPROVAL, RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.TIMEOUT},
    RunStatus.PLANNING: {RunStatus.EXECUTING, RunStatus.FAILED, RunStatus.CANCELLED},
    RunStatus.EXECUTING: {RunStatus.WAITING_APPROVAL, RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.TIMEOUT},
    RunStatus.WAITING_APPROVAL: {RunStatus.RUNNING, RunStatus.FAILED, RunStatus.CANCELLED},
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
    RunStatus.TIMEOUT: {RunStatus.FAILED, RunStatus.CANCELLED},
}


class Run(BaseModel):
    run_id: str
    agent_id: str
    agent_version: str = "1.0.0"
    tenant_id: str = "default"
    task: str

    status: RunStatus = RunStatus.CREATED

    context: dict[str, Any] = Field(default_factory=dict)

    result: Any | None = None
    error: str | None = None

    checkpoint_id: str | None = None
    approval_id: str | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    started_at: datetime | None = None
    completed_at: datetime | None = None
