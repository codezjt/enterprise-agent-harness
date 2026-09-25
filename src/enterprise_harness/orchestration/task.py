from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    RETRYING = "RETRYING"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"
    WAITING_APPROVAL = "WAITING_APPROVAL"


class Task(BaseModel):
    task_id: str
    name: str
    description: str = ""

    status: TaskStatus = TaskStatus.PENDING

    dependencies: list[str] = Field(default_factory=list)

    input: dict[str, Any] = Field(default_factory=dict)
    output: Any | None = None
    error: str | None = None

    retry_count: int = 0
    max_retries: int = 0

    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_ready(self, completed_task_ids: set[str]) -> bool:
        return (
            self.status == TaskStatus.PENDING
            and all(
                dependency in completed_task_ids
                for dependency in self.dependencies
            )
        )