from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from enterprise_harness.agent.config import AgentConfig
from enterprise_harness.policy.approval import ApprovalRequest
from enterprise_harness.runtime.models import Run


class RunRepository(ABC):

    @abstractmethod
    async def save(self, run: Run) -> None:
        ...

    @abstractmethod
    async def get(self, run_id: str) -> Run | None:
        ...

    @abstractmethod
    async def list(
        self,
        *,
        tenant_id: str | None = None,
    ) -> list[Run]:
        ...

    @abstractmethod
    async def delete(self, run_id: str) -> None:
        ...


class ApprovalRepository(ABC):

    @abstractmethod
    async def save(self, approval: ApprovalRequest) -> None:
        ...

    @abstractmethod
    async def get(self, approval_id: str) -> ApprovalRequest | None:
        ...

    @abstractmethod
    async def list_pending(
        self,
        *,
        tenant_id: str | None = None,
    ) -> list[ApprovalRequest]:
        ...

    @abstractmethod
    async def update(self, approval: ApprovalRequest) -> None:
        ...

    @abstractmethod
    async def delete(self, approval_id: str) -> None:
        ...


class AgentRepository(ABC):

    @abstractmethod
    async def save(self, config: AgentConfig) -> None:
        ...

    @abstractmethod
    async def get(
        self,
        agent_id: str,
        version: str | None = None,
    ) -> AgentConfig | None:
        ...

    @abstractmethod
    async def list_versions(self, agent_id: str) -> list[str]:
        ...

    @abstractmethod
    async def delete(
        self,
        agent_id: str,
        version: str,
    ) -> None:
        ...


class CheckpointRepository(ABC):

    @abstractmethod
    async def save(
        self,
        thread_id: str,
        checkpoint_data: dict[str, Any],
    ) -> None:
        ...

    @abstractmethod
    async def get(
        self,
        thread_id: str,
    ) -> dict[str, Any] | None:
        ...

    @abstractmethod
    async def delete(self, thread_id: str) -> None:
        ...