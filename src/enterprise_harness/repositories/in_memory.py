from __future__ import annotations

from typing import Any

from enterprise_harness.agent.config import AgentConfig
from enterprise_harness.policy.approval import ApprovalRequest, ApprovalStatus
from enterprise_harness.runtime.models import Run

from .base import (
    AgentRepository,
    ApprovalRepository,
    CheckpointRepository,
    RunRepository,
)


class InMemoryRunRepository(RunRepository):

    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}

    async def save(self, run: Run) -> None:
        self._runs[run.run_id] = run

    async def get(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    async def list(
        self,
        *,
        tenant_id: str | None = None,
    ) -> list[Run]:
        runs = list(self._runs.values())
        if tenant_id is not None:
            runs = [r for r in runs if r.tenant_id == tenant_id]
        return runs

    async def delete(self, run_id: str) -> None:
        self._runs.pop(run_id, None)


class InMemoryApprovalRepository(ApprovalRepository):

    def __init__(self) -> None:
        self._approvals: dict[str, ApprovalRequest] = {}

    async def save(self, approval: ApprovalRequest) -> None:
        self._approvals[approval.approval_id] = approval

    async def get(self, approval_id: str) -> ApprovalRequest | None:
        return self._approvals.get(approval_id)

    async def list_pending(
        self,
        *,
        tenant_id: str | None = None,
    ) -> list[ApprovalRequest]:
        results = [
            a
            for a in self._approvals.values()
            if a.status == ApprovalStatus.PENDING
        ]
        if tenant_id is not None:
            results = [a for a in results if a.tenant_id == tenant_id]
        return results

    async def update(self, approval: ApprovalRequest) -> None:
        self._approvals[approval.approval_id] = approval

    async def delete(self, approval_id: str) -> None:
        self._approvals.pop(approval_id, None)


class InMemoryAgentRepository(AgentRepository):

    def __init__(self) -> None:
        self._configs: dict[tuple[str, str], AgentConfig] = {}

    async def save(self, config: AgentConfig) -> None:
        key = (config.agent_id, config.version)
        self._configs[key] = config

    async def get(
        self,
        agent_id: str,
        version: str | None = None,
    ) -> AgentConfig | None:
        if version is not None:
            return self._configs.get((agent_id, version))

        latest: AgentConfig | None = None
        latest_ver: str = "0.0.0"
        for (aid, ver), cfg in self._configs.items():
            if aid == agent_id and ver > latest_ver:
                latest = cfg
                latest_ver = ver
        return latest

    async def list_versions(self, agent_id: str) -> list[str]:
        versions = [
            ver
            for (aid, ver) in self._configs
            if aid == agent_id
        ]
        return sorted(versions)

    async def delete(self, agent_id: str, version: str) -> None:
        self._configs.pop((agent_id, version), None)


class InMemoryCheckpointRepository(CheckpointRepository):

    def __init__(self) -> None:
        self._checkpoints: dict[str, dict[str, Any]] = {}

    async def save(
        self,
        thread_id: str,
        checkpoint_data: dict[str, Any],
    ) -> None:
        self._checkpoints[thread_id] = checkpoint_data

    async def get(self, thread_id: str) -> dict[str, Any] | None:
        return self._checkpoints.get(thread_id)

    async def delete(self, thread_id: str) -> None:
        self._checkpoints.pop(thread_id, None)