from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from enterprise_harness.agent.config import AgentConfig
from enterprise_harness.policy.approval import ApprovalRequest, ApprovalStatus
from enterprise_harness.runtime.models import Run, RunStatus

from .base import (
    AgentRepository,
    ApprovalRepository,
    CheckpointRepository,
    RunRepository,
)


def _to_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _from_json(text: str | None) -> Any:
    if text is None:
        return None
    return json.loads(text)


def _to_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _from_iso(text: str | None) -> datetime | None:
    if text is None:
        return None
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


class SqliteRunRepository(RunRepository):

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    async def close(self) -> None:
        if self._conn is not None:
            await asyncio.to_thread(self._conn.close)
            self._conn = None

    async def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = await asyncio.to_thread(
                self._create_connection
            )
        return self._conn

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                agent_version TEXT DEFAULT '1.0.0',
                tenant_id TEXT DEFAULT 'default',
                task TEXT NOT NULL,
                status TEXT DEFAULT 'CREATED',
                context TEXT DEFAULT '{}',
                result TEXT,
                error TEXT,
                checkpoint_id TEXT,
                approval_id TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            )
            """
        )
        conn.commit()
        return conn

    def _row_to_run(self, row: sqlite3.Row) -> Run:
        return Run(
            run_id=row["run_id"],
            agent_id=row["agent_id"],
            agent_version=row["agent_version"] or "1.0.0",
            tenant_id=row["tenant_id"] or "default",
            task=row["task"],
            status=RunStatus(row["status"]),
            context=_from_json(row["context"]) or {},
            result=_from_json(row["result"]),
            error=row["error"],
            checkpoint_id=row["checkpoint_id"],
            approval_id=row["approval_id"],
            created_at=_from_iso(row["created_at"]),
            started_at=_from_iso(row["started_at"]),
            completed_at=_from_iso(row["completed_at"]),
        )

    async def save(self, run: Run) -> None:
        conn = await self._ensure_conn()

        def _save() -> None:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                    (run_id, agent_id, agent_version, tenant_id, task,
                     status, context, result, error,
                     checkpoint_id, approval_id,
                     created_at, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.agent_id,
                    run.agent_version,
                    run.tenant_id,
                    run.task,
                    run.status.value,
                    _to_json(run.context),
                    _to_json(run.result),
                    run.error,
                    run.checkpoint_id,
                    run.approval_id,
                    _to_iso(run.created_at),
                    _to_iso(run.started_at),
                    _to_iso(run.completed_at),
                ),
            )
            conn.commit()

        await asyncio.to_thread(_save)

    async def get(self, run_id: str) -> Run | None:
        conn = await self._ensure_conn()

        def _get() -> sqlite3.Row | None:
            return conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()

        row = await asyncio.to_thread(_get)
        if row is None:
            return None
        return self._row_to_run(row)

    async def list(
        self,
        *,
        tenant_id: str | None = None,
    ) -> list[Run]:
        conn = await self._ensure_conn()

        def _list() -> list[sqlite3.Row]:
            if tenant_id is not None:
                rows = conn.execute(
                    "SELECT * FROM runs WHERE tenant_id = ?", (tenant_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM runs").fetchall()
            return rows

        rows = await asyncio.to_thread(_list)
        return [self._row_to_run(r) for r in rows]

    async def delete(self, run_id: str) -> None:
        conn = await self._ensure_conn()

        def _delete() -> None:
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
            conn.commit()

        await asyncio.to_thread(_delete)


class SqliteApprovalRepository(ApprovalRepository):

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    async def close(self) -> None:
        if self._conn is not None:
            await asyncio.to_thread(self._conn.close)
            self._conn = None

    async def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = await asyncio.to_thread(
                self._create_connection
            )
        return self._conn

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS approvals (
                approval_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                tenant_id TEXT,
                arguments TEXT DEFAULT '{}',
                requester_id TEXT,
                status TEXT DEFAULT 'PENDING',
                comment TEXT,
                created_at TEXT NOT NULL,
                resolved_at TEXT
            )
            """
        )
        conn.commit()
        return conn

    def _row_to_approval(self, row: sqlite3.Row) -> ApprovalRequest:
        return ApprovalRequest(
            approval_id=row["approval_id"],
            run_id=row["run_id"],
            tool_name=row["tool_name"],
            tenant_id=row["tenant_id"],
            arguments=_from_json(row["arguments"]) or {},
            requester_id=row["requester_id"],
            status=ApprovalStatus(row["status"]),
            comment=row["comment"],
            created_at=_from_iso(row["created_at"]),
            resolved_at=_from_iso(row["resolved_at"]),
        )

    async def save(self, approval: ApprovalRequest) -> None:
        conn = await self._ensure_conn()

        def _save() -> None:
            conn.execute(
                """
                INSERT OR REPLACE INTO approvals
                    (approval_id, run_id, tool_name, tenant_id,
                     arguments, requester_id, status, comment,
                     created_at, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approval.approval_id,
                    approval.run_id,
                    approval.tool_name,
                    approval.tenant_id,
                    _to_json(approval.arguments),
                    approval.requester_id,
                    approval.status.value,
                    approval.comment,
                    _to_iso(approval.created_at),
                    _to_iso(approval.resolved_at),
                ),
            )
            conn.commit()

        await asyncio.to_thread(_save)

    async def get(self, approval_id: str) -> ApprovalRequest | None:
        conn = await self._ensure_conn()

        def _get() -> sqlite3.Row | None:
            return conn.execute(
                "SELECT * FROM approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()

        row = await asyncio.to_thread(_get)
        if row is None:
            return None
        return self._row_to_approval(row)

    async def list_pending(
        self,
        *,
        tenant_id: str | None = None,
    ) -> list[ApprovalRequest]:
        conn = await self._ensure_conn()

        def _list() -> list[sqlite3.Row]:
            if tenant_id is not None:
                rows = conn.execute(
                    "SELECT * FROM approvals WHERE status = 'PENDING' AND tenant_id = ?",
                    (tenant_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM approvals WHERE status = 'PENDING'"
                ).fetchall()
            return rows

        rows = await asyncio.to_thread(_list)
        return [self._row_to_approval(r) for r in rows]

    async def update(self, approval: ApprovalRequest) -> None:
        await self.save(approval)

    async def delete(self, approval_id: str) -> None:
        conn = await self._ensure_conn()

        def _delete() -> None:
            conn.execute(
                "DELETE FROM approvals WHERE approval_id = ?",
                (approval_id,),
            )
            conn.commit()

        await asyncio.to_thread(_delete)


class SqliteAgentRepository(AgentRepository):

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    async def close(self) -> None:
        if self._conn is not None:
            await asyncio.to_thread(self._conn.close)
            self._conn = None

    async def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = await asyncio.to_thread(
                self._create_connection
            )
        return self._conn

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT NOT NULL,
                version TEXT NOT NULL,
                name TEXT NOT NULL,
                model TEXT NOT NULL,
                system_prompt TEXT DEFAULT '',
                tools TEXT DEFAULT '[]',
                skills TEXT DEFAULT '[]',
                middleware TEXT DEFAULT '[]',
                filesystem_permissions TEXT DEFAULT '[]',
                interrupt_on TEXT DEFAULT '{}',
                use_checkpointer INTEGER DEFAULT 1,
                metadata TEXT DEFAULT '{}',
                PRIMARY KEY (agent_id, version)
            )
            """
        )
        conn.commit()
        return conn

    def _row_to_config(self, row: sqlite3.Row) -> AgentConfig:
        return AgentConfig(
            agent_id=row["agent_id"],
            version=row["version"],
            name=row["name"],
            model=row["model"],
            system_prompt=row["system_prompt"] or "",
            tools=_from_json(row["tools"]) or [],
            skills=_from_json(row["skills"]) or [],
            middleware=_from_json(row["middleware"]) or [],
            filesystem_permissions=_from_json(row["filesystem_permissions"]) or [],
            interrupt_on=_from_json(row["interrupt_on"]) or {},
            use_checkpointer=bool(row["use_checkpointer"]),
            metadata=_from_json(row["metadata"]) or {},
        )

    async def save(self, config: AgentConfig) -> None:
        conn = await self._ensure_conn()

        def _save() -> None:
            conn.execute(
                """
                INSERT OR REPLACE INTO agents
                    (agent_id, version, name, model, system_prompt,
                     tools, skills, middleware, filesystem_permissions,
                     interrupt_on, use_checkpointer, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    config.agent_id,
                    config.version,
                    config.name,
                    config.model,
                    config.system_prompt,
                    _to_json(config.tools),
                    _to_json(config.skills),
                    _to_json(config.middleware),
                    _to_json(config.filesystem_permissions),
                    _to_json(config.interrupt_on),
                    1 if config.use_checkpointer else 0,
                    _to_json(config.metadata),
                ),
            )
            conn.commit()

        await asyncio.to_thread(_save)

    async def get(
        self,
        agent_id: str,
        version: str | None = None,
    ) -> AgentConfig | None:
        conn = await self._ensure_conn()

        def _get() -> sqlite3.Row | None:
            if version is not None:
                return conn.execute(
                    "SELECT * FROM agents WHERE agent_id = ? AND version = ?",
                    (agent_id, version),
                ).fetchone()
            return conn.execute(
                "SELECT * FROM agents WHERE agent_id = ? ORDER BY version DESC LIMIT 1",
                (agent_id,),
            ).fetchone()

        row = await asyncio.to_thread(_get)
        if row is None:
            return None
        return self._row_to_config(row)

    async def list_versions(self, agent_id: str) -> list[str]:
        conn = await self._ensure_conn()

        def _list() -> list[sqlite3.Row]:
            return conn.execute(
                "SELECT version FROM agents WHERE agent_id = ? ORDER BY version",
                (agent_id,),
            ).fetchall()

        rows = await asyncio.to_thread(_list)
        return [r["version"] for r in rows]

    async def delete(self, agent_id: str, version: str) -> None:
        conn = await self._ensure_conn()

        def _delete() -> None:
            conn.execute(
                "DELETE FROM agents WHERE agent_id = ? AND version = ?",
                (agent_id, version),
            )
            conn.commit()

        await asyncio.to_thread(_delete)


class SqliteCheckpointRepository(CheckpointRepository):

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    async def close(self) -> None:
        if self._conn is not None:
            await asyncio.to_thread(self._conn.close)
            self._conn = None

    async def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = await asyncio.to_thread(
                self._create_connection
            )
        return self._conn

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT PRIMARY KEY,
                checkpoint_data TEXT NOT NULL
            )
            """
        )
        conn.commit()
        return conn

    async def save(
        self,
        thread_id: str,
        checkpoint_data: dict[str, Any],
    ) -> None:
        conn = await self._ensure_conn()

        def _save() -> None:
            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints (thread_id, checkpoint_data)
                VALUES (?, ?)
                """,
                (thread_id, _to_json(checkpoint_data)),
            )
            conn.commit()

        await asyncio.to_thread(_save)

    async def get(self, thread_id: str) -> dict[str, Any] | None:
        conn = await self._ensure_conn()

        def _get() -> sqlite3.Row | None:
            return conn.execute(
                "SELECT * FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()

        row = await asyncio.to_thread(_get)
        if row is None:
            return None
        return _from_json(row["checkpoint_data"]) or {}

    async def delete(self, thread_id: str) -> None:
        conn = await self._ensure_conn()

        def _delete() -> None:
            conn.execute(
                "DELETE FROM checkpoints WHERE thread_id = ?",
                (thread_id,),
            )
            conn.commit()

        await asyncio.to_thread(_delete)