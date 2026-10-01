from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    event: str
    user_id: str
    agent_id: str
    run_id: str
    tool: str
    tenant_id: str = "default"
    arguments: dict[str, Any] = field(default_factory=dict)
    decision: str = ""
    approver: str | None = None
    result: Any = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "tool": self.tool,
            "tenant_id": self.tenant_id,
            "arguments": self.arguments,
            "decision": self.decision,
            "approver": self.approver,
            "result": self.result,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class AuditLogger:
    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def record(
        self,
        *,
        event: str,
        user_id: str,
        agent_id: str,
        run_id: str,
        tool: str,
        tenant_id: str = "default",
        arguments: dict[str, Any] | None = None,
        decision: str = "",
        approver: str | None = None,
        result: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        audit_event = AuditEvent(
            event=event,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            tool=tool,
            tenant_id=tenant_id,
            arguments=arguments or {},
            decision=decision,
            approver=approver,
            result=result,
            metadata=metadata or {},
        )

        self._events.append(audit_event)

        return audit_event

    def get_events(
        self,
        tenant_id: str | None = None,
    ) -> list[AuditEvent]:
        if tenant_id is not None:
            return [
                e for e in self._events
                if e.tenant_id == tenant_id
            ]
        return list(self._events)

    def get_run_events(self, run_id: str) -> list[AuditEvent]:
        return [
            event
            for event in self._events
            if event.run_id == run_id
        ]

    def get_tool_events(self, tool: str) -> list[AuditEvent]:
        return [
            event
            for event in self._events
            if event.tool == tool
        ]

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            event.to_dict()
            for event in self._events
        ]

    def clear(self) -> None:
        self._events.clear()