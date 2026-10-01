from __future__ import annotations

from enum import Enum


class Permission(str, Enum):
    TOOL_EXECUTE = "tool:execute"
    TOOL_READ = "tool:read"
    AGENT_CREATE = "agent:create"
    AGENT_READ = "agent:read"
    RUN_CREATE = "run:create"
    RUN_READ = "run:read"
    RUN_APPROVE = "run:approve"
    RUN_REJECT = "run:reject"


class Authorization:
    def __init__(self):
        self._role_permissions: dict[str, set[Permission]] = {}

    def grant(self, role: str, permission: Permission) -> None:
        if role not in self._role_permissions:
            self._role_permissions[role] = set()
        self._role_permissions[role].add(permission)

    def revoke(self, role: str, permission: Permission) -> None:
        if role in self._role_permissions:
            self._role_permissions[role].discard(permission)

    def has_permission(self, role: str, permission: Permission) -> bool:
        perms = self._role_permissions.get(role, set())
        return permission in perms

    def get_permissions(self, role: str) -> set[Permission]:
        return self._role_permissions.get(role, set()).copy()