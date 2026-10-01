from typing import Any

from pydantic import BaseModel, Field


class ShortTermMemory:
    def __init__(self):
        self._items: list[Any] = []

    def store(self, item: Any) -> None:
        self._items.append(item)

    def recall(self, limit: int = 10) -> list[Any]:
        if limit <= 0:
            return []

        return self._items[-limit:]

    def clear(self) -> None:
        self._items.clear()


class MemoryScope(BaseModel):
    tenant_id: str = "default"
    user_id: str = ""
    agent_id: str = ""
    run_id: str = ""


class ScopedShortTermMemory:
    def __init__(self):
        self._items: list[dict[str, Any]] = []

    def store(self, item: Any, scope: MemoryScope | None = None) -> None:
        self._items.append({
            "item": item,
            "scope": (scope or MemoryScope()).model_dump(),
        })

    def recall(
        self,
        limit: int = 10,
        scope: MemoryScope | None = None,
    ) -> list[Any]:
        filtered = self._items
        if scope is not None:
            filtered = [
                entry
                for entry in self._items
                if self._matches_scope(entry.get("scope", {}), scope)
            ]

        if limit <= 0:
            return []

        recent = filtered[-limit:]
        return [entry["item"] for entry in recent]

    def clear(self, scope: MemoryScope | None = None) -> None:
        if scope is None:
            self._items.clear()
        else:
            self._items = [
                entry
                for entry in self._items
                if not self._matches_scope(entry.get("scope", {}), scope)
            ]

    @staticmethod
    def _matches_scope(entry_scope: dict, query_scope: MemoryScope) -> bool:
        if query_scope.tenant_id and query_scope.tenant_id != "default":
            if entry_scope.get("tenant_id") != query_scope.tenant_id:
                return False
        if query_scope.user_id:
            if entry_scope.get("user_id") != query_scope.user_id:
                return False
        if query_scope.agent_id:
            if entry_scope.get("agent_id") != query_scope.agent_id:
                return False
        if query_scope.run_id:
            if entry_scope.get("run_id") != query_scope.run_id:
                return False
        return True