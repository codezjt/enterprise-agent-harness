from typing import Any


class LongTermMemory:
    def __init__(self):
        self._items: list[Any] = []

    def store(self, item: Any) -> None:
        self._items.append(item)

    def recall(self, limit: int = 10) -> list[Any]:
        if limit <= 0:
            return []

        return self._items[-limit:]