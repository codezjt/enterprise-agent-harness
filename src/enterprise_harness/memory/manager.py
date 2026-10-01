from typing import Any

from .long_term import LongTermMemory
from .short_term import MemoryScope, ScopedShortTermMemory, ShortTermMemory


class MemoryManager:
    def __init__(
        self,
        short_term: ShortTermMemory | ScopedShortTermMemory | None = None,
        long_term: LongTermMemory | None = None,
    ):
        self.short_term = short_term or ScopedShortTermMemory()
        self.long_term = long_term or LongTermMemory()

    async def store(
        self,
        item: Any,
        *,
        long_term: bool = False,
        scope: MemoryScope | None = None,
    ) -> None:
        if long_term:
            self.long_term.store(item)
        elif isinstance(self.short_term, ScopedShortTermMemory):
            self.short_term.store(item, scope=scope)
        else:
            self.short_term.store(item)

    async def recall(
        self,
        *,
        limit: int = 10,
        long_term: bool = False,
        scope: MemoryScope | None = None,
    ) -> list[Any]:
        if long_term:
            return self.long_term.recall(limit)

        if isinstance(self.short_term, ScopedShortTermMemory):
            return self.short_term.recall(limit=limit, scope=scope)

        return self.short_term.recall(limit)

    async def clear(
        self,
        scope: MemoryScope | None = None,
    ) -> None:
        if isinstance(self.short_term, ScopedShortTermMemory):
            self.short_term.clear(scope=scope)
        else:
            self.short_term.clear()

    def clear_short_term(self) -> None:
        if isinstance(self.short_term, ScopedShortTermMemory):
            self.short_term.clear()
        else:
            self.short_term.clear()