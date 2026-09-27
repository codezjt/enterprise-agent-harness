from typing import Any

from .long_term import LongTermMemory
from .short_term import ShortTermMemory


class MemoryManager:
    def __init__(
        self,
        short_term: ShortTermMemory | None = None,
        long_term: LongTermMemory | None = None,
    ):
        self.short_term = short_term or ShortTermMemory()
        self.long_term = long_term or LongTermMemory()

    def store(
        self,
        item: Any,
        *,
        long_term: bool = False,
    ) -> None:
        if long_term:
            self.long_term.store(item)
        else:
            self.short_term.store(item)

    def recall(
        self,
        *,
        limit: int = 10,
        long_term: bool = False,
    ) -> list[Any]:
        if long_term:
            return self.long_term.recall(limit)

        return self.short_term.recall(limit)

    def clear_short_term(self) -> None:
        self.short_term.clear()