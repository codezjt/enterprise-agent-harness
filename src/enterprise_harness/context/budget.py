from typing import Any

from .models import ContextItem


_PRIORITY_NAMES: dict[int, str] = {
    0: "system_policy",
    1: "task",
    2: "tool",
    3: "rag",
    4: "memory",
    5: "conversation",
}

_MAX_PRIORITY = 5


class TokenBudget:
    def __init__(self, max_tokens: int):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")

        self.max_tokens = max_tokens

    def estimate(self, content: Any) -> int:
        return len(str(content))

    def fit(
        self,
        items: list[ContextItem],
    ) -> list[ContextItem]:
        items = sorted(items, key=lambda x: x.priority)

        result: list[ContextItem] = []
        used_tokens = 0

        for item in items:
            tokens = self.estimate(item.content)

            if used_tokens + tokens <= self.max_tokens:
                result.append(item)
                used_tokens += tokens
                continue

            remaining = self.max_tokens - used_tokens
            if remaining > 100 and item.priority <= 3:
                truncated = self._truncate(item, remaining)
                result.append(truncated)
                used_tokens += self.estimate(truncated.content)
                break

            break

        return result

    def fit_with_compression(
        self,
        items: list[ContextItem],
    ) -> list[ContextItem]:
        result = self.fit(items)
        included_ids = {id(item) for item in result}

        removed = [item for item in items if id(item) not in included_ids]
        if not removed:
            return result

        compressed = self._compress(items, self.max_tokens)
        return compressed

    @staticmethod
    def _truncate(item: ContextItem, max_chars: int) -> ContextItem:
        content_str = str(item.content)
        if len(content_str) <= max_chars:
            return item

        truncated = content_str[:max_chars - 3] + "..."
        return ContextItem(
            priority=item.priority,
            source=item.source,
            content=truncated,
            metadata={**item.metadata, "truncated": True},
        )

    def _compress(
        self,
        items: list[ContextItem],
        budget: int,
    ) -> list[ContextItem]:
        items = sorted(items, key=lambda x: x.priority)
        result: list[ContextItem] = []

        p0_p1_items = [i for i in items if i.priority <= 1]
        other_items = [i for i in items if i.priority > 1]

        for item in p0_p1_items:
            result.append(item)

        used = sum(self.estimate(i.content) for i in p0_p1_items)
        remaining = budget - used

        if remaining <= 0:
            return result

        other_items.sort(key=lambda x: x.priority)

        allocated: dict[int, int] = {}
        for p in range(2, _MAX_PRIORITY + 1):
            allocated[p] = remaining // (_MAX_PRIORITY - 1)

        for p in range(2, _MAX_PRIORITY + 1):
            pr_items = [i for i in other_items if i.priority == p]
            pr_budget = allocated.get(p, 0)

            for item in pr_items:
                item_tokens = self.estimate(item.content)
                if item_tokens <= pr_budget:
                    result.append(item)
                    pr_budget -= item_tokens
                elif pr_budget > 100:
                    truncated = self._truncate(item, pr_budget)
                    result.append(truncated)
                    pr_budget -= self.estimate(truncated.content)
                else:
                    break

        result.sort(key=lambda x: x.priority)
        return result