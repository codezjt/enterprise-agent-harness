from typing import Any

from .models import ContextItem


class TokenBudget:
    def __init__(self, max_tokens: int):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")

        self.max_tokens = max_tokens

    def estimate(self, content: Any) -> int:
        """
        最小 Token 估算。

        当前阶段使用字符串长度作为近似值，
        后续可以替换成具体模型的 tokenizer。
        """
        return len(str(content))

    def fit(
        self,
        items: list[ContextItem],
    ) -> list[ContextItem]:
        result: list[ContextItem] = []
        used_tokens = 0

        for item in items:
            tokens = self.estimate(item.content)

            if used_tokens + tokens > self.max_tokens:
                break

            result.append(item)
            used_tokens += tokens

        return result