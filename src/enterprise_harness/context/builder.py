from typing import Any

from pydantic import BaseModel, Field


class ContextItem(BaseModel):
    priority: int
    source: str
    content: Any
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextBuilder:
    def build(
        self,
        *,
        system_policy: Any = None,
        task: Any = None,
        tool_results: list[Any] | None = None,
        rag_results: list[Any] | None = None,
        memory_results: list[Any] | None = None,
        conversation: list[Any] | None = None,
    ) -> list[ContextItem]:
        items: list[ContextItem] = []

        if system_policy is not None:
            items.append(
                ContextItem(
                    priority=0,
                    source="system_policy",
                    content=system_policy,
                )
            )

        if task is not None:
            items.append(
                ContextItem(
                    priority=1,
                    source="task",
                    content=task,
                )
            )

        for result in tool_results or []:
            items.append(
                ContextItem(
                    priority=2,
                    source="tool",
                    content=result,
                )
            )

        for result in rag_results or []:
            items.append(
                ContextItem(
                    priority=3,
                    source="rag",
                    content=result,
                )
            )

        for result in memory_results or []:
            items.append(
                ContextItem(
                    priority=4,
                    source="memory",
                    content=result,
                )
            )

        for message in conversation or []:
            items.append(
                ContextItem(
                    priority=5,
                    source="conversation",
                    content=message,
                )
            )

        return items