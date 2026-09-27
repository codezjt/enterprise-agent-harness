from typing import Any

from .budget import TokenBudget
from .models import ContextItem
from .routing import ContextRouting
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enterprise_harness.runtime.context import RunContext

class ContextBuilder:
    def __init__(
        self,
        token_budget: TokenBudget | None = None,
        routing: ContextRouting | None = None,
    ):
        self.token_budget = token_budget
        self.routing = routing or ContextRouting()

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

        if self.routing.use_tool_results:
            for result in tool_results or []:
                items.append(
                    ContextItem(
                        priority=2,
                        source="tool",
                        content=result,
                    )
                )

        if self.routing.use_rag_results:
            for result in rag_results or []:
                items.append(
                    ContextItem(
                        priority=3,
                        source="rag",
                        content=result,
                    )
                )

        if self.routing.use_memory_results:
            for result in memory_results or []:
                items.append(
                    ContextItem(
                        priority=4,
                        source="memory",
                        content=result,
                    )
                )

        if self.routing.use_conversation:
            for message in conversation or []:
                items.append(
                    ContextItem(
                        priority=5,
                        source="conversation",
                        content=message,
                    )
                )

        if self.token_budget is not None:
            items = self.token_budget.fit(items)

        return items

    def build_from_run_context(
            self,
            run_context: "RunContext",
    ) -> list[ContextItem]:
        return self.build(
            task=run_context.task,
            conversation=run_context.context.get(
                "conversation",
                [],
            ),
            tool_results=run_context.context.get(
                "tool_results",
                [],
            ),
            rag_results=run_context.context.get(
                "rag_results",
                [],
            ),
            memory_results=run_context.context.get(
                "memory_results",
                [],
            ),
            system_policy=run_context.context.get(
                "system_policy",
            ),
        )