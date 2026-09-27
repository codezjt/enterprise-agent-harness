from typing import Any, Callable

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict


class RuntimeState(TypedDict, total=False):
    task: str
    context: dict[str, Any]
    result: Any
    approval: dict[str, Any] | None
    waiting_for_approval: bool


class LangGraphRuntime:
    """Harness 对 LangGraph Durable Runtime 的最小封装。"""

    def __init__(
        self,
        execute: Callable[[RuntimeState], Any],
    ):
        self.execute = execute

        graph = StateGraph(RuntimeState)

        graph.add_node(
            "execute",
            self._execute_node,
        )

        graph.add_node(
            "approval",
            self._approval_node,
        )

        graph.add_edge(
            START,
            "execute",
        )

        graph.add_edge(
            "execute",
            "approval",
        )

        graph.add_edge(
            "approval",
            END,
        )

        self.graph = graph.compile(
            checkpointer=InMemorySaver(),
        )

    async def run(
            self,
            run_id: str,
            task: str,
            context: dict[str, Any] | None = None,
    ):
        config = self._config(run_id)

        return await self.graph.ainvoke(
            {
                "task": task,
                "context": context or {},
                "result": None,
                "approval": None,
                "waiting_for_approval": False,
            },
            config=config,
        )

    async def resume(
            self,
            run_id: str,
            value: Any,
    ):
        config = self._config(run_id)

        return await self.graph.ainvoke(
            Command(resume=value),
            config=config,
        )

    @staticmethod
    def _config(run_id: str) -> dict[str, Any]:
        return {
            "configurable": {
                "thread_id": run_id,
            }
        }

    @staticmethod
    def is_interrupted(
            result: dict[str, Any],
    ) -> bool:
        return bool(result.get("__interrupt__"))

    @staticmethod
    def extract_result(
            result: dict[str, Any],
    ) -> Any:
        return result.get("result")

    async def _execute_node(self, state: RuntimeState):
        return await self.execute(state)

    async def _approval_node(
        self,
        state: RuntimeState,
    ):
        approval = state.get("approval")

        if not approval:
            return {
                "waiting_for_approval": False,
            }

        if approval.get("status") != "REQUIRE_APPROVAL":
            return {
                "waiting_for_approval": False,
            }

        decision = interrupt(
            {
                "type": "approval_required",
                "message": approval.get(
                    "message",
                    "需要审批后才能继续执行",
                ),
                "approval": approval,
            }
        )

        return {
            "approval": {
                **approval,
                "decision": decision,
            },
            "waiting_for_approval": False,
        }