from __future__ import annotations

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
    approval_id: str | None

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

        graph.add_conditional_edges(
            "execute",
            self._route_after_execute,
        )

        graph.add_conditional_edges(
            "approval",
            self._route_after_approval,
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
                "approval_id": None,
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
    def _config(
        run_id: str,
    ) -> dict[str, Any]:
        return {
            "configurable": {
                "thread_id": run_id,
            }
        }

    @staticmethod
    def is_interrupted(
        result: dict[str, Any],
    ) -> bool:
        return bool(
            result.get("__interrupt__")
        )

    @staticmethod
    def extract_result(
        result: dict[str, Any],
    ) -> Any:
        return result.get("result")

    @staticmethod
    def extract_approval(
        result: dict[str, Any],
    ) -> dict[str, Any] | None:

        interrupts = result.get("__interrupt__")

        if not interrupts:
            return None

        interrupt_value = interrupts[0]

        value = getattr(
            interrupt_value,
            "value",
            interrupt_value,
        )

        if not isinstance(value, dict):
            return None

        approval = value.get("approval")

        if not isinstance(approval, dict):
            return None

        return approval

    async def _execute_node(
        self,
        state: RuntimeState,
    ):
        from enterprise_harness.gateway.exceptions import (
            ApprovalRequiredError,
        )

        try:
            result = await self.execute(state)

        except ApprovalRequiredError as exc:
            approval = {
                "status": "REQUIRE_APPROVAL",
                "approval_id": exc.approval_id,
                "run_id": exc.run_id,
                "tool_name": exc.tool_name,
                "arguments": exc.arguments,
                "message": str(exc),
            }

            return {
                "approval": approval,
                "approval_id": exc.approval_id,
                "waiting_for_approval": True,
            }

        if isinstance(result, dict):
            return result

        return {
            "result": result,
        }

    async def _approval_node(
        self,
        state: RuntimeState,
    ):
        approval = state.get("approval")

        if not approval:
            return {
                "waiting_for_approval": False,
            }

        status = approval.get("status")

        if status == "APPROVED":
            return {
                "approval": approval,
                "approval_id": approval.get(
                    "approval_id"
                ),
                "waiting_for_approval": False,
            }

        if status != "REQUIRE_APPROVAL":
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

        if not isinstance(decision, dict):
            raise PermissionError(
                "Invalid approval decision"
            )

        approval_id = decision.get(
            "approval_id"
        )

        approved = decision.get(
            "approved"
        )

        if not approval_id:
            raise PermissionError(
                "approval_id is required"
            )

        if approved is not True:
            raise PermissionError(
                "Approval was not granted"
            )

        if approval_id != approval.get(
            "approval_id"
        ):
            raise PermissionError(
                "Approval id does not match"
            )

        return {
            "approval": {
                **approval,
                "status": "APPROVED",
                "approval_id": approval_id,
                "decision": decision,
            },
            "approval_id": approval_id,
            "waiting_for_approval": False,
        }

    @staticmethod
    def _route_after_execute(
        state: RuntimeState,
    ) -> str:

        if state.get("waiting_for_approval"):
            return "approval"

        return END

    @staticmethod
    def _route_after_approval(
        state: RuntimeState,
    ) -> str:

        approval = state.get("approval")

        if (
            isinstance(approval, dict)
            and approval.get("status") == "APPROVED"
        ):
            return "execute"

        return END