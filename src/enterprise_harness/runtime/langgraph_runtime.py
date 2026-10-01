from __future__ import annotations

from typing import Any


class LangGraphRuntime:
    """
    Harness 对 LangGraph Durable Runtime 的封装。

    不再自建 wrapper graph，而是直接包装 DeepAgentRuntime
    产出的 CompiledStateGraph（即 DeepAgents 的原生 Agent Loop）。

    当 DeepAgentToolAdapter 内部捕获到 ApprovalRequiredError 时，
    它会调用 langgraph.types.interrupt() 暂停 DeepAgents loop。
    LangGraphRuntime 检测 __interrupt__ 标记 → Harness 把 Run 状态改为 WAITING_APPROVAL。
    人类审批通过后，RunManager.resume_run() 用 Command(resume={approved:True})
    让 DeepAgents loop 继续 → adapter 带 approval_id 重试 gateway.execute()。
    """

    def __init__(
        self,
        compiled_graph,
    ):
        self.graph = compiled_graph

    async def run(
        self,
        run_id: str,
        input_data: dict[str, Any],
    ):
        config = self._config(run_id)
        return await self.graph.ainvoke(input_data, config=config)

    async def resume(
        self,
        run_id: str,
        value: Any,
    ):
        from langgraph.types import Command

        config = self._config(run_id)
        return await self.graph.ainvoke(Command(resume=value), config=config)

    async def get_state(
        self,
        run_id: str,
    ):
        config = self._config(run_id)
        return self.graph.get_state(config=config)

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
        return bool(result.get("__interrupt__"))

    @staticmethod
    def extract_interrupt_value(
        result: dict[str, Any],
    ) -> Any:
        interrupts = result.get("__interrupt__")
        if not interrupts:
            return None

        interrupt_value = interrupts[0]
        value = getattr(interrupt_value, "value", interrupt_value)
        return value

    @staticmethod
    def extract_approval(
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        value = LangGraphRuntime.extract_interrupt_value(result)
        if isinstance(value, dict) and value.get("type") == "approval_required":
            return value
        return None

    @staticmethod
    def extract_result(
        result: dict[str, Any],
    ) -> Any:
        if isinstance(result, dict):
            messages = result.get("messages")
            if messages:
                last = messages[-1]
                if hasattr(last, "content"):
                    return last.content
                return last
            for key in ("result", "output", "answer"):
                if key in result:
                    return result[key]
        return result
