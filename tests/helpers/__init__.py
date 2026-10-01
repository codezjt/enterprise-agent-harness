from __future__ import annotations

from typing import Any, Callable, Coroutine

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph


def build_fake_compiled_graph(
    execute_fn: Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]],
):
    builder = StateGraph(dict)
    builder.add_node("exec", execute_fn)
    builder.set_entry_point("exec")
    builder.set_finish_point("exec")
    return builder.compile(checkpointer=MemorySaver())


class FakeLangGraphRuntimeForAgent:
    def __init__(self, agent_runtime):
        self.agent_runtime = agent_runtime

    async def run(self, run_id: str, input_data: dict):
        messages = input_data.get("messages", [{"content": ""}])
        content = messages[0].get("content", "") if messages else ""
        context = input_data.get("context", {})
        return await self.agent_runtime.run(task=content, context=context)

    async def resume(self, run_id: str, value):
        return {"result": "resumed"}

    @staticmethod
    def is_interrupted(result):
        return bool(result.get("__interrupt__"))

    @staticmethod
    def extract_result(result):
        if isinstance(result, dict):
            return result.get("result", result)
        return result

    @staticmethod
    def extract_approval(result):
        return None