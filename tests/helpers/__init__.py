from __future__ import annotations

from typing import Any, Callable, Coroutine

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from enterprise_harness.runtime.context import RunContext
from enterprise_harness.runtime.result import RuntimeResult


def build_fake_compiled_graph(
    execute_fn: Callable[[dict[str, Any]], Coroutine[Any, Any, dict[str, Any]]],
    use_checkpointer: bool = False,
):
    builder = StateGraph(dict)
    builder.add_node("exec", execute_fn)
    builder.set_entry_point("exec")
    builder.set_finish_point("exec")
    if use_checkpointer:
        return builder.compile(checkpointer=MemorySaver())
    return builder.compile()


class FakeLangGraphRuntimeForAgent:
    def __init__(self, agent_runtime):
        self.agent_runtime = agent_runtime

    async def run(self, context: RunContext) -> RuntimeResult:
        try:
            agent = self.agent_runtime.build_agent(run_context=context)

            input_data = {
                "messages": [
                    {
                        "role": "user",
                        "content": context.task,
                    }
                ],
                "context": context.context,
            }

            invoke_config = {"configurable": {"thread_id": context.run_id}}
            result = await agent.ainvoke(input_data, invoke_config)
            return RuntimeResult.completed(result=result)
        except Exception as exc:
            return RuntimeResult.failed(str(exc))

    async def resume(self, context: RunContext, value):
        try:
            agent = self.agent_runtime.build_agent(run_context=context)

            input_data = {
                "messages": [
                    {
                        "role": "user",
                        "content": context.task,
                    }
                ],
                "context": context.context,
            }

            invoke_config = {"configurable": {"thread_id": context.run_id}}
            result = await agent.ainvoke(input_data, invoke_config)
            if isinstance(result, dict):
                result["resume_value"] = value
            return RuntimeResult.completed(result=result)
        except Exception as exc:
            return RuntimeResult.failed(str(exc))

    @staticmethod
    def is_interrupted(result):
        return False

    @staticmethod
    def extract_result(result):
        if isinstance(result, dict):
            return result.get("result", result)
        return result

    @staticmethod
    def extract_approval(result):
        return None