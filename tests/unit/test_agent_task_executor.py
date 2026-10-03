import pytest

from enterprise_harness.agent.runtime import AgentRuntime
from enterprise_harness.orchestration.executor import AgentTaskExecutor
from enterprise_harness.orchestration.task import Task
from enterprise_harness.runtime.context import RunContext
from tests.helpers import FakeLangGraphRuntimeForAgent


class FakeAgentRuntime(AgentRuntime):
    async def build_agent(self, run_context=None):
        from langgraph.graph import StateGraph

        async def execute(state: dict) -> dict:
            task = "unknown"
            if run_context is not None:
                task = run_context.task
            ctx = run_context.context if run_context else {}
            result = {
                "task": task,
                "context": ctx,
            }
            state["messages"] = [type("msg", (), {"content": result})()]
            return state

        builder = StateGraph(dict)
        builder.add_node("exec", execute)
        builder.set_entry_point("exec")
        builder.set_finish_point("exec")
        return builder.compile()


def create_context() -> RunContext:
    return RunContext(
        run_id="run-001",
        agent_id="order-agent",
        task="处理订单",
        context={
            "tenant_id": "tenant-001",
        },
    )


@pytest.mark.asyncio
async def test_execute_task_with_agent_runtime():
    runtime = FakeAgentRuntime(
        config=None,
    )

    context = create_context()

    executor = AgentTaskExecutor(
        runtime=FakeLangGraphRuntimeForAgent(runtime),
        run_context=context,
    )

    task = Task(
        task_id="T1",
        name="查询订单",
        description="查询订单 1001",
        input={
            "order_id": "1001",
        },
    )

    result = await executor.execute(task)

    assert result["task"] == "查询订单 1001"

    assert result["context"]["tenant_id"] == "tenant-001"
    assert result["context"]["task_id"] == "T1"
    assert result["context"]["task_name"] == "查询订单"
    assert result["context"]["task_input"] == {
        "order_id": "1001",
    }


@pytest.mark.asyncio
async def test_executor_is_callable():
    runtime = FakeAgentRuntime(
        config=None,
    )

    context = create_context()

    executor = AgentTaskExecutor(
        runtime=FakeLangGraphRuntimeForAgent(runtime),
        run_context=context,
    )

    task = Task(
        task_id="T1",
        name="查询订单",
        description="查询订单 1001",
    )

    result = await executor(task)

    assert result["task"] == "查询订单 1001"