import pytest

from enterprise_harness.agent.runtime import AgentRuntime
from enterprise_harness.orchestration.executor import AgentTaskExecutor
from enterprise_harness.orchestration.task import Task
from enterprise_harness.runtime.context import RunContext
from tests.helpers import FakeLangGraphRuntimeForAgent


class FakeAgentRuntime(AgentRuntime):
    async def run(self, task: str, context=None):
        return {
            "task": task,
            "context": context,
        }


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
        langgraph_runtime=FakeLangGraphRuntimeForAgent(runtime),
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
        langgraph_runtime=FakeLangGraphRuntimeForAgent(runtime),
        run_context=context,
    )

    task = Task(
        task_id="T1",
        name="查询订单",
        description="查询订单 1001",
    )

    result = await executor(task)

    assert result["task"] == "查询订单 1001"