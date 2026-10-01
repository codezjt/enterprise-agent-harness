import pytest

from enterprise_harness.agent import AgentConfig
from enterprise_harness.agent.runtime import AgentRuntime
from enterprise_harness.orchestration.task import Task
from enterprise_harness.orchestration.task_graph import TaskGraph
from enterprise_harness.runtime.manager import RunManager
from enterprise_harness.runtime.models import RunStatus
from tests.helpers import FakeLangGraphRuntimeForAgent


class FakeAgentRuntime(AgentRuntime):
    async def run(self, task: str, context=None):
        return {
            "answer": f"completed: {task}",
            "task_id": context["task_id"],
        }


def create_runtime() -> FakeAgentRuntime:
    return FakeAgentRuntime(
        config=AgentConfig(
            agent_id="order-agent",
            name="Order Agent",
            model="test-model",
        )
    )


def create_graph() -> TaskGraph:
    graph = TaskGraph()

    graph.add_task(
        Task(
            task_id="T1",
            name="查询订单",
            description="查询订单 1001",
        )
    )

    graph.add_task(
        Task(
            task_id="T2",
            name="查询库存",
            description="查询商品 SKU-001 库存",
        )
    )

    graph.add_task(
        Task(
            task_id="T3",
            name="综合分析",
            description="综合订单和库存结果",
        )
    )

    graph.add_dependency("T3", "T1")
    graph.add_dependency("T3", "T2")

    return graph


@pytest.mark.asyncio
async def test_run_manager_executes_task_graph():
    runtime = create_runtime()

    manager = RunManager(
        langgraph_runtime=FakeLangGraphRuntimeForAgent(runtime),
    )

    run = await manager.create_run(
        agent_id="order-agent",
        task="处理订单 1001",
    )

    result = await manager.start_task_graph_run(
        run_id=run.run_id,
        runtime=runtime,
        task_graph=create_graph(),
    )

    assert result.status == RunStatus.COMPLETED

    assert result.result["T1"]["answer"] == (
        "completed: 查询订单 1001"
    )

    assert result.result["T2"]["answer"] == (
        "completed: 查询商品 SKU-001 库存"
    )

    assert result.result["T3"]["answer"] == (
        "completed: 综合订单和库存结果"
    )

    assert result.started_at is not None
    assert result.completed_at is not None


@pytest.mark.asyncio
async def test_run_manager_marks_failed_task_graph_as_failed():
    graph = create_graph()

    class FailingRuntime(AgentRuntime):
        async def run(self, task: str, context=None):
            if context["task_id"] == "T2":
                raise RuntimeError("inventory service unavailable")

            return {
                "answer": f"completed: {task}",
            }

    runtime = FailingRuntime(
        config=AgentConfig(
            agent_id="order-agent",
            name="Order Agent",
            model="test-model",
        )
    )

    manager = RunManager(
        langgraph_runtime=FakeLangGraphRuntimeForAgent(runtime),
    )

    run = await manager.create_run(
        agent_id="order-agent",
        task="处理订单 1001",
    )

    result = await manager.start_task_graph_run(
        run_id=run.run_id,
        runtime=runtime,
        task_graph=graph,
    )

    assert result.status == RunStatus.FAILED
    assert "T2" in result.error
    assert "inventory service unavailable" in result.error