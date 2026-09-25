import pytest

from enterprise_harness.agent import AgentConfig
from enterprise_harness.agent.runtime import AgentRuntime
from enterprise_harness.orchestration.executor import AgentTaskExecutor
from enterprise_harness.orchestration.scheduler import Scheduler
from enterprise_harness.orchestration.task import Task
from enterprise_harness.orchestration.task_graph import TaskGraph
from enterprise_harness.runtime.context import RunContext


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
            description="根据订单和库存结果进行分析",
        )
    )

    graph.add_dependency("T3", "T1")
    graph.add_dependency("T3", "T2")

    return graph


@pytest.mark.asyncio
async def test_scheduler_executes_agent_tasks():
    graph = create_graph()

    context = RunContext(
        run_id="run-001",
        agent_id="order-agent",
        task="处理订单 1001",
    )

    executor = AgentTaskExecutor(
        runtime=create_runtime(),
        run_context=context,
    )

    scheduler = Scheduler(
        graph=graph,
        executor=executor,
    )

    result = await scheduler.run()

    assert result.is_completed()

    assert result.get_task("T1").output["answer"] == (
        "completed: 查询订单 1001"
    )

    assert result.get_task("T2").output["answer"] == (
        "completed: 查询商品 SKU-001 库存"
    )

    assert result.get_task("T3").output["answer"] == (
        "completed: 根据订单和库存结果进行分析"
    )