import pytest

from enterprise_harness.agent import AgentConfig
from enterprise_harness.agent.runtime import AgentRuntime
from enterprise_harness.orchestration.planner import Plan, Planner
from enterprise_harness.orchestration.task import Task
from enterprise_harness.runtime.manager import RunManager
from enterprise_harness.runtime.models import RunStatus
from tests.helpers import FakeLangGraphRuntimeForAgent


class FakeAgentRuntime(AgentRuntime):

    async def run(self, task: str, context=None):
        return {
            "answer": f"completed: {task}",
            "task_id": context["task_id"],
        }


class FakePlanner(Planner):

    async def plan(self, task: str) -> Plan:
        return Plan(
            task=task,
            tasks=[
                Task(
                    task_id="T1",
                    name="查询订单",
                    description="查询订单 1001",
                ),
                Task(
                    task_id="T2",
                    name="查询库存",
                    description="查询库存 SKU-001",
                ),
                Task(
                    task_id="T3",
                    name="综合分析",
                    description="综合订单和库存结果",
                    dependencies=["T1", "T2"],
                ),
            ],
        )


def create_runtime() -> FakeAgentRuntime:
    return FakeAgentRuntime(
        config=AgentConfig(
            agent_id="order-agent",
            name="Order Agent",
            model="test-model",
        )
    )


@pytest.mark.asyncio
async def test_run_manager_executes_planned_run():
    runtime = create_runtime()
    manager = RunManager(langgraph_runtime=FakeLangGraphRuntimeForAgent(runtime))

    run = await manager.create_run(
        agent_id="order-agent",
        task="处理订单 1001",
    )

    result = await manager.start_planned_run(
        run_id=run.run_id,
        runtime=runtime,
        planner=FakePlanner(),
    )

    assert result.status == RunStatus.COMPLETED

    assert result.result["T1"]["answer"] == "completed: 查询订单 1001"
    assert result.result["T2"]["answer"] == "completed: 查询库存 SKU-001"
    assert result.result["T3"]["answer"] == "completed: 综合订单和库存结果"

    assert result.started_at is not None
    assert result.completed_at is not None


@pytest.mark.asyncio
async def test_planner_output_controls_task_graph():
    runtime = create_runtime()
    manager = RunManager(langgraph_runtime=FakeLangGraphRuntimeForAgent(runtime))

    run = await manager.create_run(
        agent_id="order-agent",
        task="处理订单 1001",
    )

    class SingleTaskPlanner(Planner):

        async def plan(self, task: str) -> Plan:
            return Plan(
                task=task,
                tasks=[
                    Task(
                        task_id="ONLY",
                        name="执行任务",
                        description=task,
                    )
                ],
            )

    result = await manager.start_planned_run(
        run_id=run.run_id,
        runtime=runtime,
        planner=SingleTaskPlanner(),
    )

    assert result.status == RunStatus.COMPLETED
    assert set(result.result.keys()) == {"ONLY"}
    assert result.result["ONLY"]["answer"] == "completed: 处理订单 1001"


@pytest.mark.asyncio
async def test_run_manager_rejects_invalid_plan_before_execution():
    runtime = create_runtime()
    manager = RunManager(langgraph_runtime=FakeLangGraphRuntimeForAgent(runtime))

    run = await manager.create_run(
        agent_id="order-agent",
        task="处理订单 1001",
    )

    class InvalidPlanner(Planner):

        async def plan(self, task: str) -> Plan:
            return Plan(
                task=task,
                tasks=[
                    Task(
                        task_id="T1",
                        name="查询订单",
                        description="查询订单 1001",
                        dependencies=["UNKNOWN"],
                    )
                ],
            )

    result = await manager.start_planned_run(
        run_id=run.run_id,
        runtime=runtime,
        planner=InvalidPlanner(),
    )

    assert result.status == RunStatus.FAILED

    assert "UNKNOWN" in result.error