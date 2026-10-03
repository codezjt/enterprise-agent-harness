import pytest

from enterprise_harness.agent import AgentConfig
from enterprise_harness.agent.runtime import AgentRuntime
from enterprise_harness.orchestration.planner import Plan, Planner
from enterprise_harness.orchestration.task import Task
from enterprise_harness.runtime.manager import RunManager
from enterprise_harness.runtime.models import RunStatus
from enterprise_harness.runtime.langgraph_runtime import LangGraphRuntime
from tests.helpers import FakeLangGraphRuntimeForAgent


class FakeAgentRuntime(AgentRuntime):
    async def build_agent(self, run_context=None):
        from langgraph.graph import StateGraph

        async def execute(state: dict) -> dict:
            task = "unknown"
            ctx = {}
            if run_context is not None:
                task = run_context.task
                ctx = run_context.context
            result = {
                "answer": f"completed: {task}",
                "task_id": ctx.get("task_id", ""),
            }
            state["messages"] = [type("msg", (), {"content": result})()]
            return state

        builder = StateGraph(dict)
        builder.add_node("exec", execute)
        builder.set_entry_point("exec")
        builder.set_finish_point("exec")
        return builder.compile()


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
    agent_runtime = create_runtime()
    runtime = FakeLangGraphRuntimeForAgent(agent_runtime)

    planner = FakePlanner()
    langgraph_rt = LangGraphRuntime(
        deepagent_runtime=agent_runtime,
        planner=planner,
    )

    manager = RunManager()

    run = await manager.create_run(
        agent_id="order-agent",
        task="处理订单 1001",
    )

    result = await manager.start_run(
        run_id=run.run_id,
        runtime=langgraph_rt,
    )

    assert result.status == RunStatus.COMPLETED

    assert result.result["T1"]["answer"] == "completed: 查询订单 1001"
    assert result.result["T2"]["answer"] == "completed: 查询库存 SKU-001"
    assert result.result["T3"]["answer"] == "completed: 综合订单和库存结果"

    assert result.started_at is not None
    assert result.completed_at is not None


@pytest.mark.asyncio
async def test_planner_output_controls_task_graph():
    agent_runtime = create_runtime()

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

    langgraph_rt = LangGraphRuntime(
        deepagent_runtime=agent_runtime,
        planner=SingleTaskPlanner(),
    )

    manager = RunManager()

    run = await manager.create_run(
        agent_id="order-agent",
        task="处理订单 1001",
    )

    result = await manager.start_run(
        run_id=run.run_id,
        runtime=langgraph_rt,
    )

    assert result.status == RunStatus.COMPLETED
    assert set(result.result.keys()) == {"ONLY"}
    assert result.result["ONLY"]["answer"] == "completed: 处理订单 1001"


@pytest.mark.asyncio
async def test_run_manager_rejects_invalid_plan_before_execution():
    agent_runtime = create_runtime()

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

    langgraph_rt = LangGraphRuntime(
        deepagent_runtime=agent_runtime,
        planner=InvalidPlanner(),
    )

    manager = RunManager()

    run = await manager.create_run(
        agent_id="order-agent",
        task="处理订单 1001",
    )

    result = await manager.start_run(
        run_id=run.run_id,
        runtime=langgraph_rt,
    )

    assert result.status == RunStatus.FAILED
    assert "UNKNOWN" in result.error