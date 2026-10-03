import pytest

from enterprise_harness.agent import AgentConfig
from enterprise_harness.agent.runtime import AgentRuntime
from enterprise_harness.orchestration.planner import Plan, Planner
from enterprise_harness.orchestration.task import Task
from enterprise_harness.orchestration.task_graph import TaskGraph
from enterprise_harness.runtime.manager import RunManager
from enterprise_harness.runtime.models import RunStatus
from enterprise_harness.runtime.langgraph_runtime import LangGraphRuntime


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


class FixedPlanPlanner(Planner):
    def __init__(self, tasks: list[Task]):
        self._tasks = tasks

    async def plan(self, task: str) -> Plan:
        return Plan(task=task, tasks=self._tasks)


def create_runtime() -> FakeAgentRuntime:
    return FakeAgentRuntime(
        config=AgentConfig(
            agent_id="order-agent",
            name="Order Agent",
            model="test-model",
        )
    )


@pytest.mark.asyncio
async def test_run_manager_executes_task_graph():
    agent_runtime = create_runtime()

    graph = create_graph()
    planner = FixedPlanPlanner(tasks=list(graph.tasks()))

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
    class FailingAgentRuntime(AgentRuntime):
        async def build_agent(self, run_context=None):
            from langgraph.graph import StateGraph

            async def execute(state: dict) -> dict:
                ctx = {}
                if run_context is not None:
                    ctx = run_context.context
                if ctx.get("task_id") == "T2":
                    raise RuntimeError("inventory service unavailable")
                task = run_context.task if run_context else ""
                result = {
                    "answer": f"completed: {task}",
                }
                state["messages"] = [type("msg", (), {"content": result})()]
                return state

            builder = StateGraph(dict)
            builder.add_node("exec", execute)
            builder.set_entry_point("exec")
            builder.set_finish_point("exec")
            return builder.compile()

    agent_runtime = FailingAgentRuntime(
        config=AgentConfig(
            agent_id="order-agent",
            name="Order Agent",
            model="test-model",
        )
    )

    graph = create_graph()
    planner = FixedPlanPlanner(tasks=list(graph.tasks()))

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

    assert result.status == RunStatus.FAILED
    assert "T2" in result.error
    assert "inventory service unavailable" in result.error