import pytest

from enterprise_harness.orchestration.planner import Plan, Planner
from enterprise_harness.orchestration.simple_planner import SimplePlanner
from enterprise_harness.orchestration.task import Task


@pytest.mark.asyncio
async def test_simple_planner_creates_plan():
    planner = SimplePlanner()

    plan = await planner.plan("查询订单 1001")

    assert isinstance(plan, Plan)
    assert plan.task == "查询订单 1001"

    assert len(plan.tasks) == 1

    task = plan.tasks[0]

    assert task.task_id == "T1"
    assert task.name == "执行任务"
    assert task.description == "查询订单 1001"


@pytest.mark.asyncio
async def test_plan_can_be_converted_to_task_graph():
    plan = Plan(
        task="处理订单 1001",
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
                description="综合订单和库存",
                dependencies=["T1", "T2"],
            ),
        ],
    )

    graph = plan.to_task_graph()

    assert len(graph.tasks()) == 3

    assert graph.get_task("T1").dependencies == []
    assert graph.get_task("T2").dependencies == []
    assert graph.get_task("T3").dependencies == ["T1", "T2"]

    ready_tasks = graph.get_ready_tasks()

    assert {task.task_id for task in ready_tasks} == {"T1", "T2"}


def test_planner_is_abstract():
    assert Planner.__abstractmethods__ == {"plan"}