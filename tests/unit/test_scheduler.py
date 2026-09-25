import asyncio

import pytest

from enterprise_harness.orchestration.scheduler import Scheduler
from enterprise_harness.orchestration.task import Task, TaskStatus
from enterprise_harness.orchestration.task_graph import TaskGraph


def build_graph() -> TaskGraph:
    graph = TaskGraph()

    graph.add_task(
        Task(
            task_id="T1",
            name="查询订单",
        )
    )

    graph.add_task(
        Task(
            task_id="T2",
            name="查询库存",
        )
    )

    graph.add_task(
        Task(
            task_id="T3",
            name="查询物流",
        )
    )

    graph.add_task(
        Task(
            task_id="T4",
            name="综合分析",
        )
    )

    graph.add_task(
        Task(
            task_id="T5",
            name="更新订单",
        )
    )

    graph.add_dependency("T4", "T1")
    graph.add_dependency("T4", "T2")
    graph.add_dependency("T4", "T3")

    graph.add_dependency("T5", "T4")

    return graph


@pytest.mark.asyncio
async def test_scheduler_executes_task_graph():
    graph = build_graph()
    executed = []

    async def executor(task: Task):
        executed.append(task.task_id)
        await asyncio.sleep(0.01)
        return {
            "task_id": task.task_id,
            "success": True,
        }

    scheduler = Scheduler(
        graph=graph,
        executor=executor,
    )

    result = await scheduler.run()

    assert result.is_completed() is True

    assert set(executed) == {
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
    }

    assert all(
        task.status == TaskStatus.SUCCESS
        for task in result.tasks()
    )


@pytest.mark.asyncio
async def test_scheduler_respects_dependencies():
    graph = build_graph()
    execution_order = []

    async def executor(task: Task):
        execution_order.append(task.task_id)
        return task.task_id

    scheduler = Scheduler(
        graph=graph,
        executor=executor,
    )

    await scheduler.run()

    assert execution_order.index("T1") < execution_order.index("T4")
    assert execution_order.index("T2") < execution_order.index("T4")
    assert execution_order.index("T3") < execution_order.index("T4")

    assert execution_order.index("T4") < execution_order.index("T5")


@pytest.mark.asyncio
async def test_scheduler_stops_after_failure():
    graph = build_graph()

    async def executor(task: Task):
        if task.task_id == "T2":
            raise RuntimeError("inventory service unavailable")

        return task.task_id

    scheduler = Scheduler(
        graph=graph,
        executor=executor,
    )

    result = await scheduler.run()

    assert result.has_failed() is True

    assert result.get_task("T2").status == TaskStatus.FAILED

    assert result.get_task("T4").status == TaskStatus.PENDING
    assert result.get_task("T5").status == TaskStatus.PENDING


@pytest.mark.asyncio
async def test_scheduler_respects_max_concurrency():
    graph = TaskGraph()

    for task_id in ["T1", "T2", "T3"]:
        graph.add_task(
            Task(
                task_id=task_id,
                name=task_id,
            )
        )

    current = 0
    max_current = 0

    async def executor(task: Task):
        nonlocal current, max_current

        current += 1
        max_current = max(max_current, current)

        await asyncio.sleep(0.02)

        current -= 1

        return task.task_id

    scheduler = Scheduler(
        graph=graph,
        executor=executor,
        max_concurrency=2,
    )

    await scheduler.run()

    assert max_current <= 2