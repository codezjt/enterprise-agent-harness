import asyncio

import pytest

from enterprise_harness.orchestration.scheduler import Scheduler
from enterprise_harness.orchestration.task import (
    Task,
    TaskStatus,
)
from enterprise_harness.orchestration.task_graph import (
    TaskGraph,
)


@pytest.mark.asyncio
async def test_scheduler_injects_dependency_results():
    graph = TaskGraph()

    graph.add_task(
        Task(
            task_id="T1",
            name="query_order",
            description="查询订单",
        )
    )

    graph.add_task(
        Task(
            task_id="T2",
            name="analyze_order",
            description="分析订单",
            dependencies=["T1"],
        )
    )

    received = {}

    async def executor(task: Task):
        if task.task_id == "T1":
            return {
                "order_id": "1001",
                "quantity": 100,
            }

        if task.task_id == "T2":
            received.update(
                task.input
            )
            return {
                "analysis": "inventory shortage"
            }

        raise AssertionError(
            f"Unexpected task: {task.task_id}"
        )

    scheduler = Scheduler(
        graph=graph,
        executor=executor,
    )

    result = await scheduler.run()

    assert result.is_completed()

    assert received[
        "dependency_results"
    ] == {
        "T1": {
            "order_id": "1001",
            "quantity": 100,
        }
    }


@pytest.mark.asyncio
async def test_scheduler_runs_independent_tasks_in_parallel():
    graph = TaskGraph()

    graph.add_task(
        Task(
            task_id="T1",
            name="query_order",
        )
    )

    graph.add_task(
        Task(
            task_id="T2",
            name="query_inventory",
        )
    )

    running = 0
    max_running = 0

    async def executor(task: Task):
        nonlocal running
        nonlocal max_running

        running += 1
        max_running = max(
            max_running,
            running,
        )

        await asyncio.sleep(0.05)

        running -= 1

        return task.task_id

    scheduler = Scheduler(
        graph=graph,
        executor=executor,
        max_concurrency=2,
    )

    result = await scheduler.run()

    assert result.is_completed()

    assert max_running == 2


@pytest.mark.asyncio
async def test_scheduler_respects_max_concurrency():
    graph = TaskGraph()

    for index in range(4):
        graph.add_task(
            Task(
                task_id=f"T{index}",
                name=f"task-{index}",
            )
        )

    running = 0
    max_running = 0

    async def executor(task: Task):
        nonlocal running
        nonlocal max_running

        running += 1
        max_running = max(
            max_running,
            running,
        )

        await asyncio.sleep(0.02)

        running -= 1

        return task.task_id

    scheduler = Scheduler(
        graph=graph,
        executor=executor,
        max_concurrency=2,
    )

    await scheduler.run()

    assert max_running <= 2


@pytest.mark.asyncio
async def test_failed_dependency_stops_downstream_tasks():
    graph = TaskGraph()

    graph.add_task(
        Task(
            task_id="T1",
            name="query_order",
        )
    )

    graph.add_task(
        Task(
            task_id="T2",
            name="analyze",
            dependencies=["T1"],
        )
    )

    executed = []

    async def executor(task: Task):
        executed.append(task.task_id)

        if task.task_id == "T1":
            raise RuntimeError(
                "temporary timeout"
            )

        return "unexpected"

    scheduler = Scheduler(
        graph=graph,
        executor=executor,
    )

    result = await scheduler.run()

    assert result.has_failed()

    assert "T1" in executed
    assert "T2" not in executed

    assert (
        result.get_task("T1").status
        == TaskStatus.FAILED
    )

    assert (
        result.get_task("T2").status
        == TaskStatus.PENDING
    )