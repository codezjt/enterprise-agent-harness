import pytest

from enterprise_harness.orchestration.replanner import (
    ReplanRequest,
    SimpleReplanner,
)
from enterprise_harness.orchestration.task import (
    Task,
    TaskStatus,
)


@pytest.mark.asyncio
async def test_simple_replanner_retries_failed_task():
    failed_task = Task(
        task_id="T2",
        name="查询库存",
        description="查询商品库存",
        status=TaskStatus.FAILED,
        retry_count=0,
        error="库存服务暂时不可用",
    )

    request = ReplanRequest(
        original_task="查询订单并检查库存",
        failed_task=failed_task,
    )

    replanner = SimpleReplanner()

    result = await replanner.replan(request)

    assert result.reason
    assert len(result.tasks) == 1

    new_task = result.tasks[0]

    assert new_task.task_id == "T2"
    assert new_task.status == TaskStatus.PENDING
    assert new_task.retry_count == 1
    assert new_task.error is None


@pytest.mark.asyncio
async def test_replanner_preserves_completed_and_remaining_tasks():
    completed_task = Task(
        task_id="T1",
        name="查询订单",
        description="查询订单信息",
        status=TaskStatus.SUCCESS,
        output={"order_id": "1001"},
    )

    failed_task = Task(
        task_id="T2",
        name="查询库存",
        description="查询商品库存",
        status=TaskStatus.FAILED,
        retry_count=0,
        error="库存服务失败",
        dependencies=["T1"],
    )

    remaining_task = Task(
        task_id="T3",
        name="更新订单",
        description="更新订单状态",
        status=TaskStatus.PENDING,
        dependencies=["T2"],
    )

    request = ReplanRequest(
        original_task="查询订单、库存并更新订单",
        failed_task=failed_task,
        completed_tasks=[completed_task],
        remaining_tasks=[remaining_task],
    )

    replanner = SimpleReplanner()

    result = await replanner.replan(request)

    assert len(result.tasks) == 3

    task_map = {
        task.task_id: task
        for task in result.tasks
    }

    assert task_map["T1"].status == TaskStatus.SUCCESS

    assert task_map["T2"].status == TaskStatus.PENDING
    assert task_map["T2"].retry_count == 1
    assert task_map["T2"].dependencies == ["T1"]

    assert task_map["T3"].status == TaskStatus.PENDING
    assert task_map["T3"].dependencies == ["T2"]


@pytest.mark.asyncio
async def test_replan_result_can_build_task_graph():
    completed_task = Task(
        task_id="T1",
        name="查询订单",
        description="查询订单",
        status=TaskStatus.SUCCESS,
    )

    failed_task = Task(
        task_id="T2",
        name="查询库存",
        description="查询库存",
        status=TaskStatus.FAILED,
        dependencies=["T1"],
        error="库存服务失败",
    )

    remaining_task = Task(
        task_id="T3",
        name="更新订单",
        description="更新订单",
        status=TaskStatus.PENDING,
        dependencies=["T2"],
    )

    request = ReplanRequest(
        original_task="处理订单",
        failed_task=failed_task,
        completed_tasks=[completed_task],
        remaining_tasks=[remaining_task],
    )

    replanner = SimpleReplanner()

    result = await replanner.replan(request)

    graph = result.to_task_graph()

    assert len(graph.tasks()) == 3

    assert graph.get_task("T1").status == TaskStatus.SUCCESS
    assert graph.get_task("T2").status == TaskStatus.PENDING
    assert graph.get_task("T3").status == TaskStatus.PENDING

    ready_tasks = graph.get_ready_tasks()

    assert [task.task_id for task in ready_tasks] == ["T2"]