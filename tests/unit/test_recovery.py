import pytest

from enterprise_harness.orchestration.replanner import (
    ReplanRequest,
    ReplanResult,
    Replanner,
)
from enterprise_harness.orchestration.scheduler import Scheduler
from enterprise_harness.orchestration.task import (
    Task,
    TaskStatus,
)
from enterprise_harness.orchestration.task_graph import TaskGraph
from enterprise_harness.runtime.recovery import RecoveryManager


class RetryOnceReplanner(Replanner):
    """
    测试用 Replanner。

    第一次看到失败任务时：
    FAILED -> PENDING

    第二次不应该再进入，因为测试 Executor
    会让任务成功。
    """

    async def replan(
        self,
        request: ReplanRequest,
    ) -> ReplanResult:
        failed_task = request.failed_task.model_copy(
            deep=True
        )

        failed_task.status = TaskStatus.PENDING
        failed_task.error = None
        failed_task.retry_count += 1

        tasks = [
            task.model_copy(deep=True)
            for task in request.completed_tasks
        ]

        tasks.append(failed_task)

        tasks.extend(
            task.model_copy(deep=True)
            for task in request.remaining_tasks
        )

        return ReplanResult(
            reason="retry failed task",
            tasks=tasks,
        )


@pytest.mark.asyncio
async def test_recovery_replans_failed_task_and_continues():
    graph = TaskGraph()

    graph.add_task(
        Task(
            task_id="T1",
            name="第一步",
            description="执行第一步",
        )
    )

    graph.add_task(
        Task(
            task_id="T2",
            name="第二步",
            description="执行第二步",
            dependencies=["T1"],
        )
    )

    execution_count: dict[str, int] = {}

    async def execute(task: Task):
        execution_count[task.task_id] = (
            execution_count.get(task.task_id, 0) + 1
        )

        if task.task_id == "T2":
            if execution_count[task.task_id] == 1:
                raise RuntimeError("temporary failure")

        return f"success:{task.task_id}"

    scheduler = Scheduler(
        graph=graph,
        executor=execute,
    )

    await scheduler.run()

    assert graph.get_task("T1").status == TaskStatus.SUCCESS
    assert graph.get_task("T2").status == TaskStatus.FAILED

    recovery = RecoveryManager(
        replanner=RetryOnceReplanner(),
        max_replans=1,
    )

    def scheduler_factory(new_graph: TaskGraph):
        return Scheduler(
            graph=new_graph,
            executor=execute,
        )

    recovered_graph = await recovery.recover(
        original_task="执行两个步骤",
        graph=graph,
        scheduler_factory=scheduler_factory,
    )

    assert recovered_graph.get_task("T1").status == TaskStatus.SUCCESS
    assert recovered_graph.get_task("T2").status == TaskStatus.SUCCESS

    assert execution_count["T1"] == 1
    assert execution_count["T2"] == 2

@pytest.mark.asyncio
async def test_recovery_stops_after_max_replans():
    graph = TaskGraph()

    graph.add_task(
        Task(
            task_id="T1",
            name="失败任务",
            description="始终失败",
        )
    )

    async def execute(task: Task):
        raise RuntimeError("permanent failure")

    scheduler = Scheduler(
        graph=graph,
        executor=execute,
    )

    await scheduler.run()

    recovery = RecoveryManager(
        replanner=RetryOnceReplanner(),
        max_replans=1,
    )

    def scheduler_factory(new_graph: TaskGraph):
        return Scheduler(
            graph=new_graph,
            executor=execute,
        )

    recovered_graph = await recovery.recover(
        original_task="执行失败任务",
        graph=graph,
        scheduler_factory=scheduler_factory,
    )

    assert recovered_graph.get_task("T1").status == TaskStatus.FAILED
    assert recovered_graph.get_task("T1").retry_count == 1