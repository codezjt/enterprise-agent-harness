import pytest

from enterprise_harness.orchestration.replanner import (
    SimpleReplanner,
)
from enterprise_harness.orchestration.scheduler import (
    Scheduler,
)
from enterprise_harness.orchestration.task import Task
from enterprise_harness.orchestration.task_graph import (
    TaskGraph,
)
from enterprise_harness.runtime.recovery import (
    RecoveryManager,
)
from enterprise_harness.runtime.retry import (
    RetryPolicy,
)


@pytest.mark.asyncio
async def test_recovery_retries_transient_failure():
    graph = TaskGraph()

    graph.add_task(
        Task(
            task_id="T1",
            name="query_inventory",
            max_retries=1,
        )
    )

    attempts = 0

    async def executor(task):
        nonlocal attempts

        attempts += 1

        if attempts == 1:
            raise RuntimeError(
                "connection timeout"
            )

        return {
            "inventory": 80
        }

    async def run_graph(new_graph):
        scheduler = Scheduler(
            graph=new_graph,
            executor=executor,
        )
        return await scheduler.run()

    scheduler = Scheduler(
        graph=graph,
        executor=executor,
    )

    await scheduler.run()

    assert graph.has_failed()

    recovery = RecoveryManager(
        replanner=SimpleReplanner(),
        retry_policy=RetryPolicy(),
        max_replans=0,
    )

    result = await recovery.recover(
        original_task="查询库存",
        graph=graph,
        scheduler_factory=lambda g: Scheduler(
            graph=g,
            executor=executor,
        ),
    )

    assert not result.has_failed()

    assert attempts == 2

    assert (
        result.get_task("T1").output
        == {"inventory": 80}
    )


@pytest.mark.asyncio
async def test_recovery_replans_non_retryable_failure():
    graph = TaskGraph()

    graph.add_task(
        Task(
            task_id="T1",
            name="update_order",
        )
    )

    attempts = 0

    async def executor(task):
        nonlocal attempts

        attempts += 1

        raise RuntimeError(
            "permission denied"
        )

    scheduler = Scheduler(
        graph=graph,
        executor=executor,
    )

    await scheduler.run()

    assert graph.has_failed()

    recovery = RecoveryManager(
        replanner=SimpleReplanner(),
        retry_policy=RetryPolicy(),
        max_replans=1,
    )

    result = await recovery.recover(
        original_task="修改订单",
        graph=graph,
        scheduler_factory=lambda g: Scheduler(
            graph=g,
            executor=executor,
        ),
    )

    assert result.has_failed()

    assert attempts == 2