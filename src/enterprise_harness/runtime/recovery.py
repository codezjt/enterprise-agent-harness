from __future__ import annotations

from enterprise_harness.orchestration.replanner import (
    ReplanRequest,
    Replanner,
)
from enterprise_harness.orchestration.scheduler import Scheduler
from enterprise_harness.orchestration.task import TaskStatus
from enterprise_harness.orchestration.task_graph import TaskGraph

from .retry import RetryPolicy


class RecoveryManager:
    """
    Runtime Recovery 协调器。

    负责：

        Task FAILED
             ↓
        RetryPolicy
        /        \
      Retry      Replan
        │          │
        └────┬─────┘
             ▼
          Scheduler
    """

    def __init__(
        self,
        replanner: Replanner,
        retry_policy: RetryPolicy | None = None,
        max_replans: int = 1,
    ) -> None:
        if max_replans < 0:
            raise ValueError(
                "max_replans must be greater than or equal to 0"
            )

        self.replanner = replanner
        self.retry_policy = retry_policy or RetryPolicy()
        self.max_replans = max_replans

    async def recover(
        self,
        *,
        original_task: str,
        graph: TaskGraph,
        scheduler_factory,
    ) -> TaskGraph:

        replan_count = 0
        current_graph = graph

        while current_graph.has_failed():
            failed_tasks = [
                task
                for task in current_graph.tasks()
                if task.status == TaskStatus.FAILED
            ]

            if not failed_tasks:
                return current_graph

            failed_task = failed_tasks[0]

            # 1. 首先判断是否可以 Retry
            retry_decision = self.retry_policy.should_retry(
                failed_task
            )

            if retry_decision.retry:
                retry_task = failed_task.model_copy(
                    deep=True
                )

                retry_task.status = TaskStatus.PENDING
                retry_task.retry_count += 1
                retry_task.error = None

                retry_task.input.pop(
                    "dependency_results",
                    None,
                )

                retry_tasks = [
                    task.model_copy(deep=True)
                    for task in current_graph.tasks()
                    if task.task_id != failed_task.task_id
                ]

                retry_tasks.append(retry_task)

                retry_graph = TaskGraph()

                for task in retry_tasks:
                    retry_graph.add_task(task)

                for task in retry_tasks:
                    for dependency in task.dependencies:
                        retry_graph.add_dependency(
                            task.task_id,
                            dependency,
                        )

                current_graph = retry_graph

                scheduler = scheduler_factory(
                    current_graph
                )

                await scheduler.run()

                continue

            # 2. Retry 不允许，进入 Replan
            if replan_count >= self.max_replans:
                return current_graph

            completed_tasks = [
                task
                for task in current_graph.tasks()
                if task.status == TaskStatus.SUCCESS
            ]

            remaining_tasks = [
                task
                for task in current_graph.tasks()
                if task.status not in {
                    TaskStatus.SUCCESS,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                }
            ]

            request = ReplanRequest(
                original_task=original_task,
                failed_task=failed_task,
                completed_tasks=completed_tasks,
                remaining_tasks=remaining_tasks,
            )

            replan_result = await self.replanner.replan(
                request
            )

            current_graph = replan_result.to_task_graph()

            scheduler = scheduler_factory(
                current_graph
            )

            await scheduler.run()

            replan_count += 1

        return current_graph