from __future__ import annotations

from enterprise_harness.orchestration.replanner import (
    ReplanRequest,
    Replanner,
)
from enterprise_harness.orchestration.scheduler import Scheduler
from enterprise_harness.orchestration.task import TaskStatus
from enterprise_harness.orchestration.task_graph import TaskGraph


class RecoveryManager:
    """
    任务执行失败后的 Runtime Recovery 协调器。

    RecoveryManager 不负责：
    - 具体任务执行
    - 任务规划

    它只负责协调：

        TaskGraph
            ↓
        FAILED
            ↓
        Replanner
            ↓
        新 TaskGraph
            ↓
        Scheduler
    """

    def __init__(
        self,
        replanner: Replanner,
        max_replans: int = 1,
    ) -> None:
        if max_replans < 0:
            raise ValueError("max_replans must be greater than or equal to 0")

        self.replanner = replanner
        self.max_replans = max_replans

    async def recover(
        self,
        *,
        original_task: str,
        graph: TaskGraph,
        scheduler_factory,
    ) -> TaskGraph:
        """
        对失败的 TaskGraph 执行 Recovery。

        scheduler_factory:
            根据新的 TaskGraph 创建 Scheduler。
        """

        replan_count = 0
        current_graph = graph

        while current_graph.has_failed():
            if replan_count >= self.max_replans:
                return current_graph

            failed_tasks = [
                task
                for task in current_graph.tasks()
                if task.status == TaskStatus.FAILED
            ]

            if not failed_tasks:
                return current_graph

            failed_task = failed_tasks[0]

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

            replan_result = await self.replanner.replan(request)

            current_graph = replan_result.to_task_graph()

            scheduler = scheduler_factory(current_graph)

            await scheduler.run()

            replan_count += 1

        return current_graph