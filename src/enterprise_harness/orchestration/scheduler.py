from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from .task import Task, TaskStatus
from .task_graph import TaskGraph


TaskExecutor = Callable[[Task], Awaitable[Any]]


class Scheduler:
    """
    TaskGraph 调度器。

    负责：
    1. 找到 READY Task
    2. 并行执行无依赖任务
    3. 将依赖任务结果注入下游 Task
    4. 控制最大并发数
    5. 维护 Task 状态
    """

    def __init__(
        self,
        graph: TaskGraph,
        executor: TaskExecutor,
        max_concurrency: int = 4,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError(
                "max_concurrency must be greater than 0"
            )

        self.graph = graph
        self.executor = executor
        self.max_concurrency = max_concurrency

    async def run(self) -> TaskGraph:
        while not self.graph.is_completed():
            if self.graph.has_failed():
                break

            ready_tasks = self.graph.get_ready_tasks()

            if not ready_tasks:
                pending_tasks = [
                    task
                    for task in self.graph.tasks()
                    if task.status
                    not in {
                        TaskStatus.SUCCESS,
                        TaskStatus.CANCELLED,
                        TaskStatus.FAILED,
                    }
                ]

                if pending_tasks:
                    raise RuntimeError(
                        "TaskGraph is blocked: "
                        "no ready tasks but unfinished tasks remain"
                    )

                break

            await self._execute_batch(ready_tasks)

        return self.graph

    async def _execute_batch(
        self,
        tasks: list[Task],
    ) -> None:
        semaphore = asyncio.Semaphore(
            self.max_concurrency
        )

        async def execute_task(task: Task) -> None:
            async with semaphore:
                await self._execute_task(task)

        await asyncio.gather(
            *(execute_task(task) for task in tasks)
        )

    async def _execute_task(
        self,
        task: Task,
    ) -> None:
        self._inject_dependency_results(task)

        self.graph.mark_running(task.task_id)

        try:
            result = await self.executor(task)

            self.graph.mark_success(
                task.task_id,
                output=result,
            )

        except Exception as exc:
            self.graph.mark_failed(
                task.task_id,
                error=str(exc),
            )

    def _inject_dependency_results(
        self,
        task: Task,
    ) -> None:
        """
        将已经成功完成的依赖任务结果注入当前 Task。

        例如：

        T1 -> order
        T2 -> inventory

        T3.dependencies = ["T1", "T2"]

        则：

        T3.input["dependency_results"] = {
            "T1": order,
            "T2": inventory,
        }
        """

        if not task.dependencies:
            return

        dependency_results: dict[str, Any] = {}

        for dependency_id in task.dependencies:
            dependency = self.graph.get_task(
                dependency_id
            )

            if dependency.status != TaskStatus.SUCCESS:
                raise RuntimeError(
                    f"Dependency task is not successful: "
                    f"{dependency_id}"
                )

            dependency_results[
                dependency_id
            ] = dependency.output

        task.input = {
            **task.input,
            "dependency_results": dependency_results,
        }