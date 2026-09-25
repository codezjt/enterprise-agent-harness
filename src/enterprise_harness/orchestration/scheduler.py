from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from .task import Task, TaskStatus
from .task_graph import TaskGraph


TaskExecutor = Callable[[Task], Awaitable[Any]]


class Scheduler:
    def __init__(
        self,
        graph: TaskGraph,
        executor: TaskExecutor,
        max_concurrency: int = 4,
    ) -> None:
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than 0")

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