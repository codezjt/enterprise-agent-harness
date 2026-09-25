from __future__ import annotations

from collections import deque

from .task import Task, TaskStatus


class TaskGraph:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}

    def add_task(self, task: Task) -> None:
        if task.task_id in self._tasks:
            raise ValueError(
                f"Task already exists: {task.task_id}"
            )

        self._tasks[task.task_id] = task

    def get_task(self, task_id: str) -> Task:
        try:
            return self._tasks[task_id]
        except KeyError:
            raise KeyError(f"Task not found: {task_id}")

    def tasks(self) -> list[Task]:
        return list(self._tasks.values())

    def add_dependency(
        self,
        task_id: str,
        dependency_task_id: str,
    ) -> None:
        task = self.get_task(task_id)
        self.get_task(dependency_task_id)

        if task_id == dependency_task_id:
            raise ValueError(
                f"Task cannot depend on itself: {task_id}"
            )

        if dependency_task_id not in task.dependencies:
            task.dependencies.append(dependency_task_id)

        self._validate_acyclic()

    def get_ready_tasks(
        self,
        completed_task_ids: set[str] | None = None,
    ) -> list[Task]:
        completed = completed_task_ids or {
            task.task_id
            for task in self._tasks.values()
            if task.status == TaskStatus.SUCCESS
        }

        return [
            task
            for task in self._tasks.values()
            if task.is_ready(completed)
        ]

    def mark_success(
        self,
        task_id: str,
        output=None,
    ) -> None:
        task = self.get_task(task_id)

        task.status = TaskStatus.SUCCESS
        task.output = output
        task.error = None

    def mark_failed(
        self,
        task_id: str,
        error: str,
    ) -> None:
        task = self.get_task(task_id)

        task.status = TaskStatus.FAILED
        task.error = error

    def mark_running(self, task_id: str) -> None:
        task = self.get_task(task_id)
        task.status = TaskStatus.RUNNING

    def is_completed(self) -> bool:
        return all(
            task.status in {
                TaskStatus.SUCCESS,
                TaskStatus.CANCELLED,
            }
            for task in self._tasks.values()
        )

    def has_failed(self) -> bool:
        return any(
            task.status == TaskStatus.FAILED
            for task in self._tasks.values()
        )

    def _validate_acyclic(self) -> None:
        indegree = {
            task_id: 0
            for task_id in self._tasks
        }

        graph: dict[str, list[str]] = {
            task_id: []
            for task_id in self._tasks
        }

        for task in self._tasks.values():
            for dependency in task.dependencies:
                graph[dependency].append(task.task_id)
                indegree[task.task_id] += 1

        queue = deque(
            task_id
            for task_id, degree in indegree.items()
            if degree == 0
        )

        visited = 0

        while queue:
            task_id = queue.popleft()
            visited += 1

            for next_task in graph[task_id]:
                indegree[next_task] -= 1

                if indegree[next_task] == 0:
                    queue.append(next_task)

        if visited != len(self._tasks):
            raise ValueError(
                "TaskGraph contains a cycle"
            )