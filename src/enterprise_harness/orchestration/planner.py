from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from .task import Task
from .task_graph import TaskGraph


class Plan(BaseModel):
    """
    Planner 输出的结构化任务计划。

    Plan 只描述任务和依赖关系，不负责执行。
    """

    task: str = Field(
        description="用户原始任务"
    )

    tasks: list[Task] = Field(
        default_factory=list,
        description="需要执行的任务列表"
    )

    def to_task_graph(self) -> TaskGraph:
        graph = TaskGraph()

        for task in self.tasks:
            graph.add_task(task)

        for task in self.tasks:
            for dependency in task.dependencies:
                graph.add_dependency(
                    task.task_id,
                    dependency,
                )

        return graph


class Planner(ABC):
    """
    任务规划抽象。

    Natural Language Task -> Plan
    """

    @abstractmethod
    async def plan(self, task: str) -> Plan:
        raise NotImplementedError