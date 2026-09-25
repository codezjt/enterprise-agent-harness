from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from .task import Task, TaskStatus
from .task_graph import TaskGraph


class ReplanRequest(BaseModel):
    """
    Replan 请求。

    描述一次任务执行失败后，Replanner 需要分析的上下文。
    """

    original_task: str = Field(
        description="用户原始任务"
    )

    failed_task: Task = Field(
        description="执行失败的任务"
    )

    completed_tasks: list[Task] = Field(
        default_factory=list,
        description="已经成功完成的任务",
    )

    remaining_tasks: list[Task] = Field(
        default_factory=list,
        description="尚未完成的任务",
    )


class ReplanResult(BaseModel):
    """
    Replan 结果。
    """

    reason: str = Field(
        default="",
        description="重新规划原因",
    )

    tasks: list[Task] = Field(
        default_factory=list,
        description="新的任务列表",
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


class Replanner(ABC):
    """
    任务失败后的重新规划器。

    Replanner 不负责执行任务，
    只负责生成新的执行计划。
    """

    @abstractmethod
    async def replan(
        self,
        request: ReplanRequest,
    ) -> ReplanResult:
        raise NotImplementedError


class SimpleReplanner(Replanner):
    """
    最小 Replanner 实现。

    当前策略：

    1. 保留已经成功完成的任务；
    2. 将失败任务重新置为 PENDING；
    3. 保留尚未执行的任务；
    4. 重新生成一个可继续执行的 TaskGraph。

    后续再替换成 LLM Replanner。
    """

    async def replan(
        self,
        request: ReplanRequest,
    ) -> ReplanResult:

        tasks: list[Task] = []

        # 1. 保留已经成功完成的任务
        for task in request.completed_tasks:
            completed_task = task.model_copy(
                deep=True
            )
            completed_task.status = TaskStatus.SUCCESS
            tasks.append(completed_task)

        # 2. 重新执行失败任务
        failed_task = request.failed_task.model_copy(
            deep=True
        )

        failed_task.status = TaskStatus.PENDING
        failed_task.retry_count += 1
        failed_task.error = None

        tasks.append(failed_task)

        # 3. 保留尚未执行的任务
        for task in request.remaining_tasks:
            remaining_task = task.model_copy(
                deep=True
            )

            if remaining_task.status != TaskStatus.SUCCESS:
                remaining_task.status = TaskStatus.PENDING

            tasks.append(remaining_task)

        return ReplanResult(
            reason=(
                f"重新规划失败任务: "
                f"{request.failed_task.task_id}"
            ),
            tasks=tasks,
        )