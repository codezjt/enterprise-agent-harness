from __future__ import annotations

from .planner import Plan
from .task_graph import TaskGraph


class PlanValidationError(ValueError):
    """Plan 校验失败。"""


class PlanValidator:
    """
    Planner 输出结果的结构校验器。

    Validator 只负责校验，不负责：
    - Planner
    - Task 执行
    - Scheduler
    - AgentRuntime
    """

    def validate(self, plan: Plan) -> TaskGraph:
        if not plan.task.strip():
            raise PlanValidationError(
                "Plan task must not be empty"
            )

        if not plan.tasks:
            raise PlanValidationError(
                "Plan must contain at least one task"
            )

        self._validate_task_ids(plan)
        self._validate_dependencies(plan)

        try:
            return plan.to_task_graph()
        except (ValueError, RuntimeError) as exc:
            raise PlanValidationError(str(exc)) from exc

    @staticmethod
    def _validate_task_ids(plan: Plan) -> None:
        task_ids: set[str] = set()

        for task in plan.tasks:
            if not task.task_id.strip():
                raise PlanValidationError(
                    "Task id must not be empty"
                )

            if task.task_id in task_ids:
                raise PlanValidationError(
                    f"Duplicate task id: {task.task_id}"
                )

            task_ids.add(task.task_id)

    @staticmethod
    def _validate_dependencies(plan: Plan) -> None:
        task_ids = {
            task.task_id
            for task in plan.tasks
        }

        for task in plan.tasks:
            for dependency in task.dependencies:
                if dependency not in task_ids:
                    raise PlanValidationError(
                        f"Task {task.task_id} "
                        f"depends on unknown task: {dependency}"
                    )