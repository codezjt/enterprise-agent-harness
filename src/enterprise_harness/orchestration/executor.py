from __future__ import annotations

from typing import Any

from enterprise_harness.agent.runtime import AgentRuntime
from enterprise_harness.runtime.context import RunContext

from .task import Task


class AgentTaskExecutor:
    """
    将 Orchestration Task 适配为 AgentRuntime 执行请求。
    """

    def __init__(
        self,
        runtime: AgentRuntime,
        run_context: RunContext,
    ) -> None:
        self.runtime = runtime
        self.run_context = run_context

    async def execute(
        self,
        task: Task,
    ) -> Any:
        task_context = dict(
            self.run_context.context
        )

        task_context["task_id"] = task.task_id
        task_context["task_name"] = task.name
        task_context["_run_id"] = self.run_context.run_id
        task_context["_agent_id"] = self.run_context.agent_id

        if self.run_context.principal is not None:
            task_context["_principal"] = self.run_context.principal

        if task.input:
            task_context["task_input"] = dict(
                task.input
            )

        dependency_results = task.input.get(
            "dependency_results"
        )

        if dependency_results:
            task_context[
                "dependency_results"
            ] = dependency_results

        return await self.runtime.run(
            task=(
                task.description
                or task.name
            ),
            context=task_context,
        )

    async def __call__(
        self,
        task: Task,
    ) -> Any:
        return await self.execute(task)