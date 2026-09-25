from __future__ import annotations

from typing import Any

from enterprise_harness.agent.runtime import AgentRuntime
from enterprise_harness.runtime.context import RunContext

from .task import Task


class AgentTaskExecutor:
    """
    将 Orchestration 层的 Task 转换成 AgentRuntime 的执行请求。

    Scheduler 不直接依赖 AgentRuntime，
    而是通过这个 Executor 完成两层之间的适配。
    """

    def __init__(
        self,
        runtime: AgentRuntime,
        run_context: RunContext,
    ) -> None:
        self.runtime = runtime
        self.run_context = run_context

    async def execute(self, task: Task) -> Any:
        task_context = dict(self.run_context.context)

        task_context["task_id"] = task.task_id
        task_context["task_name"] = task.name

        if task.input:
            task_context["task_input"] = task.input

        return await self.runtime.run(
            task=task.description or task.name,
            context=task_context,
        )

    async def __call__(self, task: Task) -> Any:
        return await self.execute(task)