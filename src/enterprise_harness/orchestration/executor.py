from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from enterprise_harness.runtime.langgraph_runtime import LangGraphRuntime

from enterprise_harness.runtime.context import RunContext

from .task import Task


class AgentTaskExecutor:
    def __init__(
        self,
        langgraph_runtime: "LangGraphRuntime",
        run_context: RunContext,
    ) -> None:
        self.langgraph_runtime = langgraph_runtime
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

        input_data: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        task.description
                        or task.name
                    ),
                }
            ],
            "context": task_context,
        }

        if self.run_context.built_context:
            input_data["context"] = {
                "items": [
                    item.model_dump()
                    for item in self.run_context.built_context
                ],
                **task_context,
            }

        result = await self.langgraph_runtime.run(
            run_id=self.run_context.run_id,
            input_data=input_data,
        )

        return result

    async def __call__(
        self,
        task: Task,
    ) -> Any:
        return await self.execute(task)