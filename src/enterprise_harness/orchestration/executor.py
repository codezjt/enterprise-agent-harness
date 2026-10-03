from __future__ import annotations

from typing import Any, TYPE_CHECKING

from enterprise_harness.runtime.context import RunContext
from enterprise_harness.runtime.contract import Runtime

from .task import Task


class AgentTaskExecutor:
    def __init__(
        self,
        runtime: Runtime,
        run_context: RunContext,
    ) -> None:
        self.runtime = runtime
        self.run_context = run_context

    async def execute(
        self,
        task: Task,
    ) -> Any:
        task_ctx = self._build_task_context(task)

        result = await self.runtime.run(task_ctx)

        if result.is_completed:
            return result.result
        elif result.is_failed:
            return {
                "error": result.error or "task execution failed",
                "status": "FAILED",
            }
        elif result.is_waiting_approval:
            from langgraph.types import interrupt

            return interrupt({
                "type": "approval_required",
                "approval_id": result.approval_id,
                "task_id": task.task_id,
                "task_name": task.name,
            })
        else:
            return {"status": "CANCELLED"}

    async def __call__(
        self,
        task: Task,
    ) -> Any:
        return await self.execute(task)

    def _build_task_context(self, task: Task) -> RunContext:
        context_data = dict(self.run_context.context)
        context_data["task_id"] = task.task_id
        context_data["task_name"] = task.name
        context_data["_run_id"] = self.run_context.run_id
        context_data["_agent_id"] = self.run_context.agent_id

        if self.run_context.principal is not None:
            context_data["_principal"] = self.run_context.principal

        if task.input:
            context_data["task_input"] = dict(task.input)

        dependency_results = task.input.get("dependency_results") if task.input else None
        if dependency_results:
            context_data["dependency_results"] = dependency_results

        return RunContext(
            run_id=self.run_context.run_id,
            agent_id=self.run_context.agent_id,
            agent_version=self.run_context.agent_version,
            tenant_id=self.run_context.tenant_id,
            task=task.description or task.name,
            trace_id=self.run_context.trace_id,
            context=context_data,
            principal=self.run_context.principal,
            trace_root_span_id=self.run_context.trace_root_span_id,
            current_span_id=self.run_context.current_span_id,
            metadata=dict(self.run_context.metadata),
            built_context=list(self.run_context.built_context),
        )