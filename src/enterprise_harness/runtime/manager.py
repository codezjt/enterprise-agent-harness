from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from enterprise_harness.agent import AgentRuntime

from .langgraph_runtime import LangGraphRuntime
from .models import Run, RunStatus


class RunManager:
    """负责 Agent Run 的生命周期管理。"""

    def __init__(
        self,
        langgraph_runtime: LangGraphRuntime | None = None,
    ):
        self.runs: dict[str, Run] = {}
        self.langgraph_runtime = langgraph_runtime

    async def create_run(
        self,
        agent_id: str,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Run:
        run = Run(
            run_id=str(uuid4()),
            agent_id=agent_id,
            task=task,
            context=context or {},
        )

        self.runs[run.run_id] = run

        return run

    async def start_run(
        self,
        run_id: str,
        runtime: AgentRuntime,
    ) -> Run:
        run = self._get_run(run_id)

        run.status = RunStatus.RUNNING
        run.started_at = datetime.now(timezone.utc)

        try:
            result = await runtime.run(
                task=run.task,
                context=run.context,
            )

            run.result = result
            run.status = RunStatus.COMPLETED

        except Exception as exc:
            run.status = RunStatus.FAILED
            run.error = str(exc)

        finally:
            run.completed_at = datetime.now(timezone.utc)

        return run

    async def start_langgraph_run(
        self,
        run_id: str,
    ) -> Run:
        if self.langgraph_runtime is None:
            raise RuntimeError(
                "LangGraph runtime is not configured"
            )

        run = self._get_run(run_id)

        run.status = RunStatus.RUNNING
        run.started_at = datetime.now(timezone.utc)

        try:
            result = await self.langgraph_runtime.run(
                run_id=run.run_id,
                task=run.task,
                context=run.context,
            )

            if self.langgraph_runtime.is_interrupted(result):
                run.status = RunStatus.WAITING_APPROVAL
                return run

            run.result = self.langgraph_runtime.extract_result(result)
            run.status = RunStatus.COMPLETED
            run.completed_at = datetime.now(timezone.utc)

        except Exception as exc:
            run.status = RunStatus.FAILED
            run.error = str(exc)
            run.completed_at = datetime.now(timezone.utc)

        return run

    async def resume_run(
        self,
        run_id: str,
        value: Any,
    ) -> Run:
        if self.langgraph_runtime is None:
            raise RuntimeError(
                "LangGraph runtime is not configured"
            )

        run = self._get_run(run_id)

        if run.status != RunStatus.WAITING_APPROVAL:
            raise ValueError(
                f"Run cannot be resumed from status: {run.status}"
            )

        run.status = RunStatus.RUNNING

        try:
            result = await self.langgraph_runtime.resume(
                run_id=run.run_id,
                value=value,
            )

            if self.langgraph_runtime.is_interrupted(result):
                run.status = RunStatus.WAITING_APPROVAL
                return run

            run.result = self.langgraph_runtime.extract_result(result)
            run.status = RunStatus.COMPLETED
            run.completed_at = datetime.now(timezone.utc)

        except Exception as exc:
            run.status = RunStatus.FAILED
            run.error = str(exc)
            run.completed_at = datetime.now(timezone.utc)

        return run

    async def cancel_run(
        self,
        run_id: str,
    ) -> Run:
        run = self._get_run(run_id)

        run.status = RunStatus.CANCELLED
        run.completed_at = datetime.now(timezone.utc)

        return run

    async def get_run(
        self,
        run_id: str,
    ) -> Run:
        return self._get_run(run_id)

    def _get_run(
        self,
        run_id: str,
    ) -> Run:
        run = self.runs.get(run_id)

        if run is None:
            raise KeyError(
                f"Run not found: {run_id}"
            )

        return run