from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from enterprise_harness.agent.runtime import AgentRuntime
from enterprise_harness.observability import TraceManager
from enterprise_harness.policy.rbac import Principal

from .context_builder import RunContextBuilder
from .langgraph_runtime import LangGraphRuntime
from .models import Run, RunStatus


class RunManager:

    def __init__(
        self,
        langgraph_runtime: LangGraphRuntime | None = None,
        trace_manager: TraceManager | None = None,
    ):
        self.runs: dict[str, Run] = {}

        self.langgraph_runtime = (
            langgraph_runtime
        )

        self.trace_manager = (
            trace_manager
            or TraceManager()
        )

        self.context_builder = (
            RunContextBuilder(
                self.trace_manager
            )
        )

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
        principal: Principal | None = None,
    ) -> Run:

        run = self._get_run(run_id)

        run.status = RunStatus.RUNNING
        run.started_at = datetime.now(
            timezone.utc
        )

        run_context = self.context_builder.build(
            run,
            principal=principal,
        )

        try:

            result = await runtime.run_with_context(
                run_context
            )

            run.result = result
            run.status = RunStatus.COMPLETED

            self.trace_manager.finish_span(
                run_context.trace_root_span_id,
                output=result,
            )

        except Exception as exc:

            run.status = RunStatus.FAILED
            run.error = str(exc)

            if run_context.trace_root_span_id:

                self.trace_manager.fail_span(
                    run_context.trace_root_span_id,
                    exc,
                )

        finally:

            run.completed_at = (
                datetime.now(timezone.utc)
            )

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

        run.started_at = datetime.now(
            timezone.utc
        )

        try:

            result = (
                await self.langgraph_runtime.run(
                    run_id=run.run_id,
                    task=run.task,
                    context=run.context,
                )
            )

            if self.langgraph_runtime.is_interrupted(
                result
            ):

                run.status = (
                    RunStatus.WAITING_APPROVAL
                )

                return run

            run.result = (
                self.langgraph_runtime.extract_result(
                    result
                )
            )

            run.status = RunStatus.COMPLETED

            run.completed_at = (
                datetime.now(timezone.utc)
            )

        except Exception as exc:

            run.status = RunStatus.FAILED
            run.error = str(exc)

            run.completed_at = (
                datetime.now(timezone.utc)
            )

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
                f"Run cannot be resumed from status: "
                f"{run.status}"
            )

        run.status = RunStatus.RUNNING

        try:

            result = (
                await self.langgraph_runtime.resume(
                    run_id=run.run_id,
                    value=value,
                )
            )

            if self.langgraph_runtime.is_interrupted(
                result
            ):

                run.status = (
                    RunStatus.WAITING_APPROVAL
                )

                return run

            run.result = (
                self.langgraph_runtime.extract_result(
                    result
                )
            )

            run.status = RunStatus.COMPLETED

            run.completed_at = (
                datetime.now(timezone.utc)
            )

        except Exception as exc:

            run.status = RunStatus.FAILED
            run.error = str(exc)

            run.completed_at = (
                datetime.now(timezone.utc)
            )

        return run

    async def cancel_run(
        self,
        run_id: str,
    ) -> Run:

        run = self._get_run(run_id)

        run.status = RunStatus.CANCELLED

        run.completed_at = (
            datetime.now(timezone.utc)
        )

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