from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from enterprise_harness.observability import (
    MetricCollector,
    TraceManager,
)
from enterprise_harness.policy.rbac import Principal
from enterprise_harness.context import ContextBuilder, ContextProvider

from .context import RunContext
from .context_builder import RunContextBuilder
from .contract import Runtime
from .models import Run, RunStatus
from .result import RuntimeStatus
from enterprise_harness.repositories import (
    InMemoryRunRepository,
    RunRepository,
)


class RunManager:
    """
    企业级 Run 编排器。

    RunManager 不知道 DeepAgents、也不知道 LangGraph 的具体实现。
    它只知道 Runtime 契约：

        Runtime.run(context) → RuntimeResult
        Runtime.resume(context, value) → RuntimeResult
        Runtime.cancel(context) → RuntimeResult

    不再出现 hasattr(runtime, "build_agent")、isinstance 等类型判断。
    所有 Planner / TaskGraph / Scheduler / Recovery 逻辑都在 Runtime 内部。
    """

    def __init__(
        self,
        runtime: Runtime | None = None,
        trace_manager: TraceManager | None = None,
        metric_collector: MetricCollector | None = None,
        context_provider: ContextProvider | None = None,
        context_builder: ContextBuilder | None = None,
        run_repository: RunRepository | None = None,
    ):
        self.runs: dict[str, Run] = {}
        self._run_repository = run_repository or InMemoryRunRepository()
        self._cancel_events: dict[str, asyncio.Event] = {}

        self._runtime = runtime

        self.trace_manager = trace_manager or TraceManager()

        self._context_builder = context_builder or ContextBuilder()

        self.context_builder = RunContextBuilder(self.trace_manager)

        self.context_provider = context_provider

        self.metric_collector = metric_collector

    # ---- Core API ----

    async def create_run(
        self,
        agent_id: str,
        task: str,
        context: dict[str, Any] | None = None,
        agent_version: str = "1.0.0",
        tenant_id: str = "default",
    ) -> Run:
        run = Run(
            run_id=str(uuid4()),
            agent_id=agent_id,
            agent_version=agent_version,
            tenant_id=tenant_id,
            task=task,
            trace_id=str(uuid4()),
            context=context or {},
        )
        self.runs[run.run_id] = run
        self._cancel_events[run.run_id] = asyncio.Event()
        await self._run_repository.save(run)
        return run

    async def start_run(
        self,
        run_id: str,
        runtime: Runtime,
        principal: Principal | None = None,
    ) -> Run:
        run = self._get_run(run_id)

        if not RunStatus.is_valid_transition(run.status, RunStatus.RUNNING):
            raise ValueError(
                f"Cannot start run {run_id} in status {run.status.value}"
            )

        run.status = RunStatus.RUNNING
        run.started_at = datetime.now(timezone.utc)

        run_context = self.context_builder.build(
            run,
            principal=principal,
        )

        if self.context_provider is not None:
            await self.context_provider.enrich_run_context(run_context)
            built_ctx = self._context_builder.build_from_run_context(run_context)
            run_context.built_context = built_ctx

        try:
            runtime_result = await runtime.run(run_context)
            self._apply_runtime_result(run, runtime_result)

            if run_context.trace_root_span_id:
                if runtime_result.is_failed:
                    self.trace_manager.fail_span(
                        run_context.trace_root_span_id,
                        RuntimeError(runtime_result.error or "Runtime failed"),
                    )
                elif runtime_result.is_completed:
                    self.trace_manager.finish_span(
                        run_context.trace_root_span_id,
                        output=runtime_result.result,
                    )

        except Exception as exc:
            run.status = RunStatus.FAILED
            run.error = str(exc)

            if run_context.trace_root_span_id:
                self.trace_manager.fail_span(
                    run_context.trace_root_span_id,
                    exc,
                )

        if run.status != RunStatus.WAITING_APPROVAL:
            run.completed_at = datetime.now(timezone.utc)

        if self.metric_collector is not None:
            self.metric_collector.record_agent_run(
                success=run.status == RunStatus.COMPLETED
            )

        self._cancel_events.pop(run_id, None)
        await self._run_repository.save(run)
        return run

    async def resume_run(
        self,
        run_id: str,
        value: Any,
    ) -> Run:
        run = self._get_run(run_id)

        if run.status != RunStatus.WAITING_APPROVAL:
            raise ValueError(
                f"Run cannot be resumed from status: {run.status}"
            )

        runtime = self._runtime
        if runtime is None:
            raise RuntimeError(
                "No Runtime configured for resume"
            )

        if not runtime.supports_resume:
            raise RuntimeError(
                f"Runtime {runtime.__class__.__name__} does not support resume"
            )

        run.status = RunStatus.RUNNING
        run_context = self.context_builder.build(run)

        try:
            runtime_result = await runtime.resume(run_context, value)
            self._apply_runtime_result(run, runtime_result)
        except Exception as exc:
            run.status = RunStatus.FAILED
            run.error = str(exc)

        if run.status != RunStatus.WAITING_APPROVAL:
            run.completed_at = datetime.now(timezone.utc)

        await self._run_repository.save(run)
        return run

    async def cancel_run(
        self,
        run_id: str,
    ) -> Run:
        run = self._get_run(run_id)

        if not RunStatus.is_valid_transition(run.status, RunStatus.CANCELLED):
            raise ValueError(
                f"Cannot cancel run {run_id} in status {run.status.value}"
            )

        runtime = self._runtime
        if runtime is not None and runtime.supports_cancel:
            run_context = self.context_builder.build(run)
            await runtime.cancel(run_context)

        run.status = RunStatus.CANCELLED
        run.completed_at = datetime.now(timezone.utc)

        cancel_event = self._cancel_events.get(run_id)
        if cancel_event is not None:
            cancel_event.set()

        await self._run_repository.save(run)
        return run

    # ---- Query API ----

    async def get_run(self, run_id: str) -> Run:
        return self._get_run(run_id)

    async def list_runs(
        self,
        tenant_id: str | None = None,
    ) -> list[Run]:
        return await self._run_repository.list(tenant_id=tenant_id)

    def is_cancelled(self, run_id: str) -> bool:
        event = self._cancel_events.get(run_id)
        if event is None:
            return False
        return event.is_set()

    # ---- Internal ----

    def _get_run(self, run_id: str) -> Run:
        run = self.runs.get(run_id)
        if run is None:
            raise KeyError(f"Run not found: {run_id}")
        return run

    def _apply_runtime_result(
        self,
        run: Run,
        result: Any,
    ) -> None:
        if result.status == RuntimeStatus.WAITING_APPROVAL:
            run.status = RunStatus.WAITING_APPROVAL
            run.approval_id = result.approval_id
            run.checkpoint_id = result.checkpoint_id
            return

        if result.status == RuntimeStatus.COMPLETED:
            run.result = result.result
            run.status = RunStatus.COMPLETED
            run.approval_id = None
            run.checkpoint_id = result.checkpoint_id
            return

        if result.status == RuntimeStatus.FAILED:
            run.status = RunStatus.FAILED
            run.error = result.error
            return

        if result.status == RuntimeStatus.CANCELLED:
            run.status = RunStatus.CANCELLED
            return

        raise ValueError(
            f"Unsupported runtime result status: {result.status}"
        )