import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from enterprise_harness.agent.runtime import AgentRuntime
from enterprise_harness.observability import (
    MetricCollector,
    TraceManager,
)
from enterprise_harness.policy.rbac import Principal

from enterprise_harness.orchestration.executor import AgentTaskExecutor
from enterprise_harness.orchestration.plan_validator import PlanValidator
from enterprise_harness.orchestration.planner import Planner
from enterprise_harness.orchestration.replanner import (
    Replanner,
    SimpleReplanner,
)
from enterprise_harness.orchestration.scheduler import Scheduler
from enterprise_harness.orchestration.task import TaskStatus
from enterprise_harness.orchestration.task_graph import TaskGraph
from enterprise_harness.runtime.recovery import RecoveryManager
from enterprise_harness.context import ContextBuilder, ContextProvider

from .context import RunContext
from .context_builder import RunContextBuilder
from .langgraph_runtime import LangGraphRuntime
from .models import Run, RunStatus
from .agent_runtime_adapter import AgentRuntimeAdapter
from enterprise_harness.runtime.langgraph_runtime_adapter import (
    LangGraphRuntimeAdapter,
)
from enterprise_harness.runtime.result import (
    RuntimeResult,
    RuntimeStatus,
)
from enterprise_harness.repositories import (
    InMemoryRunRepository,
    RunRepository,
)

class RunManager:

    def __init__(
        self,
        langgraph_runtime: LangGraphRuntime | None = None,
        trace_manager: TraceManager | None = None,
        replanner: Replanner | None = None,
        max_replans: int = 1,
        metric_collector: MetricCollector | None = None,
        context_provider: ContextProvider | None = None,
        context_builder: ContextBuilder | None = None,
        run_repository: RunRepository | None = None,
    ):
        self.runs: dict[str, Run] = {}
        self._run_repository = run_repository or InMemoryRunRepository()
        self._cancel_events: dict[str, asyncio.Event] = {}

        self._langgraph_runtime = langgraph_runtime
        self._runtime_store: dict[str, LangGraphRuntime] = {}

        self.trace_manager = (
            trace_manager
            or TraceManager()
        )

        self._context_builder = context_builder or ContextBuilder()

        self.context_builder = (
            RunContextBuilder(
                self.trace_manager
            )
        )

        self.context_provider = context_provider

        self.plan_validator = PlanValidator()

        self.recovery_manager = RecoveryManager(
            replanner=replanner or SimpleReplanner(),
            max_replans=max_replans,
            metric_collector=metric_collector,
        )

        self.metric_collector = metric_collector

    def _get_or_create_langgraph_runtime(
        self,
        run_id: str,
        deepagent_runtime=None,
        run_context: "RunContext | None" = None,
    ) -> LangGraphRuntime:
        existing = self._runtime_store.get(run_id)
        if existing is not None:
            return existing

        if self._langgraph_runtime is not None:
            self._runtime_store[run_id] = self._langgraph_runtime
            return self._langgraph_runtime

        if deepagent_runtime is not None:
            graph = deepagent_runtime.build_agent(run_context)
            lgr = LangGraphRuntime(compiled_graph=graph)
            self._runtime_store[run_id] = lgr
            return lgr

        raise RuntimeError(
            "No LangGraphRuntime available. Provide deepagent_runtime or pre-configure langgraph_runtime."
        )

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
            context=context or {},
        )

        self.runs[run.run_id] = run
        self._cancel_events[run.run_id] = asyncio.Event()
        await self._run_repository.save(run)
        return run

    async def start_run(
            self,
            run_id: str,
            runtime: AgentRuntime,
            principal: Principal | None = None,
    ) -> Run:

        run = self._get_run(run_id)

        if not RunStatus.is_valid_transition(run.status, RunStatus.RUNNING):
            raise ValueError(
                f"Cannot start run {run_id} in status {run.status.value}"
            )

        cancel_event = asyncio.Event()
        self._cancel_events[run_id] = cancel_event

        run.status = RunStatus.RUNNING
        run.started_at = datetime.now(
            timezone.utc
        )

        run_context = self.context_builder.build(
            run,
            principal=principal,
        )

        if self.context_provider is not None:
            await self.context_provider.enrich_run_context(run_context)
            built_ctx = self._context_builder.build_from_run_context(run_context)
            run_context.built_context = built_ctx

        has_build_agent = (
            runtime is not None
            and hasattr(runtime, "build_agent")
            and callable(getattr(runtime, "build_agent", None))
        )

        if has_build_agent:

            langgraph_runtime = self._get_or_create_langgraph_runtime(
                run_id, runtime, run_context=run_context
            )
            adapter = LangGraphRuntimeAdapter(langgraph_runtime)
        elif self._langgraph_runtime is not None:

            self._runtime_store[run_id] = self._langgraph_runtime
            adapter = LangGraphRuntimeAdapter(self._langgraph_runtime)
        elif runtime is not None:

            adapter = AgentRuntimeAdapter(runtime)
        else:

            raise RuntimeError(
                "No runtime available. Provide runtime, deepagent_runtime, "
                "or pre-configure langgraph_runtime."
            )

        try:

            runtime_result = await adapter.run(
                run_context
            )

            self._apply_runtime_result(run, runtime_result)

            if run_context.trace_root_span_id:
                if runtime_result.is_failed:
                    self.trace_manager.fail_span(
                        run_context.trace_root_span_id,
                        RuntimeError(
                            runtime_result.error
                            or "Agent runtime failed"
                        ),
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
            run.completed_at = (
                datetime.now(timezone.utc)
            )

        if self.metric_collector is not None:
            self.metric_collector.record_agent_run(
                success=run.status == RunStatus.COMPLETED
            )

        self._cancel_events.pop(run_id, None)

        await self._run_repository.save(run)
        return run

    async def start_langgraph_run(
            self,
            run_id: str,
            deepagent_runtime=None,
            principal: Principal | None = None,
    ) -> Run:

        if deepagent_runtime is None and self._langgraph_runtime is None:
            raise ValueError(
                "deepagent_runtime is required for start_langgraph_run"
            )

        return await self.start_run(
            run_id=run_id,
            runtime=deepagent_runtime,
            principal=principal,
        )

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

        run.status = RunStatus.RUNNING

        run_context = self.context_builder.build(run)

        langgraph_runtime = self._runtime_store.get(run_id)
        if langgraph_runtime is None:
            raise RuntimeError(
                "Cannot resume: no LangGraphRuntime found for this run. "
                "Did you pass a deepagent_runtime on start?"
            )

        adapter = LangGraphRuntimeAdapter(langgraph_runtime)

        try:
            runtime_result = await adapter.resume(run_context, value)
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

        run.status = RunStatus.CANCELLED

        run.completed_at = (
            datetime.now(timezone.utc)
        )

        cancel_event = self._cancel_events.get(run_id)
        if cancel_event is not None:
            cancel_event.set()

        await self._run_repository.save(run)
        return run

    def is_cancelled(self, run_id: str) -> bool:
        event = self._cancel_events.get(run_id)
        if event is None:
            return False
        return event.is_set()

    async def list_runs(
        self,
        tenant_id: str | None = None,
    ) -> list[Run]:
        runs = await self._run_repository.list(tenant_id=tenant_id)
        return runs

    async def get_run(
        self,
        run_id: str,
    ) -> Run:

        return self._get_run(run_id)

    def get_runs(
        self,
        *,
        tenant_id: str | None = None,
    ) -> list[Run]:
        runs = list(self.runs.values())
        if tenant_id is not None:
            runs = [
                r for r in runs
                if r.tenant_id == tenant_id
            ]
        return runs

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

    async def start_task_graph_run(
        self,
        run_id: str,
        runtime: AgentRuntime,
        task_graph: TaskGraph,
        principal=None,
    ) -> Run:

        run = self._get_run(run_id)

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

        has_build_agent = (
            runtime is not None
            and hasattr(runtime, "build_agent")
            and callable(getattr(runtime, "build_agent", None))
        )

        if has_build_agent:
            langgraph_runtime = self._get_or_create_langgraph_runtime(
                run_id, runtime, run_context=run_context
            )
        elif self._langgraph_runtime is not None:
            langgraph_runtime = self._langgraph_runtime
        else:
            raise RuntimeError(
                "No LangGraphRuntime available. Provide deepagent_runtime or pre-configure langgraph_runtime."
            )

        executor = AgentTaskExecutor(
            langgraph_runtime=langgraph_runtime,
            run_context=run_context,
        )

        scheduler = Scheduler(
            graph=task_graph,
            executor=executor,
        )

        try:

            result_graph = await scheduler.run()

            run.result = {
                task.task_id: task.output
                for task in result_graph.tasks()
            }

            if result_graph.has_failed():

                run.status = RunStatus.FAILED

                if self.metric_collector is not None:
                    self.metric_collector.record_task(
                        success=False
                    )

                failed_tasks = [
                    task
                    for task in result_graph.tasks()
                    if task.status == TaskStatus.FAILED
                ]

                run.error = "; ".join(
                    f"{task.task_id}: {task.error}"
                    for task in failed_tasks
                )

            else:

                run.status = RunStatus.COMPLETED

                if self.metric_collector is not None:
                    self.metric_collector.record_task(
                        success=True
                    )

            if run_context.trace_root_span_id:

                self.trace_manager.finish_span(
                    run_context.trace_root_span_id,
                    output=run.result,
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

            run.completed_at = datetime.now(
                timezone.utc
            )

        return run

    async def start_planned_run(
        self,
        run_id: str,
        runtime: AgentRuntime,
        planner: Planner,
        principal: Principal | None = None,
    ) -> Run:

        run = self._get_run(run_id)

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

            plan = await planner.plan(
                run.task
            )

            task_graph = (
                self.plan_validator.validate(
                    plan
                )
            )

            has_build_agent = (
                runtime is not None
                and hasattr(runtime, "build_agent")
                and callable(getattr(runtime, "build_agent", None))
            )

            if has_build_agent:
                langgraph_runtime = self._get_or_create_langgraph_runtime(
                    run_id, runtime, run_context=run_context
                )
            elif self._langgraph_runtime is not None:
                langgraph_runtime = self._langgraph_runtime
            else:
                raise RuntimeError(
                    "No LangGraphRuntime available. Provide deepagent_runtime or pre-configure langgraph_runtime."
                )

            executor = AgentTaskExecutor(
                langgraph_runtime=langgraph_runtime,
                run_context=run_context,
            )

            scheduler = Scheduler(
                graph=task_graph,
                executor=executor,
            )

            result_graph = await scheduler.run()

            if result_graph.has_failed():

                def scheduler_factory(
                    new_graph: TaskGraph,
                ) -> Scheduler:

                    return Scheduler(
                        graph=new_graph,
                        executor=executor,
                    )

                result_graph = (
                    await self.recovery_manager.recover(
                        original_task=run.task,
                        graph=result_graph,
                        scheduler_factory=scheduler_factory,
                    )
                )

            run.result = {
                task.task_id: task.output
                for task in result_graph.tasks()
            }

            if result_graph.has_failed():

                run.status = RunStatus.FAILED

                if self.metric_collector is not None:
                    self.metric_collector.record_task(
                        success=False
                    )

                failed_tasks = [
                    task
                    for task in result_graph.tasks()
                    if task.status == TaskStatus.FAILED
                ]

                run.error = "; ".join(
                    f"{task.task_id}: {task.error}"
                    for task in failed_tasks
                )

            else:

                run.status = RunStatus.COMPLETED

                if self.metric_collector is not None:
                    self.metric_collector.record_task(
                        success=True
                    )

            if run_context.trace_root_span_id:

                self.trace_manager.finish_span(
                    run_context.trace_root_span_id,
                    output=run.result,
                )

        except Exception as exc:

            import traceback

            traceback.print_exc()

            run.status = RunStatus.FAILED

            run.error = (
                f"{type(exc).__name__}: {exc}"
            )

            if run_context.trace_root_span_id:

                self.trace_manager.fail_span(
                    run_context.trace_root_span_id,
                    exc,
                )

        finally:

            run.completed_at = datetime.now(
                timezone.utc
            )

        return run

    def _apply_runtime_result(
            self,
            run: Run,
            result: RuntimeResult,
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