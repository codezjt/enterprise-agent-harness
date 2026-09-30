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

class RunManager:

    def __init__(
        self,
        langgraph_runtime: LangGraphRuntime | None = None,
        trace_manager: TraceManager | None = None,
        replanner: Replanner | None = None,
        max_replans: int = 1,
        metric_collector: MetricCollector | None = None,
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

        self.plan_validator = PlanValidator()

        self.recovery_manager = RecoveryManager(
            replanner=replanner or SimpleReplanner(),
            max_replans=max_replans,
        )

        self.metric_collector = metric_collector

        self.langgraph_runtime_adapter = (
            LangGraphRuntimeAdapter(langgraph_runtime)
            if langgraph_runtime is not None
            else None
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

        adapter = AgentRuntimeAdapter(runtime)

        try:

            runtime_result = await adapter.run(
                run_context
            )

            if runtime_result.is_failed:

                run.status = RunStatus.FAILED
                run.error = runtime_result.error

                if run_context.trace_root_span_id:
                    self.trace_manager.fail_span(
                        run_context.trace_root_span_id,
                        RuntimeError(
                            runtime_result.error
                            or "Agent runtime failed"
                        ),
                    )

            elif runtime_result.is_cancelled:

                run.status = RunStatus.CANCELLED
                run.result = runtime_result.result

                if run_context.trace_root_span_id:
                    self.trace_manager.finish_span(
                        run_context.trace_root_span_id,
                        output=runtime_result.result,
                    )

            elif runtime_result.is_waiting_approval:

                run.status = RunStatus.WAITING_APPROVAL
                run.result = runtime_result.result

                if runtime_result.approval_id is not None:
                    run.approval_id = runtime_result.approval_id

                if runtime_result.checkpoint_id is not None:
                    run.checkpoint_id = runtime_result.checkpoint_id

            else:

                run.result = runtime_result.result
                run.status = RunStatus.COMPLETED

                if run_context.trace_root_span_id:
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

        finally:

            run.completed_at = (
                datetime.now(timezone.utc)
            )

        return run

    async def start_langgraph_run(
            self,
            run_id: str,
    ) -> Run:

        if self.langgraph_runtime_adapter is None:
            raise RuntimeError(
                "LangGraph runtime is not configured"
            )

        run = self._get_run(run_id)

        run.status = RunStatus.RUNNING

        run.started_at = datetime.now(
            timezone.utc
        )

        run_context = self.context_builder.build(
            run,
        )

        try:
            runtime_result = (
                await self.langgraph_runtime_adapter.run(
                    run_context
                )
            )

            self._apply_runtime_result(
                run,
                runtime_result,
            )

        except Exception as exc:

            run.status = RunStatus.FAILED
            run.error = str(exc)

        if run.status != RunStatus.WAITING_APPROVAL:
            run.completed_at = (
                datetime.now(timezone.utc)
            )

        return run

    async def resume_run(
            self,
            run_id: str,
            value: Any,
    ) -> Run:
        if self.langgraph_runtime_adapter is None:
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

        run_context = self.context_builder.build(
            run,
        )

        try:
            runtime_result = (
                await self.langgraph_runtime_adapter.resume(
                    run_context,
                    value,
                )
            )

            self._apply_runtime_result(
                run,
                runtime_result,
            )

        except Exception as exc:
            run.status = RunStatus.FAILED
            run.error = str(exc)

        if run.status != RunStatus.WAITING_APPROVAL:
            run.completed_at = datetime.now(timezone.utc)

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

        executor = AgentTaskExecutor(
            runtime=runtime,
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

        try:

            # 1. Planner：
            # 自然语言任务 -> Plan
            plan = await planner.plan(
                run.task
            )

            # 2. Plan：
            # Plan -> TaskGraph
            task_graph = (
                self.plan_validator.validate(
                    plan
                )
            )

            # 3. TaskGraph + Scheduler：
            # 执行
            executor = AgentTaskExecutor(
                runtime=runtime,
                run_context=run_context,
            )

            scheduler = Scheduler(
                graph=task_graph,
                executor=executor,
            )

            result_graph = await scheduler.run()

            # 4. 如果任务失败，
            #    进入 Recovery
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

            # 5. 收集最终任务结果
            run.result = {
                task.task_id: task.output
                for task in result_graph.tasks()
            }

            # 6. 根据 Recovery 后的 TaskGraph
            #    最终状态决定 Run 状态
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