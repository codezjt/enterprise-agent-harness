from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from enterprise_harness.agent.runtime import AgentRuntime
from enterprise_harness.observability import TraceManager
from enterprise_harness.policy.rbac import Principal

from .context_builder import RunContextBuilder
from .langgraph_runtime import LangGraphRuntime
from .models import Run, RunStatus

from enterprise_harness.orchestration.executor import AgentTaskExecutor
from enterprise_harness.orchestration.scheduler import Scheduler
from enterprise_harness.orchestration.task_graph import TaskGraph
from enterprise_harness.orchestration.task import TaskStatus
from enterprise_harness.orchestration.planner import Planner
from enterprise_harness.orchestration.plan_validator import PlanValidator

from enterprise_harness.orchestration.replanner import SimpleReplanner
from enterprise_harness.runtime.recovery import RecoveryManager

from enterprise_harness.orchestration.replanner import Replanner
from enterprise_harness.orchestration.replanner import SimpleReplanner

from enterprise_harness.observability import (
    MetricCollector,
    TraceManager,
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
            run.completed_at = datetime.now(timezone.utc)

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
            # 1. Planner：自然语言任务 -> Plan
            plan = await planner.plan(run.task)

            # 2. Plan：Plan -> TaskGraph
            task_graph = self.plan_validator.validate(plan)

            # 3. TaskGraph + Scheduler -> 执行
            executor = AgentTaskExecutor(
                runtime=runtime,
                run_context=run_context,
            )

            scheduler = Scheduler(
                graph=task_graph,
                executor=executor,
            )

            result_graph = await scheduler.run()

            # 4. 如果任务失败，进入 Recovery
            if result_graph.has_failed():
                def scheduler_factory(new_graph: TaskGraph) -> Scheduler:
                    return Scheduler(
                        graph=new_graph,
                        executor=executor,
                    )

                result_graph = await self.recovery_manager.recover(
                    original_task=run.task,
                    graph=result_graph,
                    scheduler_factory=scheduler_factory,
                )

            # 5. 收集最终任务结果
            run.result = {
                task.task_id: task.output
                for task in result_graph.tasks()
            }

            # 6. 根据 Recovery 后的 TaskGraph 最终状态
            #    决定 Run 状态
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

            run.error = f"{type(exc).__name__}: {exc}"

            if run_context.trace_root_span_id:
                self.trace_manager.fail_span(
                    run_context.trace_root_span_id,
                    exc,
                )

        finally:
            run.completed_at = datetime.now(timezone.utc)

        return run

    def _apply_runtime_result(
            self,
            run: Run,
            result: dict[str, Any],
    ) -> None:
        if (
                self.langgraph_runtime is not None
                and self.langgraph_runtime.is_interrupted(result)
        ):
            run.status = RunStatus.WAITING_APPROVAL
            return

        run.result = result
        run.status = RunStatus.COMPLETED