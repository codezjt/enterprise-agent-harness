from __future__ import annotations

from typing import Any

from langgraph.types import Command

from enterprise_harness.observability.metrics import MetricCollector
from enterprise_harness.runtime.contract import Runtime
from enterprise_harness.runtime.context import RunContext
from enterprise_harness.runtime.result import RuntimeResult

from ..orchestration.executor import AgentTaskExecutor
from ..orchestration.plan_validator import PlanValidator
from ..orchestration.planner import Planner
from ..orchestration.replanner import Replanner, SimpleReplanner
from ..orchestration.scheduler import Scheduler
from ..orchestration.task import TaskStatus
from ..orchestration.task_graph import TaskGraph
from .recovery import RecoveryManager


class LangGraphRuntime(Runtime):
    """
    LangGraph Durable Runtime — Harness 唯一正式 Runtime 实现。

    架构位置：

        RunManager
            ↓
        Runtime (contract)
            ↓
        LangGraphRuntime  ← 你在这里
            ↓
        DeepAgentRuntime.build_agent()
            ↓
        DeepAgents (LangGraph CompiledStateGraph)
            ↓
        Planner → TaskGraph → Scheduler（如果配置了 Planner）
            ↓
        ToolGateway

    LangGraphRuntime 负责：
      - Durable Execution（checkpoint / interrupt / resume）
      - Planner → TaskGraph → Scheduler 编排
      - Retry → Replan 恢复闭环
      - Approval 暂停 → WAITING_APPROVAL → resume
      - 所有结果统一转换为 RuntimeResult
    """

    def __init__(
        self,
        compiled_graph=None,
        deepagent_runtime=None,
        planner: Planner | None = None,
        plan_validator: PlanValidator | None = None,
        replanner: Replanner | None = None,
        max_replans: int = 1,
        metric_collector: MetricCollector | None = None,
    ):
        self._graph = compiled_graph
        self._deepagent_runtime = deepagent_runtime
        self._planner = planner
        self._plan_validator = plan_validator or PlanValidator()
        self._replanner = replanner or SimpleReplanner()
        self._max_replans = max_replans
        self._metric_collector = metric_collector

        self._recovery_manager = RecoveryManager(
            replanner=self._replanner,
            max_replans=self._max_replans,
            metric_collector=metric_collector,
        )

    def _get_or_build_graph(self, run_context: RunContext):
        if self._graph is not None:
            return self._graph
        if self._deepagent_runtime is not None:
            self._graph = self._deepagent_runtime.build_agent(run_context)
            return self._graph
        raise RuntimeError(
            "LangGraphRuntime has neither compiled_graph nor deepagent_runtime"
        )

    def _config(self, run_id: str) -> dict[str, Any]:
        return {
            "configurable": {
                "thread_id": run_id,
            }
        }

    # ---- Runtime contract implementation ----

    async def run(
        self,
        context: RunContext,
    ) -> RuntimeResult:
        if self._planner is not None:
            return await self._run_planned(context)
        return await self._run_direct(context)

    async def resume(
        self,
        context: RunContext,
        value: Any,
    ) -> RuntimeResult:
        graph = self._get_or_build_graph(context)

        try:
            result = await graph.ainvoke(
                Command(resume=value),
                config=self._config(context.run_id),
            )
        except Exception as exc:
            return RuntimeResult.failed(
                error=str(exc),
                metadata={"phase": "graph_resume"},
            )

        return self._process_graph_result(
            context.run_id,
            result,
        )

    async def cancel(
        self,
        context: RunContext,
    ) -> RuntimeResult:
        graph = self._get_or_build_graph(context)

        state = graph.get_state(config=self._config(context.run_id))

        if state is None or state.next == ():
            return RuntimeResult.cancelled(
                metadata={"phase": "already_terminal"},
            )

        try:
            graph.update_state(
                config=self._config(context.run_id),
                values=None,
            )
        except Exception:
            pass

        return RuntimeResult.cancelled(
            metadata={"phase": "cancelled"},
        )

    # ---- Properties ----

    @property
    def supports_resume(self) -> bool:
        return True

    @property
    def supports_cancel(self) -> bool:
        return True

    # ---- Planned execution (Planner → TaskGraph → Scheduler) ----

    async def _run_planned(
        self,
        context: RunContext,
    ) -> RuntimeResult:
        plan = await self._planner.plan(context.task)
        task_graph = self._plan_validator.validate(plan)

        executor = AgentTaskExecutor(
            runtime=self,
            run_context=context,
        )

        scheduler = Scheduler(graph=task_graph, executor=executor)
        result_graph = await scheduler.run()

        if result_graph.has_failed():
            def scheduler_factory(new_graph: TaskGraph) -> Scheduler:
                return Scheduler(graph=new_graph, executor=executor)

            result_graph = await self._recovery_manager.recover(
                original_task=context.task,
                graph=result_graph,
                scheduler_factory=scheduler_factory,
            )

        task_results = {
            task.task_id: task.output
            for task in result_graph.tasks()
        }

        if result_graph.has_failed():
            failed_tasks = [
                task for task in result_graph.tasks()
                if task.status == TaskStatus.FAILED
            ]
            error_msg = "; ".join(
                f"{task.task_id}: {task.error}"
                for task in failed_tasks
            )
            return RuntimeResult.failed(
                error=error_msg,
                result=task_results,
                metadata={"phase": "planned"},
            )

        return RuntimeResult.completed(
            result=task_results,
            metadata={"phase": "planned"},
        )

    # ---- Direct execution ----

    async def _run_direct(
        self,
        context: RunContext,
    ) -> RuntimeResult:
        graph = self._get_or_build_graph(context)

        input_data = {
            "messages": [
                {
                    "role": "user",
                    "content": context.task,
                }
            ],
            "context": context.context,
        }

        try:
            result = await graph.ainvoke(
                input_data,
                config=self._config(context.run_id),
            )
        except Exception as exc:
            return RuntimeResult.failed(
                error=str(exc),
                metadata={"phase": "graph_invoke"},
            )

        return self._process_graph_result(
            context.run_id,
            result,
        )

    # ---- Internal helpers ----

    def _process_graph_result(
        self,
        run_id: str,
        result: dict[str, Any],
    ) -> RuntimeResult:
        if self.is_interrupted(result):
            interrupt_value = self.extract_interrupt_value(result)
            approval = self.extract_approval(result)

            if approval is not None:
                return RuntimeResult.waiting_approval(
                    approval_id=approval.get("approval_id"),
                    checkpoint_id=run_id,
                    result=approval,
                )

            return RuntimeResult.waiting_approval(
                checkpoint_id=run_id,
                result={"interrupt": interrupt_value},
            )

        output = self.extract_result(result)
        return RuntimeResult.completed(result=output)

    # ---- Static helpers ----

    @staticmethod
    def is_interrupted(result: dict[str, Any]) -> bool:
        return bool(result.get("__interrupt__"))

    @staticmethod
    def extract_interrupt_value(result: dict[str, Any]) -> Any:
        interrupts = result.get("__interrupt__")
        if not interrupts:
            return None
        interrupt_value = interrupts[0]
        return getattr(interrupt_value, "value", interrupt_value)

    @staticmethod
    def extract_approval(result: dict[str, Any]) -> dict[str, Any] | None:
        value = LangGraphRuntime.extract_interrupt_value(result)
        if isinstance(value, dict) and value.get("type") == "approval_required":
            return value
        return None

    @staticmethod
    def extract_result(result: dict[str, Any]) -> Any:
        if isinstance(result, dict):
            messages = result.get("messages")
            if messages:
                last = messages[-1]
                if hasattr(last, "content"):
                    return last.content
                return last
            for key in ("result", "output", "answer"):
                if key in result:
                    return result[key]
        return result