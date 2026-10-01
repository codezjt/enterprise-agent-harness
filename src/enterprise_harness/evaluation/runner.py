from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from enterprise_harness.agent.runtime import AgentRuntime
from enterprise_harness.observability import AuditLogger, MetricCollector, TraceManager
from enterprise_harness.observability.trace import SpanType
from enterprise_harness.policy.rbac import Principal
from enterprise_harness.runtime.models import RunStatus

from .dataset import (
    EvaluationDataset,
    EvaluationInput,
    EvaluationResult,
    ToolEvaluation,
)
from .tool_recorder import ToolCallRecorder


class EvaluationRunner:
    """Evaluation 执行器。

    负责：
    1. 遍历 Dataset 中每条 Case
    2. 通过 RunManager + AgentRuntime 执行
    3. 从 TraceManager 提取真实 Tool Calls
    4. 从 MetricCollector 提取运行时指标
    5. 评估 Tool 选择准确率 + Task Success + 多维度指标
    """

    def __init__(
        self,
        runtime: AgentRuntime,
        run_manager=None,
        principal: Principal | None = None,
        trace_manager: TraceManager | None = None,
        metric_collector: MetricCollector | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.runtime = runtime
        self.run_manager = run_manager
        self.principal = principal
        self.trace_manager = trace_manager or TraceManager()
        self.metric_collector = metric_collector or MetricCollector()
        self.audit_logger = audit_logger or AuditLogger()
        self.tool_recorder = ToolCallRecorder(self.trace_manager)

    async def run(
        self,
        dataset: EvaluationDataset,
    ) -> EvaluationResult:
        result = EvaluationResult(
            run_id=str(uuid4()),
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            total_cases=len(dataset.cases),
        )

        for case in dataset.cases:
            start = time.perf_counter()

            try:
                actual_tools = await self._execute_and_capture(case)
                result.success_cases += 1
            except Exception:
                result.failed_cases += 1
                actual_tools = []

            latency_ms = (time.perf_counter() - start) * 1000
            result.latencies_ms.append(latency_ms)

            tool_eval = self._evaluate_tools(
                case=case,
                actual_tools=actual_tools,
            )
            result.tool_evaluations.append(tool_eval)

        metrics_snapshot = self.metric_collector.snapshot()
        result.replan_count = metrics_snapshot.replans
        result.retry_count = metrics_snapshot.tool_retries
        result.policy_denials = metrics_snapshot.policy_denied
        result.input_tokens = metrics_snapshot.input_tokens
        result.output_tokens = metrics_snapshot.output_tokens
        result.estimated_cost = metrics_snapshot.estimated_cost

        audit_events = self.audit_logger.get_events()
        result.approval_required = sum(
            1 for e in audit_events if e.decision == "require_approval"
        )
        result.approval_approved = sum(
            1 for e in audit_events
            if e.decision == "require_approval"
            and isinstance(e.result, dict)
            and e.result.get("approved", False)
        )

        return result

    async def _execute_and_capture(self, case: EvaluationInput) -> list[str]:
        if self.run_manager is not None:
            return await self._execute_via_run_manager(case)
        return await self._execute_via_runtime(case)

    async def _execute_via_run_manager(self, case: EvaluationInput) -> list[str]:
        run = await self.run_manager.create_run(
            agent_id=self.runtime.config.agent_id,
            task=case.input,
            agent_version=self.runtime.config.version,
            tenant_id=getattr(self.principal, "tenant_id", "default"),
            context=case.metadata or {},
        )

        run_result = await self.run_manager.start_run(
            run_id=run.run_id,
            runtime=self.runtime,
            principal=self.principal,
        )

        tool_names = self.tool_recorder.get_tool_names(run.run_id)

        if run_result.status != RunStatus.COMPLETED:
            raise RuntimeError(
                f"Run {run.run_id} did not complete: {run_result.status.value}"
            )

        return tool_names

    async def _execute_via_runtime(self, case: EvaluationInput) -> list[str]:
        actual_tools: list[str] = []

        trace_id = str(uuid4())
        run_id = f"eval-{trace_id[:12]}"

        root_span = self.trace_manager.start_run_span(
            run_id=run_id,
            agent_name=self.runtime.config.agent_id,
            input=case.input,
        )

        merged_context = {"run_id": run_id, **(case.metadata or {})}

        try:
            await self.runtime.run(
                task=case.input,
                context=merged_context,
            )
            self.trace_manager.finish_span(root_span.span_id)
        except Exception as exc:
            self.trace_manager.fail_span(root_span.span_id, exc)
            raise

        tool_spans = self.trace_manager.get_run_spans(run_id)
        for span in tool_spans:
            if span.component == SpanType.TOOL:
                actual_tools.append(span.name)

        return actual_tools

    @staticmethod
    def _evaluate_tools(
        case: EvaluationInput,
        actual_tools: list[str],
    ) -> ToolEvaluation:
        expected = set(case.expected_tools)
        actual = set(actual_tools)

        tp = len(expected & actual)
        fp = len(actual - expected)
        fn = len(expected - actual)

        return ToolEvaluation(
            case_id=case.id,
            expected=list(expected),
            actual=list(actual),
            tp=tp,
            fp=fp,
            fn=fn,
        )