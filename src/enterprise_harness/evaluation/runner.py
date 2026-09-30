from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from enterprise_harness.agent.runtime import AgentRuntime
from enterprise_harness.policy.rbac import Principal

from .dataset import (
    EvaluationDataset,
    EvaluationInput,
    EvaluationResult,
    ToolEvaluation,
)


class EvaluationRunner:
    """Evaluation 执行器。

    负责：
    1. 遍历 Dataset 中每条 Case
    2. 调用 AgentRuntime 执行
    3. 收集执行过程中使用的 Tools
    4. 评估 Tool 选择准确率 + Task Success

    第一版用 Runtime 直接跑，后续可以接入完整 RunManager。
    """

    def __init__(
        self,
        runtime: AgentRuntime,
        principal: Principal | None = None,
    ) -> None:
        self.runtime = runtime
        self.principal = principal

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
            case_tools: list[str] = []
            start = time.perf_counter()

            try:
                agent_result = await self.runtime.run(
                    task=case.input,
                    context={
                        "_evaluation_case": case.id,
                        "_evaluation_capture_tools": case_tools,
                    },
                )
                result.success_cases += 1

            except Exception as exc:
                result.failed_cases += 1
                agent_result = None

            latency_ms = (time.perf_counter() - start) * 1000
            result.latencies_ms.append(latency_ms)

            tool_eval = self._evaluate_tools(
                case=case,
                actual_tools=case_tools,
            )
            result.tool_evaluations.append(tool_eval)

        return result

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
