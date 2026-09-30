from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class EvaluationInput(BaseModel):
    """一条 Evaluation Dataset 记录。"""

    id: str
    input: str
    expected_tools: list[str] = Field(default_factory=list)
    expected_result: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class EvaluationDataset:
    """Evaluation 数据集。"""

    name: str
    version: str = "1.0.0"
    cases: list[EvaluationInput] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.cases:
            return

        seen: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                raise ValueError(
                    f"Duplicate case id: {case.id}"
                )
            seen.add(case.id)

    def get_case(self, case_id: str) -> EvaluationInput:
        for case in self.cases:
            if case.id == case_id:
                return case
        raise KeyError(f"Case not found: {case_id}")


@dataclass
class ToolEvaluation:
    """单条 Case 的 Tool 选择评估结果。"""

    case_id: str
    expected: list[str]
    actual: list[str]
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        total = self.tp + self.fp
        return self.tp / total if total > 0 else 0.0

    @property
    def recall(self) -> float:
        total = self.tp + self.fn
        return self.tp / total if total > 0 else 0.0


@dataclass
class EvaluationResult:
    """一次完整 Evaluation Run 的结果。"""

    run_id: str
    dataset_name: str
    dataset_version: str

    total_cases: int = 0
    success_cases: int = 0
    failed_cases: int = 0

    tool_evaluations: list[ToolEvaluation] = field(default_factory=list)

    latencies_ms: list[float] = field(default_factory=list)
    estimated_cost: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_cases == 0:
            return 0.0
        return self.success_cases / self.total_cases

    @property
    def average_tool_precision(self) -> float:
        if not self.tool_evaluations:
            return 0.0
        return sum(t.precision for t in self.tool_evaluations) / len(self.tool_evaluations)

    @property
    def average_tool_recall(self) -> float:
        if not self.tool_evaluations:
            return 0.0
        return sum(t.recall for t in self.tool_evaluations) / len(self.tool_evaluations)

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset": f"{self.dataset_name}@{self.dataset_version}",
            "total_cases": self.total_cases,
            "success_rate": self.success_rate,
            "tool_precision": self.average_tool_precision,
            "tool_recall": self.average_tool_recall,
            "estimated_cost": self.estimated_cost,
        }
