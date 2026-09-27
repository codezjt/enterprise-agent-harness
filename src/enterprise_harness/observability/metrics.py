from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable
from .cost import CostTracker


@dataclass
class MetricSnapshot:
    """
    当前运行时指标快照。

    第一版采用内存聚合：
    - 简单
    - 无外部基础设施
    - 方便测试
    后续可以替换成 Prometheus / OpenTelemetry Metrics。
    """

    agent_runs: int = 0
    agent_successes: int = 0

    task_runs: int = 0
    task_successes: int = 0

    tool_calls: int = 0
    tool_successes: int = 0

    tool_retries: int = 0
    replans: int = 0
    human_approvals: int = 0

    policy_denied: int = 0

    latencies_ms: list[float] = field(default_factory=list)

    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0

    @property
    def agent_success_rate(self) -> float:
        return self._rate(self.agent_successes, self.agent_runs)

    @property
    def task_success_rate(self) -> float:
        return self._rate(self.task_successes, self.task_runs)

    @property
    def tool_success_rate(self) -> float:
        return self._rate(self.tool_successes, self.tool_calls)

    @property
    def tool_retry_rate(self) -> float:
        return self._rate(self.tool_retries, self.tool_calls)

    @property
    def replan_rate(self) -> float:
        return self._rate(self.replans, self.task_runs)

    @property
    def average_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return mean(self.latencies_ms)

    @property
    def p95_latency_ms(self) -> float:
        return self._percentile(self.latencies_ms, 0.95)

    @property
    def p99_latency_ms(self) -> float:
        return self._percentile(self.latencies_ms, 0.99)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        return {
            "agent_runs": self.agent_runs,
            "agent_successes": self.agent_successes,
            "agent_success_rate": self.agent_success_rate,
            "task_runs": self.task_runs,
            "task_successes": self.task_successes,
            "task_success_rate": self.task_success_rate,
            "tool_calls": self.tool_calls,
            "tool_successes": self.tool_successes,
            "tool_success_rate": self.tool_success_rate,
            "tool_retries": self.tool_retries,
            "tool_retry_rate": self.tool_retry_rate,
            "replans": self.replans,
            "replan_rate": self.replan_rate,
            "human_approvals": self.human_approvals,
            "policy_denied": self.policy_denied,
            "average_latency_ms": self.average_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.estimated_cost,
        }

    @staticmethod
    def _rate(successes: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return successes / total

    @staticmethod
    def _percentile(
        values: Iterable[float],
        percentile: float,
    ) -> float:
        values = sorted(values)

        if not values:
            return 0.0

        if not 0 < percentile <= 1:
            raise ValueError("percentile must be between 0 and 1")

        index = (len(values) - 1) * percentile
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)

        if lower == upper:
            return float(values[lower])

        weight = index - lower

        return (
            values[lower] * (1 - weight)
            + values[upper] * weight
        )


class MetricCollector:
    """
    Metrics 聚合器。

    对外提供业务语义明确的记录方法，
    避免业务代码直接修改 MetricSnapshot。
    """

    def __init__(self, cost_tracker: CostTracker | None = None) -> None:
        self.cost_tracker = cost_tracker

        self._agent_runs = 0
        self._agent_successes = 0
        self._task_runs = 0
        self._task_successes = 0
        self._tool_calls = 0
        self._tool_successes = 0
        self._tool_retries = 0
        self._replans = 0
        self._human_approvals = 0
        self._policy_denied = 0
        self._latencies_ms: list[float] = []
        self._input_tokens = 0
        self._output_tokens = 0
        self._estimated_cost = 0.0

    def record_agent_run(self, *, success: bool) -> None:
        self._agent_runs += 1

        if success:
            self._agent_successes += 1

    def record_task(self, *, success: bool) -> None:
        self._task_runs += 1

        if success:
            self._task_successes += 1

    def record_tool(self, *, success: bool) -> None:
        self._tool_calls += 1

        if success:
            self._tool_successes += 1

    def record_tool_retry(self) -> None:
        self._tool_retries += 1

    def record_replan(self) -> None:
        self._replans += 1

    def record_human_approval(self) -> None:
        self._human_approvals += 1

    def record_policy_denied(self) -> None:
        self._policy_denied += 1

    def record_latency(self, latency_ms: float) -> None:
        if latency_ms < 0:
            raise ValueError("latency_ms must be greater than or equal to 0")

        self._latencies_ms.append(latency_ms)

    def record_tokens(
            self,
            *,
            input_tokens: int,
            output_tokens: int,
    ) -> None:
        if input_tokens < 0:
            raise ValueError(
                "input_tokens must be greater than or equal to 0"
            )

        if output_tokens < 0:
            raise ValueError(
                "output_tokens must be greater than or equal to 0"
            )

        self._input_tokens += input_tokens
        self._output_tokens += output_tokens

        if self.cost_tracker is not None:
            record = self.cost_tracker.record(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            self._estimated_cost += record.total_cost

    def record_cost(self, cost: float) -> None:
        if cost < 0:
            raise ValueError("cost must be greater than or equal to 0")

        self._estimated_cost += cost

    def snapshot(self) -> MetricSnapshot:
        return MetricSnapshot(
            agent_runs=self._agent_runs,
            agent_successes=self._agent_successes,
            task_runs=self._task_runs,
            task_successes=self._task_successes,
            tool_calls=self._tool_calls,
            tool_successes=self._tool_successes,
            tool_retries=self._tool_retries,
            replans=self._replans,
            human_approvals=self._human_approvals,
            policy_denied=self._policy_denied,
            latencies_ms=list(self._latencies_ms),
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            estimated_cost=self._estimated_cost,
        )

    def reset(self) -> None:
        self._agent_runs = 0
        self._agent_successes = 0
        self._task_runs = 0
        self._task_successes = 0
        self._tool_calls = 0
        self._tool_successes = 0
        self._tool_retries = 0
        self._replans = 0
        self._human_approvals = 0
        self._policy_denied = 0
        self._latencies_ms = []
        self._input_tokens = 0
        self._output_tokens = 0
        self._estimated_cost = 0.0