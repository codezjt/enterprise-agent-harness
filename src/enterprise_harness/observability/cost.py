from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """
    模型 Token 价格。

    单位：
    - input_per_1k: 每 1000 input tokens 的价格
    - output_per_1k: 每 1000 output tokens 的价格

    具体货币单位由调用方约定，例如 USD。
    """

    input_per_1k: float
    output_per_1k: float

    def __post_init__(self) -> None:
        if self.input_per_1k < 0:
            raise ValueError("input_per_1k must be greater than or equal to 0")

        if self.output_per_1k < 0:
            raise ValueError("output_per_1k must be greater than or equal to 0")


@dataclass(frozen=True)
class CostRecord:
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float


class CostTracker:
    """
    LLM Token 成本统计器。

    职责：
    1. 根据 Token 数量和模型价格计算单次成本
    2. 累计 Input / Output Token
    3. 累计 Estimated Cost
    4. 提供当前成本快照

    不负责：
    - 查询模型价格
    - 调用模型
    - 持久化数据库
    """

    def __init__(self, pricing: ModelPricing) -> None:
        self.pricing = pricing

        self._input_tokens = 0
        self._output_tokens = 0
        self._estimated_cost = 0.0

    def record(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> CostRecord:
        self._validate_tokens(input_tokens, output_tokens)

        input_cost = (
            input_tokens / 1000
        ) * self.pricing.input_per_1k

        output_cost = (
            output_tokens / 1000
        ) * self.pricing.output_per_1k

        total_cost = input_cost + output_cost

        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        self._estimated_cost += total_cost

        return CostRecord(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
        )

    @property
    def input_tokens(self) -> int:
        return self._input_tokens

    @property
    def output_tokens(self) -> int:
        return self._output_tokens

    @property
    def total_tokens(self) -> int:
        return self._input_tokens + self._output_tokens

    @property
    def estimated_cost(self) -> float:
        return self._estimated_cost

    def snapshot(self) -> dict[str, int | float]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost": self.estimated_cost,
        }

    def reset(self) -> None:
        self._input_tokens = 0
        self._output_tokens = 0
        self._estimated_cost = 0.0

    @staticmethod
    def _validate_tokens(
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