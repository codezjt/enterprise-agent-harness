import pytest

from enterprise_harness.context import ContextItem, TokenBudget


def test_token_budget():
    budget = TokenBudget(max_tokens=10)

    items = [
        ContextItem(
            priority=0,
            source="system_policy",
            content="12345",
        ),
        ContextItem(
            priority=1,
            source="task",
            content="12345",
        ),
        ContextItem(
            priority=2,
            source="tool",
            content="12345",
        ),
    ]

    result = budget.fit(items)

    assert len(result) == 2
    assert result[0].source == "system_policy"
    assert result[1].source == "task"


def test_token_budget_rejects_invalid_limit():
    with pytest.raises(ValueError):
        TokenBudget(max_tokens=0)


def test_token_budget_estimate():
    budget = TokenBudget(max_tokens=10)

    assert budget.estimate("hello") == 5