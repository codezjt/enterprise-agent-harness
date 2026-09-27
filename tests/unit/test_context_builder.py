from enterprise_harness.context import ContextBuilder
from enterprise_harness.context import TokenBudget

def test_context_builder_priority():
    builder = ContextBuilder()

    items = builder.build(
        system_policy="policy",
        task="分析订单",
        tool_results=["order"],
        rag_results=["rag"],
        memory_results=["memory"],
        conversation=["hello"],
    )

    assert [item.priority for item in items] == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]

    assert [item.source for item in items] == [
        "system_policy",
        "task",
        "tool",
        "rag",
        "memory",
        "conversation",
    ]

    def test_context_builder_with_token_budget():
        builder = ContextBuilder(
            token_budget=TokenBudget(max_tokens=10),
        )

        items = builder.build(
            system_policy="12345",
            task="12345",
            tool_results=["12345"],
        )

        assert len(items) == 2

        assert items[0].source == "system_policy"
        assert items[1].source == "task"