from enterprise_harness.context import ContextBuilder


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