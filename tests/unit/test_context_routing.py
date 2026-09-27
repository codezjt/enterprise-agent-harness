from enterprise_harness.context import ContextBuilder, ContextRouting


def test_context_routing():
    routing = ContextRouting(
        use_rag_results=False,
        use_memory_results=False,
    )

    builder = ContextBuilder(
        routing=routing,
    )

    items = builder.build(
        system_policy="policy",
        task="task",
        tool_results=["tool"],
        rag_results=["rag"],
        memory_results=["memory"],
        conversation=["conversation"],
    )

    assert [item.source for item in items] == [
        "system_policy",
        "task",
        "tool",
        "conversation",
    ]


def test_context_routing_enabled_sources():
    routing = ContextRouting(
        use_rag_results=False,
        use_memory_results=False,
    )

    assert routing.enabled_sources() == {
        "system_policy",
        "task",
        "tool",
        "conversation",
    }