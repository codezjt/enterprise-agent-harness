from enterprise_harness.context import ContextBuilder
from enterprise_harness.runtime.context import RunContext


def test_build_context_from_run_context():
    run_context = RunContext(
        run_id="run-1",
        agent_id="agent-1",
        task="分析订单",
        context={
            "system_policy": "禁止删除订单",
            "tool_results": ["订单状态：已发货"],
            "rag_results": ["订单处理规范"],
            "memory_results": ["用户经常查询订单"],
            "conversation": ["帮我查询订单"],
        },
    )

    builder = ContextBuilder()

    items = builder.build_from_run_context(
        run_context,
    )

    assert [item.source for item in items] == [
        "system_policy",
        "task",
        "tool",
        "rag",
        "memory",
        "conversation",
    ]

    assert items[0].content == "禁止删除订单"
    assert items[1].content == "分析订单"