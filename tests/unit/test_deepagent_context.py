from enterprise_harness.context import ContextBuilder
from enterprise_harness.runtime.context import RunContext


def test_run_context_builds_context():
    run_context = RunContext(
        run_id="run-1",
        agent_id="agent-1",
        task="分析订单",
        context={
            "system_policy": "禁止删除订单",
            "memory_results": ["用户经常查询订单"],
        },
    )

    builder = ContextBuilder()

    built_context = builder.build_from_run_context(
        run_context
    )

    run_context.built_context = built_context

    assert len(run_context.built_context) == 3

    assert run_context.built_context[0].source == "system_policy"
    assert run_context.built_context[1].source == "task"
    assert run_context.built_context[2].source == "memory"