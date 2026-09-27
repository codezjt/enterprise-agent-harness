from enterprise_harness.agent import AgentConfig
from enterprise_harness.agent.deepagent_runtime import DeepAgentRuntime
from enterprise_harness.context import ContextBuilder
from enterprise_harness.runtime.context import RunContext


def test_run_context_builds_context():
    config = AgentConfig(
        agent_id="test-agent",
        name="Test Agent",
        model="test-model",
    )

    runtime = DeepAgentRuntime(
        config=config,
        context_builder=ContextBuilder(),
    )

    run_context = RunContext(
        run_id="run-1",
        agent_id="test-agent",
        task="分析订单",
        context={
            "system_policy": "禁止删除订单",
            "rag_results": ["订单处理规范"],
            "memory_results": ["用户历史订单"],
        },
    )

    items = runtime.context_builder.build_from_run_context(
        run_context
    )

    run_context.built_context = items

    assert len(run_context.built_context) == 4

    assert run_context.built_context[0].source == (
        "system_policy"
    )

    assert run_context.built_context[1].source == "task"

    assert run_context.built_context[2].source == "rag"

    assert run_context.built_context[3].source == "memory"