import pytest

from enterprise_harness.agent import (
    AgentConfig,
    AgentRuntime,
)
from enterprise_harness.observability import (
    SpanStatus,
    SpanType,
)
from enterprise_harness.policy.rbac import Principal
from enterprise_harness.runtime import RunManager, LangGraphRuntime
from enterprise_harness.runtime.context import RunContext


class FakeAgentRuntime(AgentRuntime):
    async def build_agent(self, run_context=None):
        from langgraph.graph import StateGraph

        async def execute(state: dict) -> dict:
            task = "unknown"
            if run_context is not None:
                task = run_context.task
            result = f"completed: {task}"
            state["messages"] = [type("msg", (), {"content": result})()]
            return state

        builder = StateGraph(dict)
        builder.add_node("exec", execute)
        builder.set_entry_point("exec")
        builder.set_finish_point("exec")
        return builder.compile()


class FailingAgentRuntime(AgentRuntime):
    async def build_agent(self, run_context=None):
        from langgraph.graph import StateGraph

        async def execute(state: dict) -> dict:
            raise RuntimeError("agent execution failed")

        builder = StateGraph(dict)
        builder.add_node("exec", execute)
        builder.set_entry_point("exec")
        builder.set_finish_point("exec")
        return builder.compile()


@pytest.mark.asyncio
async def test_run_manager_creates_context_and_agent_trace():
    config = AgentConfig(
        agent_id="test-agent",
        name="Test Agent",
        model="test-model",
    )

    agent_runtime = FakeAgentRuntime(config)
    runtime = LangGraphRuntime(deepagent_runtime=agent_runtime)

    manager = RunManager()

    run = await manager.create_run(
        agent_id="test-agent",
        task="查询订单 ORD001",
        context={
            "tenant_id": "tenant-001",
        },
    )

    principal = Principal(
        principal_id="user-001",
        role="operator",
    )

    result = await manager.start_run(
        run_id=run.run_id,
        runtime=runtime,
        principal=principal,
    )

    assert result.status.value == "COMPLETED"

    assert "completed: 查询订单 ORD001" in str(result.result)

    spans = manager.trace_manager.get_run_spans(
        run.run_id
    )

    assert len(spans) == 1

    span = spans[0]

    assert span.component == SpanType.AGENT
    assert span.name == "test-agent"
    assert span.status == SpanStatus.SUCCESS


@pytest.mark.asyncio
async def test_failed_agent_execution_creates_failed_trace():
    config = AgentConfig(
        agent_id="test-agent",
        name="Test Agent",
        model="test-model",
    )

    agent_runtime = FailingAgentRuntime(config)
    runtime = LangGraphRuntime(deepagent_runtime=agent_runtime)

    manager = RunManager()

    run = await manager.create_run(
        agent_id="test-agent",
        task="执行失败测试",
    )

    result = await manager.start_run(
        run_id=run.run_id,
        runtime=runtime,
    )

    assert result.status.value == "FAILED"

    assert result.error == (
        "agent execution failed"
    )

    spans = manager.trace_manager.get_run_spans(
        run.run_id
    )

    assert len(spans) == 1

    span = spans[0]

    assert span.component == SpanType.AGENT
    assert span.status == SpanStatus.FAILED
    assert span.metadata["error"] == (
        "agent execution failed"
    )