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
from enterprise_harness.runtime import RunManager


class FakeAgentRuntime(AgentRuntime):

    async def run(
        self,
        task: str,
        context=None,
    ):
        return {
            "answer": f"completed: {task}",
            "context": context,
        }


@pytest.mark.asyncio
async def test_run_manager_creates_context_and_agent_trace():

    config = AgentConfig(
        agent_id="test-agent",
        name="Test Agent",
        model="test-model",
    )

    runtime = FakeAgentRuntime(config)

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

    assert result.result["answer"] == (
        "completed: 查询订单 ORD001"
    )

    spans = manager.trace_manager.get_run_spans(
        run.run_id
    )

    assert len(spans) == 1

    span = spans[0]

    assert span.span_type == SpanType.AGENT
    assert span.name == "test-agent"
    assert span.status == SpanStatus.SUCCESS
@pytest.mark.asyncio
async def test_failed_agent_execution_creates_failed_trace():

    config = AgentConfig(
        agent_id="test-agent",
        name="Test Agent",
        model="test-model",
    )

    class FailingRuntime(AgentRuntime):

        async def run(
            self,
            task: str,
            context=None,
        ):
            raise RuntimeError(
                "agent execution failed"
            )

    runtime = FailingRuntime(config)

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

    assert span.span_type == SpanType.AGENT
    assert span.status == SpanStatus.FAILED
    assert span.error == (
        "agent execution failed"
    )