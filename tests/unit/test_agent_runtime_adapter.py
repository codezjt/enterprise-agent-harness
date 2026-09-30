import pytest

from enterprise_harness.agent.config import AgentConfig
from enterprise_harness.agent.runtime import AgentRuntime
from enterprise_harness.runtime import (
    AgentRuntimeAdapter,
    RuntimeResult,
    RuntimeStatus,
)


def create_agent_config() -> AgentConfig:
    return AgentConfig(
        agent_id="test-agent",
        name="Test Agent",
        model="test-model",
    )


class FakeAgentRuntime(AgentRuntime):

    async def run(self, context):
        return await self.run_with_context(context)

    async def run_with_context(self, context):
        return {
            "message": "completed",
            "context": context,
        }


class StringAgentRuntime(AgentRuntime):

    async def run(self, context):
        return await self.run_with_context(context)

    async def run_with_context(self, context):
        return "completed"


class RuntimeResultAgentRuntime(AgentRuntime):

    async def run(self, context):
        return await self.run_with_context(context)

    async def run_with_context(self, context):
        return RuntimeResult.completed(
            result={
                "message": "already normalized",
            }
        )


class FailingAgentRuntime(AgentRuntime):

    async def run(self, context):
        return await self.run_with_context(context)

    async def run_with_context(self, context):
        raise RuntimeError(
            "agent execution failed"
        )


@pytest.mark.asyncio
async def test_agent_runtime_adapter_executes_existing_runtime():
    config = create_agent_config()
    runtime = FakeAgentRuntime(config)

    adapter = AgentRuntimeAdapter(
        runtime
    )

    result = await adapter.run(
        {
            "run_id": "run-001",
            "task": "hello",
        }
    )

    assert isinstance(result, RuntimeResult)
    assert result.status == RuntimeStatus.COMPLETED

    assert result.result == {
        "message": "completed",
        "context": {
            "run_id": "run-001",
            "task": "hello",
        },
    }


@pytest.mark.asyncio
async def test_agent_runtime_adapter_preserves_string_result():
    config = create_agent_config()
    runtime = StringAgentRuntime(config)

    adapter = AgentRuntimeAdapter(
        runtime
    )

    result = await adapter.run(
        {
            "run_id": "run-001",
        }
    )

    assert result.status == RuntimeStatus.COMPLETED
    assert result.result == "completed"


@pytest.mark.asyncio
async def test_agent_runtime_adapter_preserves_runtime_result():
    config = create_agent_config()
    runtime = RuntimeResultAgentRuntime(config)

    adapter = AgentRuntimeAdapter(
        runtime
    )

    result = await adapter.run(
        {
            "run_id": "run-001",
        }
    )

    assert result.status == RuntimeStatus.COMPLETED

    assert result.result == {
        "message": "already normalized",
    }


@pytest.mark.asyncio
async def test_agent_runtime_adapter_converts_exception_to_failed_result():
    config = create_agent_config()
    runtime = FailingAgentRuntime(config)

    adapter = AgentRuntimeAdapter(
        runtime
    )

    result = await adapter.run(
        {
            "run_id": "run-001",
        }
    )

    assert result.status == RuntimeStatus.FAILED
    assert result.is_failed is True
    assert result.error == "agent execution failed"
    assert result.result is None


def test_agent_runtime_adapter_does_not_support_resume():
    config = create_agent_config()
    runtime = FakeAgentRuntime(config)

    adapter = AgentRuntimeAdapter(
        runtime
    )

    assert adapter.supports_resume is False


@pytest.mark.asyncio
async def test_agent_runtime_adapter_resume_is_not_supported():
    config = create_agent_config()
    runtime = FakeAgentRuntime(config)

    adapter = AgentRuntimeAdapter(
        runtime
    )

    with pytest.raises(NotImplementedError):
        await adapter.resume(
            {
                "run_id": "run-001",
            },
            {
                "approved": True,
            },
        )


def test_agent_runtime_adapter_does_not_support_cancel():
    config = create_agent_config()
    runtime = FakeAgentRuntime(config)

    adapter = AgentRuntimeAdapter(
        runtime
    )

    assert adapter.supports_cancel is False


@pytest.mark.asyncio
async def test_agent_runtime_adapter_cancel_is_not_supported():
    config = create_agent_config()
    runtime = FakeAgentRuntime(config)

    adapter = AgentRuntimeAdapter(
        runtime
    )

    with pytest.raises(NotImplementedError):
        await adapter.cancel(
            {
                "run_id": "run-001",
            }
        )