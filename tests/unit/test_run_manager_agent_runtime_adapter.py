import pytest

from enterprise_harness.agent.config import AgentConfig
from enterprise_harness.agent.runtime import AgentRuntime
from enterprise_harness.runtime import RunManager
from enterprise_harness.runtime.models import RunStatus


def create_agent_config() -> AgentConfig:
    return AgentConfig(
        agent_id="run-manager-test-agent",
        name="Run Manager Test Agent",
        model="test-model",
    )


class FakeAgentRuntime(AgentRuntime):

    async def run(self, context):
        return await self.run_with_context(context)

    async def run_with_context(self, context):
        return {
            "message": "hello",
            "run_id": context.run_id,
        }


class FailingAgentRuntime(AgentRuntime):

    async def run(self, context):
        return await self.run_with_context(context)

    async def run_with_context(self, context):
        raise RuntimeError(
            "agent execution failed"
        )


@pytest.mark.asyncio
async def test_start_run_uses_agent_runtime_adapter():
    config = create_agent_config()

    runtime = FakeAgentRuntime(config)

    manager = RunManager()

    run = await manager.create_run(
        agent_id=config.agent_id,
        task="hello",
    )

    result = await manager.start_run(
        run_id=run.run_id,
        runtime=runtime,
    )

    assert result.status == RunStatus.COMPLETED

    assert result.result == {
        "message": "hello",
        "run_id": run.run_id,
    }

    assert result.error is None
    assert result.completed_at is not None


@pytest.mark.asyncio
async def test_start_run_preserves_agent_runtime_failure():
    config = create_agent_config()

    runtime = FailingAgentRuntime(config)

    manager = RunManager()

    run = await manager.create_run(
        agent_id=config.agent_id,
        task="hello",
    )

    result = await manager.start_run(
        run_id=run.run_id,
        runtime=runtime,
    )

    assert result.status == RunStatus.FAILED

    assert result.error == (
        "agent execution failed"
    )

    assert result.result is None
    assert result.completed_at is not None


@pytest.mark.asyncio
async def test_start_run_executes_through_agent_runtime_adapter(
    monkeypatch,
):
    config = create_agent_config()

    runtime = FakeAgentRuntime(config)

    manager = RunManager()

    run = await manager.create_run(
        agent_id=config.agent_id,
        task="hello",
    )

    called = False

    async def fake_adapter_run(
        self,
        context,
    ):
        nonlocal called

        called = True

        from enterprise_harness.runtime import (
            RuntimeResult,
        )

        return RuntimeResult.completed(
            result={
                "message": "from-adapter",
            }
        )

    monkeypatch.setattr(
        "enterprise_harness.runtime.agent_runtime_adapter.AgentRuntimeAdapter.run",
        fake_adapter_run,
    )

    result = await manager.start_run(
        run_id=run.run_id,
        runtime=runtime,
    )

    assert called is True

    assert result.status == RunStatus.COMPLETED

    assert result.result == {
        "message": "from-adapter",
    }