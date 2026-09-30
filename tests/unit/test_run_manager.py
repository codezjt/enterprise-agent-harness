import pytest

from enterprise_harness.agent import AgentConfig, AgentRuntime
from enterprise_harness.runtime import RunManager, RunStatus


class FakeAgentRuntime(AgentRuntime):

    async def run(self, task: str, context=None):
        return {
            "answer": f"completed: {task}",
            "context": context,
        }


def create_runtime() -> FakeAgentRuntime:
    config = AgentConfig(
        agent_id="test-agent",
        name="Test Agent",
        model="test-model",
    )

    return FakeAgentRuntime(config)


@pytest.mark.asyncio
async def test_create_run():
    manager = RunManager()

    run = await manager.create_run(
        agent_id="test-agent",
        task="test task",
    )

    assert run.agent_id == "test-agent"
    assert run.task == "test task"
    assert run.status == RunStatus.CREATED
    assert run.run_id in manager.runs


@pytest.mark.asyncio
async def test_start_run():
    manager = RunManager()
    runtime = create_runtime()

    run = await manager.create_run(
        agent_id="test-agent",
        task="test task",
    )

    result = await manager.start_run(
        run_id=run.run_id,
        runtime=runtime,
    )

    assert result.status == RunStatus.COMPLETED
    assert result.result["answer"] == "completed: test task"
    assert result.started_at is not None
    assert result.completed_at is not None


@pytest.mark.asyncio
async def test_failed_run():
    class FailedRuntime(AgentRuntime):

        async def run(self, task: str, context=None):
            raise RuntimeError("agent failed")

    config = AgentConfig(
        agent_id="test-agent",
        name="Test Agent",
        model="test-model",
    )

    runtime = FailedRuntime(config)

    manager = RunManager()

    run = await manager.create_run(
        agent_id="test-agent",
        task="test task",
    )

    result = await manager.start_run(
        run_id=run.run_id,
        runtime=runtime,
    )

    assert result.status == RunStatus.FAILED
    assert result.error == "agent failed"
    assert result.completed_at is not None


@pytest.mark.asyncio
async def test_cancel_run():
    manager = RunManager()

    run = await manager.create_run(
        agent_id="test-agent",
        task="test task",
    )

    result = await manager.cancel_run(run.run_id)

    assert result.status == RunStatus.CANCELLED
    assert result.completed_at is not None


@pytest.mark.asyncio
async def test_get_run():
    manager = RunManager()

    run = await manager.create_run(
        agent_id="test-agent",
        task="test task",
    )

    result = await manager.get_run(run.run_id)

    assert result.run_id == run.run_id