import pytest

from enterprise_harness.agent import AgentConfig, AgentRuntime
from enterprise_harness.runtime import RunManager, RunStatus, LangGraphRuntime


class FakeAgentRuntime(AgentRuntime):
    async def build_agent(self, run_context=None):
        from langgraph.graph import StateGraph

        async def execute(state: dict) -> dict:
            task = "unknown"
            ctx = {}
            if run_context is not None:
                task = run_context.task
                ctx = run_context.context
            result = {
                "answer": f"completed: {task}",
                "context": ctx,
            }
            state["messages"] = [type("msg", (), {"content": result})()]
            return state

        builder = StateGraph(dict)
        builder.add_node("exec", execute)
        builder.set_entry_point("exec")
        builder.set_finish_point("exec")
        return builder.compile()


def create_runtime() -> LangGraphRuntime:
    config = AgentConfig(
        agent_id="test-agent",
        name="Test Agent",
        model="test-model",
    )
    return LangGraphRuntime(deepagent_runtime=FakeAgentRuntime(config))


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
    assert run.trace_id != ""


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
    class FailingAgentRuntime(AgentRuntime):
        async def build_agent(self, run_context=None):
            from langgraph.graph import StateGraph

            async def execute(state: dict) -> dict:
                raise RuntimeError("agent failed")

            builder = StateGraph(dict)
            builder.add_node("exec", execute)
            builder.set_entry_point("exec")
            builder.set_finish_point("exec")
            return builder.compile()

    config = AgentConfig(
        agent_id="test-agent",
        name="Test Agent",
        model="test-model",
    )
    runtime = LangGraphRuntime(deepagent_runtime=FailingAgentRuntime(config))

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