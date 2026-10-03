import pytest

from enterprise_harness.runtime.context import RunContext
from enterprise_harness.runtime.langgraph_runtime import (
    LangGraphRuntime,
)
from tests.helpers import build_fake_compiled_graph


async def execute(state):
    return {
        "result": f"completed: {state.get('task', '')}"
    }


@pytest.mark.asyncio
async def test_run():
    graph = build_fake_compiled_graph(execute)
    runtime = LangGraphRuntime(compiled_graph=graph)

    context = RunContext(
        run_id="run-001",
        agent_id="test-agent",
        task="hello",
    )

    result = await runtime.run(context)

    assert result.is_completed
    assert result.result["result"] == "completed: hello"