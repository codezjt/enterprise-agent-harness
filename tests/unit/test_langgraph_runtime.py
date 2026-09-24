import pytest

from enterprise_harness.runtime.langgraph_runtime import (
    LangGraphRuntime,
)


async def execute(state):
    return {
        "result": f"completed: {state['task']}"
    }


@pytest.mark.asyncio
async def test_run():
    runtime = LangGraphRuntime(
        execute=execute,
    )

    result = await runtime.run(
        run_id="run-001",
        task="hello",
    )

    assert result["result"] == "completed: hello"