import pytest

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

    result = await runtime.run(
        run_id="run-001",
        input_data={"task": "hello"},
    )

    assert result["result"] == "completed: hello"