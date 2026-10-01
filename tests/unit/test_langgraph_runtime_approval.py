import pytest

from enterprise_harness.gateway.exceptions import (
    ApprovalRequiredError,
)
from enterprise_harness.runtime.langgraph_runtime import (
    LangGraphRuntime,
)
from tests.helpers import build_fake_compiled_graph


@pytest.mark.asyncio
async def test_approval_required_is_interrupted():
    async def execute(state):
        return {"result": "ok"}

    graph = build_fake_compiled_graph(execute)
    runtime = LangGraphRuntime(compiled_graph=graph)

    result = await runtime.run(
        run_id="run-001",
        input_data={"task": "test"},
    )

    assert not runtime.is_interrupted(result)


@pytest.mark.asyncio
async def test_execute_node_preserves_dict_result():
    async def execute(state):
        return {
            "order_id": "1001",
            "quantity": 80,
            "updated": True,
        }

    graph = build_fake_compiled_graph(execute)
    runtime = LangGraphRuntime(compiled_graph=graph)

    result = await runtime.run(
        run_id="run-001",
        input_data={"context": {}, "task": "test"},
    )

    assert result["order_id"] == "1001"
    assert result["quantity"] == 80
    assert result["updated"] is True


@pytest.mark.asyncio
async def test_runtime_simple_run():
    async def execute(state):
        return {"result": "completed"}

    graph = build_fake_compiled_graph(execute)
    runtime = LangGraphRuntime(compiled_graph=graph)

    result = await runtime.run(
        run_id="run-001",
        input_data={"task": "update order", "context": {}},
    )

    assert result["result"] == "completed"


@pytest.mark.asyncio
async def test_extract_result_from_dict():
    async def execute(state):
        return {"result": "completed"}

    graph = build_fake_compiled_graph(execute)
    runtime = LangGraphRuntime(compiled_graph=graph)

    result = await runtime.run(
        run_id="run-001",
        input_data={"task": "test"},
    )

    extracted = runtime.extract_result(result)
    assert extracted == "completed"


@pytest.mark.asyncio
async def test_extract_result_fallback():
    async def execute(state):
        return {"data": "some data"}

    graph = build_fake_compiled_graph(execute)
    runtime = LangGraphRuntime(compiled_graph=graph)

    result = await runtime.run(
        run_id="run-001",
        input_data={"task": "test"},
    )

    extracted = runtime.extract_result(result)
    assert extracted == {"data": "some data"}