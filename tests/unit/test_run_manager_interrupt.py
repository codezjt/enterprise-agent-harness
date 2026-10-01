import pytest

from enterprise_harness.runtime import (
    RunManager,
    RunStatus,
)
from enterprise_harness.runtime.langgraph_runtime import (
    LangGraphRuntime,
)
from tests.helpers import build_fake_compiled_graph


@pytest.mark.asyncio
async def test_run_waits_for_approval():
    async def execute(state):
        return {"__interrupt__": [{"value": {"type": "approval", "message": "是否执行订单更新？"}}]}

    graph = build_fake_compiled_graph(execute)
    langgraph_runtime = LangGraphRuntime(compiled_graph=graph)

    class FakeBuildAgent:
        def build_agent(self, run_context=None):
            return graph

    manager = RunManager()

    run = await manager.create_run(
        agent_id="order-agent",
        task="更新订单 O1001",
    )

    result = await manager.start_langgraph_run(
        run.run_id,
        deepagent_runtime=FakeBuildAgent(),
    )

    assert result.status == RunStatus.WAITING_APPROVAL
    assert result.completed_at is None


@pytest.mark.asyncio
async def test_resume_after_approval():
    called_count = [0]

    class FakeInterruptRuntime:
        async def run(self, run_id, input_data):
            called_count[0] += 1
            if called_count[0] == 1:
                return {"__interrupt__": [{"value": {"type": "approval", "message": "是否执行订单更新？"}}]}
            return {"result": {"status": "updated", "task": "更新订单 O1001"}}

        async def resume(self, run_id, value):
            called_count[0] += 1
            return {"result": {"status": "updated", "task": "更新订单 O1001"}}

        @staticmethod
        def is_interrupted(result):
            return bool(result.get("__interrupt__"))

        @staticmethod
        def extract_result(result):
            if isinstance(result, dict):
                return result.get("result")
            return result

        @staticmethod
        def extract_approval(result):
            return None

    manager = RunManager(
        langgraph_runtime=FakeInterruptRuntime(),
    )

    run = await manager.create_run(
        agent_id="order-agent",
        task="更新订单 O1001",
    )

    waiting = await manager.start_langgraph_run(
        run.run_id,
    )

    assert waiting.status == RunStatus.WAITING_APPROVAL

    completed = await manager.resume_run(
        run_id=run.run_id,
        value=True,
    )

    assert completed.status == RunStatus.COMPLETED