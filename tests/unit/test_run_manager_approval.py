from typing import Any

import pytest

from enterprise_harness.gateway.exceptions import (
    ApprovalRequiredError,
)
from enterprise_harness.runtime.langgraph_runtime import (
    LangGraphRuntime,
)
from enterprise_harness.runtime.models import (
    RunStatus,
)
from enterprise_harness.runtime.manager import (
    RunManager,
)
from tests.helpers import build_fake_compiled_graph


@pytest.mark.asyncio
async def test_start_langgraph_run_enters_waiting_approval():
    approval_id = "approval-001"

    async def execute(state):
        raise ApprovalRequiredError(
            approval_id=approval_id,
            run_id="run-001",
            tool_name="delete_order",
            arguments={"order_id": "ORDER-001"},
        )

    graph = build_fake_compiled_graph(execute)
    langgraph_runtime = LangGraphRuntime(compiled_graph=graph)

    class FakeBuildAgent:
        def build_agent(self, run_context=None):
            return graph

    manager = RunManager()

    run = await manager.create_run(
        agent_id="test-agent",
        task="delete order",
    )

    run.context["run_id"] = run.run_id

    result = await manager.start_langgraph_run(
        run_id=run.run_id,
        deepagent_runtime=FakeBuildAgent(),
    )

    assert result.status in (RunStatus.WAITING_APPROVAL, RunStatus.FAILED)


@pytest.mark.asyncio
async def test_start_langgraph_run_fails_when_interrupt_has_no_approval():
    async def execute(state):
        return {"__interrupt__": [{"value": {"type": "manual_input", "message": "manual input"}}]}

    graph = build_fake_compiled_graph(execute)

    class FakeBuildAgent:
        def build_agent(self, run_context=None):
            return graph

    manager = RunManager()

    run = await manager.create_run(
        agent_id="test-agent",
        task="manual task",
    )

    result = await manager.start_langgraph_run(
        run_id=run.run_id,
        deepagent_runtime=FakeBuildAgent(),
    )

    assert result.status in (RunStatus.WAITING_APPROVAL, RunStatus.FAILED)


@pytest.mark.asyncio
async def test_resume_run_requires_waiting_approval():
    manager = RunManager()

    run = await manager.create_run(
        agent_id="test-agent",
        task="normal task",
    )

    with pytest.raises(ValueError):
        await manager.resume_run(
            run_id=run.run_id,
            value={
                "approved": True,
            },
        )