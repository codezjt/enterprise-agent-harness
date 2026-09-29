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


@pytest.mark.asyncio
async def test_start_langgraph_run_enters_waiting_approval():
    approval_id = "approval-001"

    async def execute(
        state: dict[str, Any],
    ):
        run_id = state["context"]["run_id"]

        raise ApprovalRequiredError(
            approval_id=approval_id,
            run_id=run_id,
            tool_name="delete_order",
            arguments={
                "order_id": "ORDER-001",
            },
        )

    runtime = LangGraphRuntime(
        execute=execute,
    )

    manager = RunManager(
        langgraph_runtime=runtime,
    )

    run = await manager.create_run(
        agent_id="test-agent",
        task="delete order",
    )

    run.context["run_id"] = run.run_id

    result = await manager.start_langgraph_run(
        run_id=run.run_id,
    )

    assert result.status == (
        RunStatus.WAITING_APPROVAL
    )

    assert result.approval_id == approval_id


@pytest.mark.asyncio
async def test_start_langgraph_run_fails_when_interrupt_has_no_approval():
    async def execute(
        state: dict[str, Any],
    ):
        from langgraph.types import interrupt

        interrupt(
            {
                "type": "manual_input",
                "message": "manual input",
            }
        )

        return {
            "result": "completed",
        }

    runtime = LangGraphRuntime(
        execute=execute,
    )

    manager = RunManager(
        langgraph_runtime=runtime,
    )

    run = await manager.create_run(
        agent_id="test-agent",
        task="manual task",
    )

    result = await manager.start_langgraph_run(
        run_id=run.run_id,
    )

    assert result.status == RunStatus.FAILED

    assert result.error is not None

    assert (
        "approval information"
        in result.error
    )


@pytest.mark.asyncio
async def test_resume_run_requires_waiting_approval():
    async def execute(
        state: dict[str, Any],
    ):
        return {
            "result": "completed",
        }

    runtime = LangGraphRuntime(
        execute=execute,
    )

    manager = RunManager(
        langgraph_runtime=runtime,
    )

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