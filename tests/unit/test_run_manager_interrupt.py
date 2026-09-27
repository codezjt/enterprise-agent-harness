import pytest

from enterprise_harness.runtime import (
    RunManager,
    RunStatus,
)
from enterprise_harness.runtime.langgraph_runtime import (
    LangGraphRuntime,
)
from langgraph.types import interrupt


async def execute(state):
    approval = interrupt(
        {
            "type": "approval",
            "message": "是否执行订单更新？",
        }
    )

    if approval:
        return {
            "result": {
                "status": "updated",
                "task": state["task"],
            }
        }

    return {
        "result": {
            "status": "rejected",
        }
    }

@pytest.mark.asyncio
async def test_run_waits_for_approval():
    runtime = LangGraphRuntime(
        execute=execute,
    )

    manager = RunManager(
        langgraph_runtime=runtime,
    )

    run = await manager.create_run(
        agent_id="order-agent",
        task="更新订单 O1001",
    )

    result = await manager.start_langgraph_run(
        run.run_id,
    )

    print("RUN STATUS:", result.status)
    print("RUN ERROR:", result.error)
    print("RUN RESULT:", result.result)
    print("CHECKPOINT:", result.checkpoint_id)
    assert result.status == RunStatus.WAITING_APPROVAL
    assert result.completed_at is None


@pytest.mark.asyncio
async def test_resume_after_approval():
    runtime = LangGraphRuntime(
        execute=execute,
    )

    manager = RunManager(
        langgraph_runtime=runtime,
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

    assert completed.result == {
        "status": "updated",
        "task": "更新订单 O1001",
    }