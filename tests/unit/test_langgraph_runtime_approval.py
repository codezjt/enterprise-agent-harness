from typing import Any

import pytest

from enterprise_harness.gateway.exceptions import (
    ApprovalRequiredError,
)
from enterprise_harness.runtime.langgraph_runtime import (
    LangGraphRuntime,
)


@pytest.mark.asyncio
async def test_approval_required_interrupts_runtime():
    async def execute(
        state: dict[str, Any],
    ):
        raise ApprovalRequiredError(
            approval_id="approval-001",
            run_id="run-001",
            tool_name="delete_order",
            arguments={
                "order_id": "ORDER-001",
            },
        )

    runtime = LangGraphRuntime(
        execute=execute,
    )

    result = await runtime.run(
        run_id="run-001",
        task="delete order",
    )

    assert runtime.is_interrupted(result)

    approval = runtime.extract_approval(
        result
    )

    assert approval is not None
    assert approval["status"] == "REQUIRE_APPROVAL"
    assert approval["approval_id"] == "approval-001"
    assert approval["run_id"] == "run-001"
    assert approval["tool_name"] == "delete_order"
    assert approval["arguments"] == {
        "order_id": "ORDER-001",
    }


@pytest.mark.asyncio
async def test_resume_after_approval():
    async def execute(
        state: dict[str, Any],
    ):
        raise ApprovalRequiredError(
            approval_id="approval-002",
            run_id="run-002",
            tool_name="delete_order",
            arguments={
                "order_id": "ORDER-002",
            },
        )

    runtime = LangGraphRuntime(
        execute=execute,
    )

    result = await runtime.run(
        run_id="run-002",
        task="delete order",
    )

    assert runtime.is_interrupted(result)

    resumed = await runtime.resume(
        run_id="run-002",
        value={
            "approved": True,
        },
    )

    assert resumed is not None