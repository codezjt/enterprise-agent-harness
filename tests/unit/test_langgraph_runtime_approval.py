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

    executed = False

    async def execute(state):
        nonlocal executed

        if not executed:
            executed = True

            raise ApprovalRequiredError(
                approval_id="approval-001",
                run_id="run-001",
                tool_name="update_order",
                arguments={
                    "order_id": "1001",
                },
                message="需要审批",
            )

        return {
            "result": "completed",
        }

    runtime = LangGraphRuntime(
        execute=execute,
    )

    result = await runtime.run(
        run_id="run-001",
        task="update order",
    )

    assert runtime.is_interrupted(result)

    approval = runtime.extract_approval(result)

    assert approval is not None
    assert approval["approval_id"] == "approval-001"

    result = await runtime.resume(
        run_id="run-001",
        value={
            "approval_id": "approval-001",
            "approved": True,
        },
    )

    assert not runtime.is_interrupted(result)
    assert runtime.extract_result(result) == "completed"


@pytest.mark.asyncio
async def test_execute_node_preserves_dict_result():
    async def execute(state):
        return {
            "order_id": "1001",
            "quantity": 80,
            "updated": True,
        }

    runtime = LangGraphRuntime(
        execute=execute,
    )

    result = await runtime._execute_node(
        {
            "context": {},
            "result": None,
            "approval": None,
            "approval_id": None,
            "waiting_for_approval": False,
        }
    )

    assert result == {
        "result": {
            "order_id": "1001",
            "quantity": 80,
            "updated": True,
        }
    }


@pytest.mark.asyncio
async def test_execute_node_wraps_plain_result():
    async def execute(state):
        return {
            "order_id": "1001",
            "quantity": 80,
            "updated": True,
        }

    runtime = LangGraphRuntime(
        execute=execute,
    )

    result = await runtime._execute_node(
        {
            "task": "update order",
            "context": {},
        }
    )

    assert result == {
        "result": {
            "order_id": "1001",
            "quantity": 80,
            "updated": True,
        }
    }

@pytest.mark.asyncio
async def test_execute_node_preserves_result_envelope():
    async def execute(state):
        return {
            "result": "completed",
        }

    runtime = LangGraphRuntime(
        execute=execute,
    )

    result = await runtime._execute_node(
        {
            "task": "update order",
            "context": {},
        }
    )

    assert result == {
        "result": "completed",
    }