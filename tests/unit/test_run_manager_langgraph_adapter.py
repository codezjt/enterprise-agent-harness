from __future__ import annotations

import pytest

from enterprise_harness.runtime.context import RunContext
from enterprise_harness.runtime.langgraph_runtime_adapter import (
    LangGraphRuntimeAdapter,
)
from enterprise_harness.runtime.manager import RunManager
from enterprise_harness.runtime.models import RunStatus
from enterprise_harness.runtime.result import RuntimeResult
from tests.helpers import FakeLangGraphRuntimeForAgent


class FakeLangGraphRuntime:

    async def run(
        self,
        run_id: str,
        input_data: dict,
    ):
        return {
            "result": "completed",
        }

    async def resume(
        self,
        run_id: str,
        value,
    ):
        return {
            "result": "resumed",
        }

    @staticmethod
    def is_interrupted(result):
        return False

    @staticmethod
    def extract_result(result):
        return result.get("result")

    @staticmethod
    def extract_approval(result):
        return None


@pytest.mark.asyncio
async def test_run_manager_start_langgraph_run_uses_unified_runtime_result():
    runtime = FakeLangGraphRuntime()

    manager = RunManager(
        langgraph_runtime=runtime,
    )

    run = await manager.create_run(
        agent_id="test-agent",
        task="hello",
    )

    result = await manager.start_langgraph_run(
        run_id=run.run_id,
    )

    assert result.status == (
        RunStatus.COMPLETED
    )

    assert result.result == "completed"

    assert result.started_at is not None

    assert result.completed_at is not None


@pytest.mark.asyncio
async def test_run_manager_start_langgraph_run_waits_for_approval():
    class ApprovalRuntime:

        async def run(
            self,
            run_id,
            input_data: dict,
        ):
            return {
                "__interrupt__": [
                    {
                        "value": {
                            "type": "approval_required",
                        }
                    }
                ]
            }

        async def resume(
            self,
            run_id,
            value,
        ):
            return {
                "result": "completed",
            }

        @staticmethod
        def is_interrupted(result):
            return bool(
                result.get("__interrupt__")
            )

        @staticmethod
        def extract_result(result):
            return result.get("result")

        @staticmethod
        def extract_approval(result):
            return {
                "status": "REQUIRE_APPROVAL",
                "approval_id": "approval-001",
                "run_id": "run-001",
                "tool_name": "update_order",
                "arguments": {
                    "order_id": "1001",
                },
                "message": "Approval required",
            }

    manager = RunManager(
        langgraph_runtime=ApprovalRuntime(),
    )

    run = await manager.create_run(
        agent_id="test-agent",
        task="update order",
    )

    result = await manager.start_langgraph_run(
        run_id=run.run_id,
    )

    assert result.status == (
        RunStatus.WAITING_APPROVAL
    )

    assert result.approval_id == (
        "approval-001"
    )

    assert result.completed_at is None


@pytest.mark.asyncio
async def test_run_manager_resume_langgraph_run_uses_unified_runtime_result():
    class ApprovalRuntime:

        async def run(
            self,
            run_id,
            input_data: dict,
        ):
            return {
                "__interrupt__": [
                    {
                        "value": {
                            "type": "approval_required",
                        }
                    }
                ]
            }

        async def resume(
            self,
            run_id,
            value,
        ):
            return {
                "result": "resumed",
            }

        @staticmethod
        def is_interrupted(result):
            return bool(
                result.get("__interrupt__")
            )

        @staticmethod
        def extract_result(result):
            return result.get("result")

        @staticmethod
        def extract_approval(result):
            return {
                "status": "REQUIRE_APPROVAL",
                "approval_id": "approval-001",
            }

    manager = RunManager(
        langgraph_runtime=ApprovalRuntime(),
    )

    run = await manager.create_run(
        agent_id="test-agent",
        task="update order",
    )

    run = await manager.start_langgraph_run(
        run_id=run.run_id,
    )

    assert run.status == (
        RunStatus.WAITING_APPROVAL
    )

    assert run.approval_id == (
        "approval-001"
    )

    result = await manager.resume_run(
        run_id=run.run_id,
        value={
            "approval_id": "approval-001",
            "approved": True,
        },
    )

    assert result.status == (
        RunStatus.COMPLETED
    ), result.error
    assert result.result == "resumed"

    assert result.approval_id is None

    assert result.completed_at is not None


@pytest.mark.asyncio
async def test_run_manager_rejects_resume_when_not_waiting():
    runtime = FakeLangGraphRuntime()

    manager = RunManager(
        langgraph_runtime=runtime,
    )

    run = await manager.create_run(
        agent_id="test-agent",
        task="hello",
    )

    with pytest.raises(ValueError):
        await manager.resume_run(
            run_id=run.run_id,
            value={
                "approved": True,
            },
        )