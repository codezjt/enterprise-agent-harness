from __future__ import annotations

import pytest

from enterprise_harness.runtime import (
    LangGraphRuntimeAdapter,
    RuntimeResult,
    RuntimeStatus,
)
from enterprise_harness.runtime.context import RunContext


def create_context() -> RunContext:
    return RunContext(
        run_id="run-001",
        agent_id="test-agent",
        task="hello",
        context={
            "user_id": "1001",
        },
    )


class FakeLangGraphRuntime:

    async def run(
        self,
        run_id: str,
        task: str,
        context=None,
    ):
        return {
            "task": task,
            "context": context,
            "result": "completed",
        }

    async def resume(
        self,
        run_id: str,
        value,
    ):
        return {
            "result": value,
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
        interrupts = result.get(
            "__interrupt__"
        )

        if not interrupts:
            return None

        interrupt_value = interrupts[0]

        if isinstance(interrupt_value, dict):
            value = interrupt_value.get(
                "value",
                interrupt_value,
            )
        else:
            value = getattr(
                interrupt_value,
                "value",
                interrupt_value,
            )

        if not isinstance(value, dict):
            return None

        approval = value.get("approval")

        if not isinstance(approval, dict):
            return None

        return approval


class InterruptLangGraphRuntime:

    async def run(
        self,
        run_id: str,
        task: str,
        context=None,
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
        run_id: str,
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
            "run_id": "run-001",
            "tool_name": "update_order",
            "arguments": {
                "order_id": "1001",
            },
            "message": "Approval required",
        }


class FailingLangGraphRuntime:

    async def run(
        self,
        run_id: str,
        task: str,
        context=None,
    ):
        raise RuntimeError(
            "langgraph execution failed"
        )

    async def resume(
        self,
        run_id: str,
        value,
    ):
        raise RuntimeError(
            "langgraph resume failed"
        )

    @staticmethod
    def is_interrupted(result):
        return False

    @staticmethod
    def extract_result(result):
        return None

    @staticmethod
    def extract_approval(result):
        return None


@pytest.mark.asyncio
async def test_langgraph_runtime_adapter_completes_run():
    runtime = FakeLangGraphRuntime()

    adapter = LangGraphRuntimeAdapter(
        runtime
    )

    result = await adapter.run(
        create_context()
    )

    assert isinstance(
        result,
        RuntimeResult,
    )

    assert result.status == (
        RuntimeStatus.COMPLETED
    )

    assert result.is_completed is True

    assert result.result == "completed"


@pytest.mark.asyncio
async def test_langgraph_runtime_adapter_passes_run_context():
    runtime = FakeLangGraphRuntime()

    adapter = LangGraphRuntimeAdapter(
        runtime
    )

    context = create_context()

    result = await adapter.run(context)

    assert result.status == (
        RuntimeStatus.COMPLETED
    )

    assert result.result == "completed"


@pytest.mark.asyncio
async def test_langgraph_runtime_adapter_converts_interrupt_to_waiting_approval():
    runtime = InterruptLangGraphRuntime()

    adapter = LangGraphRuntimeAdapter(
        runtime
    )

    result = await adapter.run(
        create_context()
    )

    assert result.status == (
        RuntimeStatus.WAITING_APPROVAL
    )

    assert result.is_waiting_approval is True

    assert result.approval_id == (
        "approval-001"
    )

    assert result.checkpoint_id is None

    assert result.result == {
        "__interrupt__": [
            {
                "value": {
                    "type": "approval_required",
                }
            }
        ]
    }

    assert result.metadata["approval"] == {
        "status": "REQUIRE_APPROVAL",
        "approval_id": "approval-001",
        "run_id": "run-001",
        "tool_name": "update_order",
        "arguments": {
            "order_id": "1001",
        },
        "message": "Approval required",
    }


@pytest.mark.asyncio
async def test_langgraph_runtime_adapter_resume_completes():
    runtime = InterruptLangGraphRuntime()

    adapter = LangGraphRuntimeAdapter(
        runtime
    )

    result = await adapter.resume(
        create_context(),
        {
            "approved": True,
            "approval_id": "approval-001",
        },
    )

    assert result.status == (
        RuntimeStatus.COMPLETED
    )

    assert result.is_completed is True

    assert result.result == "resumed"


@pytest.mark.asyncio
async def test_langgraph_runtime_adapter_converts_run_exception():
    runtime = FailingLangGraphRuntime()

    adapter = LangGraphRuntimeAdapter(
        runtime
    )

    result = await adapter.run(
        create_context()
    )

    assert result.status == (
        RuntimeStatus.FAILED
    )

    assert result.is_failed is True

    assert result.error == (
        "langgraph execution failed"
    )

    assert result.result is None


@pytest.mark.asyncio
async def test_langgraph_runtime_adapter_converts_resume_exception():
    runtime = FailingLangGraphRuntime()

    adapter = LangGraphRuntimeAdapter(
        runtime
    )

    result = await adapter.resume(
        create_context(),
        {
            "approved": True,
        },
    )

    assert result.status == (
        RuntimeStatus.FAILED
    )

    assert result.is_failed is True

    assert result.error == (
        "langgraph resume failed"
    )

    assert result.result is None


def test_langgraph_runtime_adapter_supports_resume():
    runtime = FakeLangGraphRuntime()

    adapter = LangGraphRuntimeAdapter(
        runtime
    )

    assert adapter.supports_resume is True


def test_langgraph_runtime_adapter_does_not_support_cancel():
    runtime = FakeLangGraphRuntime()

    adapter = LangGraphRuntimeAdapter(
        runtime
    )

    assert adapter.supports_cancel is False