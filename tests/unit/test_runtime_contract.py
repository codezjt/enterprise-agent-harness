import pytest

from enterprise_harness.runtime import (
    Runtime,
    RuntimeResult,
    RuntimeStatus,
)


class FakeRuntime(Runtime):

    async def run(self, context):
        return RuntimeResult.completed(
            result={
                "message": "completed",
                "context": context,
            }
        )


class FailingRuntime(Runtime):

    async def run(self, context):
        return RuntimeResult.failed(
            "runtime execution failed"
        )


class WaitingRuntime(Runtime):

    async def run(self, context):
        return RuntimeResult.waiting_approval(
            approval_id="approval-001",
            checkpoint_id="checkpoint-001",
        )

    async def resume(self, context, value):
        return RuntimeResult.completed(
            result={
                "approved": value,
            }
        )

    @property
    def supports_resume(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_runtime_contract_returns_completed_result():
    runtime = FakeRuntime()

    result = await runtime.run(
        {
            "run_id": "run-001",
            "task": "hello",
        }
    )

    assert isinstance(result, RuntimeResult)
    assert result.status == RuntimeStatus.COMPLETED
    assert result.is_completed is True
    assert result.is_failed is False
    assert result.result == {
        "message": "completed",
        "context": {
            "run_id": "run-001",
            "task": "hello",
        },
    }


@pytest.mark.asyncio
async def test_runtime_failed_result():
    runtime = FailingRuntime()

    result = await runtime.run(
        {
            "run_id": "run-001",
        }
    )

    assert result.status == RuntimeStatus.FAILED
    assert result.is_failed is True
    assert result.error == "runtime execution failed"
    assert result.result is None


@pytest.mark.asyncio
async def test_runtime_waiting_approval_result():
    runtime = WaitingRuntime()

    result = await runtime.run(
        {
            "run_id": "run-001",
        }
    )

    assert result.status == RuntimeStatus.WAITING_APPROVAL
    assert result.is_waiting_approval is True
    assert result.approval_id == "approval-001"
    assert result.checkpoint_id == "checkpoint-001"


@pytest.mark.asyncio
async def test_runtime_resume_contract():
    runtime = WaitingRuntime()

    assert runtime.supports_resume is True

    result = await runtime.resume(
        {
            "run_id": "run-001",
        },
        {
            "approval_id": "approval-001",
            "approved": True,
        },
    )

    assert result.status == RuntimeStatus.COMPLETED
    assert result.result == {
        "approved": {
            "approval_id": "approval-001",
            "approved": True,
        }
    }


@pytest.mark.asyncio
async def test_runtime_default_resume_not_supported():
    runtime = FakeRuntime()

    assert runtime.supports_resume is False

    with pytest.raises(NotImplementedError):
        await runtime.resume(
            {
                "run_id": "run-001",
            },
            True,
        )


@pytest.mark.asyncio
async def test_runtime_cancel_default_not_supported():
    runtime = FakeRuntime()

    assert runtime.supports_cancel is False

    with pytest.raises(NotImplementedError):
        await runtime.cancel(
            {
                "run_id": "run-001",
            }
        )


def test_runtime_result_completed_factory():
    result = RuntimeResult.completed(
        result="hello",
        metadata={
            "source": "test",
        },
    )

    assert result.status == RuntimeStatus.COMPLETED
    assert result.result == "hello"
    assert result.error is None
    assert result.metadata == {
        "source": "test",
    }


def test_runtime_result_failed_factory():
    result = RuntimeResult.failed(
        "boom",
        result="partial",
        metadata={
            "retry_count": 2,
        },
    )

    assert result.status == RuntimeStatus.FAILED
    assert result.error == "boom"
    assert result.result == "partial"
    assert result.metadata == {
        "retry_count": 2,
    }


def test_runtime_result_waiting_approval_factory():
    result = RuntimeResult.waiting_approval(
        approval_id="approval-001",
        checkpoint_id="checkpoint-001",
    )

    assert result.status == RuntimeStatus.WAITING_APPROVAL
    assert result.approval_id == "approval-001"
    assert result.checkpoint_id == "checkpoint-001"


def test_runtime_result_cancelled_factory():
    result = RuntimeResult.cancelled(
        result="cancelled-result",
    )

    assert result.status == RuntimeStatus.CANCELLED
    assert result.is_cancelled is True
    assert result.result == "cancelled-result"