import pytest

from enterprise_harness.runtime import (
    RunManager,
    RunStatus,
)
from enterprise_harness.runtime.context import RunContext
from enterprise_harness.runtime.result import RuntimeResult, RuntimeStatus


class FakeInterruptRuntime:

    def __init__(self):
        self._call_count = 0

    async def run(
        self,
        context: RunContext,
    ) -> RuntimeResult:
        self._call_count += 1
        if self._call_count == 1:
            return RuntimeResult.waiting_approval(
                result={"__interrupt__": [{"value": {"type": "approval", "message": "是否执行订单更新？"}}]}
            )
        return RuntimeResult.completed(
            result={"status": "updated", "task": context.task}
        )

    async def resume(
        self,
        context: RunContext,
        value,
    ) -> RuntimeResult:
        self._call_count += 1
        return RuntimeResult.completed(
            result={"status": "updated", "task": context.task}
        )

    @property
    def supports_resume(self) -> bool:
        return True

    @property
    def supports_cancel(self) -> bool:
        return False


@pytest.mark.asyncio
async def test_run_waits_for_approval():
    runtime = FakeInterruptRuntime()

    manager = RunManager(runtime=runtime)

    run = await manager.create_run(
        agent_id="order-agent",
        task="更新订单 O1001",
    )

    result = await manager.start_run(
        run_id=run.run_id,
        runtime=runtime,
    )

    assert result.status == RunStatus.WAITING_APPROVAL
    assert result.completed_at is None


@pytest.mark.asyncio
async def test_resume_after_approval():
    runtime = FakeInterruptRuntime()

    manager = RunManager(runtime=runtime)

    run = await manager.create_run(
        agent_id="order-agent",
        task="更新订单 O1001",
    )

    waiting = await manager.start_run(
        run_id=run.run_id,
        runtime=runtime,
    )

    assert waiting.status == RunStatus.WAITING_APPROVAL

    completed = await manager.resume_run(
        run_id=run.run_id,
        value=True,
    )

    assert completed.status == RunStatus.COMPLETED