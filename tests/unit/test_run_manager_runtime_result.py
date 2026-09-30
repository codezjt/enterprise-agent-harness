from enterprise_harness.runtime.models import Run, RunStatus
from enterprise_harness.runtime.result import (
    RuntimeResult,
    RuntimeStatus,
)
from enterprise_harness.runtime.manager import RunManager


def create_run() -> Run:
    return Run(
        run_id="run-001",
        agent_id="agent-001",
        task="test task",
    )


def test_apply_runtime_result_completed():
    manager = RunManager()
    run = create_run()

    result = RuntimeResult.completed(
        result="hello",
    )

    manager._apply_runtime_result(run, result)

    assert run.status == RunStatus.COMPLETED
    assert run.result == "hello"
    assert run.approval_id is None


def test_apply_runtime_result_waiting_approval():
    manager = RunManager()
    run = create_run()

    result = RuntimeResult.waiting_approval(
        approval_id="approval-001",
    )

    manager._apply_runtime_result(run, result)

    assert run.status == RunStatus.WAITING_APPROVAL
    assert run.approval_id == "approval-001"


def test_apply_runtime_result_failed():
    manager = RunManager()
    run = create_run()

    result = RuntimeResult.failed(
        "execution failed",
    )

    manager._apply_runtime_result(run, result)

    assert run.status == RunStatus.FAILED
    assert run.error == "execution failed"


def test_apply_runtime_result_cancelled():
    manager = RunManager()
    run = create_run()

    result = RuntimeResult.cancelled()

    manager._apply_runtime_result(run, result)

    assert run.status == RunStatus.CANCELLED


def test_apply_runtime_result_clears_previous_approval():
    manager = RunManager()
    run = create_run()

    run.approval_id = "approval-old"

    result = RuntimeResult.completed(
        result="completed",
    )

    manager._apply_runtime_result(run, result)

    assert run.status == RunStatus.COMPLETED
    assert run.result == "completed"
    assert run.approval_id is None