from enterprise_harness.orchestration.task import Task
from enterprise_harness.runtime.retry import (
    FailureType,
    RetryPolicy,
)


def test_retry_transient_error():
    task = Task(
        task_id="T1",
        name="查询库存",
        error="inventory service timeout",
        retry_count=0,
        max_retries=2,
    )

    decision = RetryPolicy().should_retry(task)

    assert decision.retry is True
    assert decision.failure_type == FailureType.TRANSIENT


def test_transient_error_stops_after_max_retries():
    task = Task(
        task_id="T1",
        name="查询库存",
        error="inventory service timeout",
        retry_count=2,
        max_retries=2,
    )

    decision = RetryPolicy().should_retry(task)

    assert decision.retry is False
    assert decision.failure_type == FailureType.TRANSIENT


def test_permission_error_should_not_retry():
    task = Task(
        task_id="T1",
        name="修改订单",
        error="permission denied",
        retry_count=0,
        max_retries=3,
    )

    decision = RetryPolicy().should_retry(task)

    assert decision.retry is False
    assert decision.failure_type == FailureType.PERMISSION


def test_validation_error_should_not_retry():
    task = Task(
        task_id="T1",
        name="调用工具",
        error="invalid parameter",
        retry_count=0,
        max_retries=3,
    )

    decision = RetryPolicy().should_retry(task)

    assert decision.retry is False
    assert decision.failure_type == FailureType.VALIDATION


def test_business_error_should_not_retry():
    task = Task(
        task_id="T1",
        name="更新订单",
        error="business error: order not supported",
        retry_count=0,
        max_retries=3,
    )

    decision = RetryPolicy().should_retry(task)

    assert decision.retry is False
    assert decision.failure_type == FailureType.BUSINESS


def test_unknown_error_can_retry():
    task = Task(
        task_id="T1",
        name="执行任务",
        error="unexpected internal error",
        retry_count=0,
        max_retries=1,
    )

    decision = RetryPolicy().should_retry(task)

    assert decision.retry is True
    assert decision.failure_type == FailureType.UNKNOWN


def test_unknown_error_stops_after_max_retries():
    task = Task(
        task_id="T1",
        name="执行任务",
        error="unexpected internal error",
        retry_count=1,
        max_retries=1,
    )

    decision = RetryPolicy().should_retry(task)

    assert decision.retry is False
    assert decision.failure_type == FailureType.UNKNOWN