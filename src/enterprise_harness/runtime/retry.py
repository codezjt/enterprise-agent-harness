from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from enterprise_harness.orchestration.task import Task


class FailureType(str, Enum):
    """
    任务失败类型。
    """

    TRANSIENT = "TRANSIENT"
    PERMISSION = "PERMISSION"
    VALIDATION = "VALIDATION"
    BUSINESS = "BUSINESS"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RetryDecision:
    """
    Retry 判断结果。
    """

    retry: bool
    reason: str = ""
    failure_type: FailureType = FailureType.UNKNOWN


class RetryPolicy:
    """
    企业任务 Retry 策略。

    当前根据：
    - 失败类型
    - retry_count
    - max_retries

    判断是否 Retry。
    """

    def should_retry(
        self,
        task: Task,
    ) -> RetryDecision:

        failure_type = self.classify_failure(
            task.error
        )

        # 临时性错误允许 Retry
        if failure_type == FailureType.TRANSIENT:
            if task.retry_count < task.max_retries:
                return RetryDecision(
                    retry=True,
                    reason=(
                        f"task {task.task_id} "
                        f"has transient failure "
                        f"and can retry "
                        f"({task.retry_count}/{task.max_retries})"
                    ),
                    failure_type=failure_type,
                )

            return RetryDecision(
                retry=False,
                reason=(
                    f"task {task.task_id} "
                    f"reached max retries"
                ),
                failure_type=failure_type,
            )

        # 权限、参数、业务错误不应该简单 Retry
        if failure_type in {
            FailureType.PERMISSION,
            FailureType.VALIDATION,
            FailureType.BUSINESS,
        }:
            return RetryDecision(
                retry=False,
                reason=(
                    f"task {task.task_id} "
                    f"has non-retryable "
                    f"{failure_type.value.lower()} failure"
                ),
                failure_type=failure_type,
            )

        # 未知异常暂时按照 retry_count 判断
        if task.retry_count < task.max_retries:
            return RetryDecision(
                retry=True,
                reason=(
                    f"task {task.task_id} "
                    f"has unknown failure "
                    f"and can retry "
                    f"({task.retry_count}/{task.max_retries})"
                ),
                failure_type=failure_type,
            )

        return RetryDecision(
            retry=False,
            reason=(
                f"task {task.task_id} "
                f"reached max retries"
            ),
            failure_type=failure_type,
        )

    @staticmethod
    def classify_failure(
        error: str | None,
    ) -> FailureType:

        if not error:
            return FailureType.UNKNOWN

        message = error.lower()

        if any(
            keyword in message
            for keyword in (
                "timeout",
                "timed out",
                "connection",
                "temporarily",
                "temporary",
                "unavailable",
                "503",
                "502",
                "429",
            )
        ):
            return FailureType.TRANSIENT

        if any(
            keyword in message
            for keyword in (
                "permission denied",
                "forbidden",
                "unauthorized",
                "access denied",
                "403",
                "401",
            )
        ):
            return FailureType.PERMISSION

        if any(
            keyword in message
            for keyword in (
                "invalid argument",
                "validation",
                "schema",
                "invalid parameter",
                "parameter error",
            )
        ):
            return FailureType.VALIDATION

        if any(
            keyword in message
            for keyword in (
                "business error",
                "not supported",
                "not found",
                "conflict",
                "already exists",
            )
        ):
            return FailureType.BUSINESS

        return FailureType.UNKNOWN