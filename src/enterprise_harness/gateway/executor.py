import asyncio
import inspect
from typing import Any

from enterprise_harness.observability.metrics import MetricCollector

from .exceptions import ToolExecutionError, ToolTimeoutError
from .models import ToolDefinition

_RETRYABLE_ERROR_TYPES: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
    ConnectionRefusedError,
    ConnectionResetError,
    BrokenPipeError,
    OSError,
)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, ToolTimeoutError):
        return True
    if isinstance(exc, _RETRYABLE_ERROR_TYPES):
        return True
    if isinstance(exc, ToolExecutionError):
        cause = exc.__cause__
        if cause is not None:
            return _is_retryable(cause)
        return False
    return False


class ToolExecutor:

    def __init__(
        self,
        metric_collector: MetricCollector | None = None,
    ):
        self._default_timeout: float = 60.0
        self._metric_collector = metric_collector

    async def execute(
        self,
        tool: ToolDefinition,
        arguments: dict[str, Any],
    ) -> Any:
        if tool.handler is None:
            raise ToolExecutionError(
                tool.name,
                f"Tool handler is not configured: {tool.name}",
            )

        timeout = tool.timeout or self._default_timeout

        retry_policy = tool.retry_policy or {}
        max_retries = retry_policy.get("max_retries", 0)
        retry_delay = retry_policy.get("retry_delay", 1.0)

        last_exc: BaseException | None = None

        for attempt in range(max_retries + 1):
            try:
                coro = self._invoke_handler(tool, arguments)
                result = await asyncio.wait_for(coro, timeout=timeout)
                return result
            except asyncio.TimeoutError:
                last_exc = ToolTimeoutError(tool.name, timeout)
            except ToolExecutionError as exc:
                last_exc = exc
                break
            except Exception as exc:
                last_exc = exc

            if attempt < max_retries and _is_retryable(last_exc):
                if self._metric_collector is not None:
                    self._metric_collector.record_tool_retry()
                await asyncio.sleep(retry_delay)
                continue
            break

        if isinstance(last_exc, ToolTimeoutError):
            if max_retries > 0:
                raise ToolExecutionError(
                    tool.name,
                    f"Tool failed after {max_retries} retries: {last_exc}",
                ) from last_exc
            raise last_exc

        if isinstance(last_exc, ToolExecutionError):
            raise last_exc

        if isinstance(last_exc, _RETRYABLE_ERROR_TYPES) and max_retries > 0:
            raise ToolExecutionError(
                tool.name,
                f"Tool failed after {max_retries} retries: {last_exc}",
            ) from last_exc

        if last_exc is not None:
            raise ToolExecutionError(
                tool.name,
                f"Tool execution failed: {last_exc}",
            ) from last_exc

        raise ToolExecutionError(
            tool.name,
            "Tool execution failed for unknown reason",
        )

    async def _invoke_handler(
        self,
        tool: ToolDefinition,
        arguments: dict[str, Any],
    ) -> Any:
        result = tool.handler(**arguments)

        if inspect.isawaitable(result):
            return await result

        return result