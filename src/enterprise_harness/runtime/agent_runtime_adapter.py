from __future__ import annotations

from typing import Any

from enterprise_harness.agent.runtime import AgentRuntime

from .contract import Runtime
from .result import RuntimeResult


class AgentRuntimeAdapter(Runtime):
    """
    Adapt the existing AgentRuntime to the unified Runtime contract.

    The existing AgentRuntime remains unchanged for backward compatibility.
    This adapter is the compatibility boundary used by the unified runtime
    layer.
    """

    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime

    async def run(
        self,
        context: Any,
    ) -> RuntimeResult:
        try:
            result = await self.runtime.run_with_context(
                context
            )

            if isinstance(result, RuntimeResult):
                return result

            return RuntimeResult.completed(
                result=result,
            )

        except Exception as exc:
            return RuntimeResult.failed(
                str(exc),
            )