from __future__ import annotations

from typing import Any

from .context import RunContext
from .contract import Runtime
from .langgraph_runtime import LangGraphRuntime
from .result import RuntimeResult


class LangGraphRuntimeAdapter(Runtime):
    """
    Adapt LangGraphRuntime to the unified Runtime contract.
    """

    def __init__(
        self,
        runtime: LangGraphRuntime,
    ) -> None:
        self.runtime = runtime

    async def run(
        self,
        context: RunContext,
    ) -> RuntimeResult:
        try:
            input_data = {
                "messages": [
                    {
                        "role": "user",
                        "content": context.task,
                    }
                ],
                "context": context.context,
            }
            result = await self.runtime.run(
                run_id=context.run_id,
                input_data=input_data,
            )

            return self._to_runtime_result(result)

        except Exception as exc:
            return RuntimeResult.failed(
                str(exc),
            )

    async def resume(
        self,
        context: RunContext,
        value: Any,
    ) -> RuntimeResult:
        try:
            result = await self.runtime.resume(
                run_id=context.run_id,
                value=value,
            )

            return self._to_runtime_result(result)

        except Exception as exc:
            return RuntimeResult.failed(
                str(exc),
            )

    @property
    def supports_resume(self) -> bool:
        return True

    def _to_runtime_result(
        self,
        result: dict[str, Any],
    ) -> RuntimeResult:
        if self.runtime.is_interrupted(result):
            approval = self.runtime.extract_approval(
                result
            )

            approval_id = None

            if isinstance(approval, dict):
                value = approval.get("approval_id")

                if value is not None:
                    approval_id = str(value)

            return RuntimeResult.waiting_approval(
                approval_id=approval_id,
                result=result,
                metadata={
                    "approval": approval,
                },
            )

        return RuntimeResult.completed(
            result=self.runtime.extract_result(
                result
            ),
        )