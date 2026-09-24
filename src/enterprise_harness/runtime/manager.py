from datetime import datetime, timezone
from uuid import uuid4
from typing import Any

from enterprise_harness.agent import AgentRuntime
from .models import Run, RunStatus


class RunManager:
    """负责 Agent Run 的生命周期管理。"""

    def __init__(self):
        self.runs: dict[str, Run] = {}

    async def create_run(
        self,
        agent_id: str,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Run:
        run = Run(
            run_id=str(uuid4()),
            agent_id=agent_id,
            task=task,
            context=context or {},
        )

        self.runs[run.run_id] = run

        return run

    async def start_run(
        self,
        run_id: str,
        runtime: AgentRuntime,
    ) -> Run:
        run = self._get_run(run_id)

        run.status = RunStatus.RUNNING
        run.started_at = datetime.now(timezone.utc)

        try:
            result = await runtime.run(
                task=run.task,
                context=run.context,
            )

            run.result = result
            run.status = RunStatus.COMPLETED

        except Exception as exc:
            run.status = RunStatus.FAILED
            run.error = str(exc)

        finally:
            run.completed_at = datetime.now(timezone.utc)

        return run

    async def cancel_run(self, run_id: str) -> Run:
        run = self._get_run(run_id)

        run.status = RunStatus.CANCELLED
        run.completed_at = datetime.now(timezone.utc)

        return run

    async def get_run(self, run_id: str) -> Run:
        return self._get_run(run_id)

    def _get_run(self, run_id: str) -> Run:
        run = self.runs.get(run_id)

        if run is None:
            raise KeyError(f"Run not found: {run_id}")

        return run