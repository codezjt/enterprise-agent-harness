from __future__ import annotations

from typing import Any

from enterprise_harness.policy.rbac import Principal

from .context import RunContext
from .models import Run


class RunContextFactory:

    @staticmethod
    def create(
        run: Run,
        *,
        principal: Principal | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunContext:

        return RunContext(
            run_id=run.run_id,
            agent_id=run.agent_id,
            task=run.task,
            context=dict(run.context),
            principal=principal,
            metadata=metadata or {},
        )