from __future__ import annotations

from typing import Any

from enterprise_harness.observability import TraceManager

from .context import RunContext
from .context_factory import RunContextFactory
from .models import Run


class RunContextBuilder:

    def __init__(
        self,
        trace_manager: TraceManager,
    ):
        self.trace_manager = trace_manager

    def build(
        self,
        run: Run,
        *,
        principal=None,
        metadata: dict[str, Any] | None = None,
    ) -> RunContext:

        context = RunContextFactory.create(
            run,
            principal=principal,
            metadata=metadata,
        )

        root_span = self.trace_manager.start_run_span(
            run_id=run.run_id,
            agent_name=run.agent_id,
            tenant_id=run.tenant_id,
            trace_id=context.trace_id,
            input={
                "task": run.task,
                "context": run.context,
            },
            metadata={
                "agent_id": run.agent_id,
                "agent_version": run.agent_version,
                "tenant_id": run.tenant_id,
                "trace_id": context.trace_id,
            },
        )

        context.trace_root_span_id = root_span.span_id
        context.current_span_id = root_span.span_id

        return context