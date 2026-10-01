from typing import Any, TYPE_CHECKING

from enterprise_harness.memory import MemoryManager, MemoryScope
from enterprise_harness.rag import Retriever

if TYPE_CHECKING:
    from enterprise_harness.observability import TraceManager
    from enterprise_harness.observability.trace import SpanType


class ContextProvider:
    def __init__(
        self,
        memory_manager: MemoryManager | None = None,
        retriever: Retriever | None = None,
        rag_top_k: int = 5,
        trace_manager: "TraceManager | None" = None,
    ):
        self.memory_manager = memory_manager
        self.retriever = retriever
        self.rag_top_k = rag_top_k
        self.trace_manager = trace_manager

    async def get_memory(
        self,
        *,
        query: str,
        scope: MemoryScope | None = None,
        limit: int = 10,
    ) -> list[Any]:
        if self.memory_manager is None:
            return []

        return await self.memory_manager.recall(
            limit=limit,
            scope=scope,
        )

    async def get_rag(
        self,
        *,
        query: str,
    ) -> list[Any]:
        if self.retriever is None:
            return []

        return await self.retriever.retrieve(
            query=query,
            top_k=self.rag_top_k,
        )

    async def store_memory(
        self,
        item: Any,
        *,
        scope: MemoryScope | None = None,
        long_term: bool = False,
    ) -> None:
        if self.memory_manager is None:
            return

        await self.memory_manager.store(
            item,
            long_term=long_term,
            scope=scope,
        )

    async def enrich_run_context(
        self,
        run_context: Any,
    ) -> None:
        from enterprise_harness.runtime.context import RunContext
        from enterprise_harness.observability import SpanStatus, SpanType

        scope = MemoryScope(
            tenant_id=run_context.tenant_id if isinstance(run_context, RunContext) else "default",
            agent_id=run_context.agent_id if isinstance(run_context, RunContext) else "",
            run_id=run_context.run_id if isinstance(run_context, RunContext) else "",
        )

        memory_span = None
        rag_span = None

        if self.trace_manager is not None and isinstance(run_context, RunContext):
            memory_span = self.trace_manager.start_span(
                run_id=run_context.run_id,
                span_type=SpanType.MEMORY,
                name="memory_recall",
                parent_span_id=run_context.current_span_id,
            )

        try:
            memory_results = await self.get_memory(
                query=run_context.task,
                scope=scope,
            )
            if memory_span is not None:
                self.trace_manager.finish_span(
                    memory_span.span_id,
                    output={"count": len(memory_results)},
                    status=SpanStatus.SUCCESS,
                )
        except Exception as exc:
            if memory_span is not None:
                self.trace_manager.fail_span(memory_span.span_id, exc)
            memory_results = []

        if self.trace_manager is not None and isinstance(run_context, RunContext):
            rag_span = self.trace_manager.start_span(
                run_id=run_context.run_id,
                span_type=SpanType.RAG,
                name="rag_retrieve",
                parent_span_id=run_context.current_span_id,
            )

        try:
            rag_results = await self.get_rag(
                query=run_context.task,
            )
            if rag_span is not None:
                self.trace_manager.finish_span(
                    rag_span.span_id,
                    output={"count": len(rag_results)},
                    status=SpanStatus.SUCCESS,
                )
        except Exception as exc:
            if rag_span is not None:
                self.trace_manager.fail_span(rag_span.span_id, exc)
            rag_results = []

        run_context.context["memory_results"] = memory_results
        run_context.context["rag_results"] = rag_results