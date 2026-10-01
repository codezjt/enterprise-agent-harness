import pytest

from enterprise_harness.context import (
    ContextBuilder,
    ContextItem,
    ContextProvider,
    TokenBudget,
)
from enterprise_harness.memory import (
    MemoryManager,
    MemoryScope,
    ScopedShortTermMemory,
)
from enterprise_harness.rag import (
    Document,
    InMemoryRetriever,
    RetrievalResult,
)


class TestInMemoryRetriever:

    def test_add_and_retrieve_documents(self):
        retriever = InMemoryRetriever()
        retriever.add_document(
            Document(
                doc_id="doc-1",
                title="Python Guide",
                content="Python is a programming language. It supports async programming with asyncio.",
            )
        )
        retriever.add_document(
            Document(
                doc_id="doc-2",
                title="Java Guide",
                content="Java is a statically typed language. It uses Spring Boot for web apps.",
            )
        )

        assert retriever.document_count == 2

    @pytest.mark.asyncio
    async def test_retrieve_by_query(self):
        retriever = InMemoryRetriever()
        retriever.add_document(
            Document(
                doc_id="doc-1",
                title="Async Programming",
                content="Python asyncio provides async/await syntax for concurrent programming.",
            )
        )
        retriever.add_document(
            Document(
                doc_id="doc-2",
                title="Web Frameworks",
                content="FastAPI is an async web framework built on Starlette.",
            )
        )

        results = await retriever.retrieve(query="async programming", top_k=5)

        assert len(results) >= 1
        assert any("asyncio" in r.content for r in results)
        assert all(isinstance(r, RetrievalResult) for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_no_match(self):
        retriever = InMemoryRetriever()
        retriever.add_document(
            Document(
                doc_id="doc-1",
                title="Python",
                content="Python is a programming language.",
            )
        )

        results = await retriever.retrieve(query="xyzabc123", top_k=5)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_retrieve_with_score(self):
        retriever = InMemoryRetriever()
        retriever.add_document(
            Document(
                doc_id="doc-1",
                title="Test",
                content="Machine learning is a subset of artificial intelligence.",
            )
        )

        results = await retriever.retrieve(query="machine learning", top_k=5)
        assert len(results) >= 1
        assert results[0].score > 0
        assert results[0].source == "Test"

    def test_index_documents(self):
        retriever = InMemoryRetriever()
        docs = [
            Document(doc_id="d1", title="A", content="Content A."),
            Document(doc_id="d2", title="B", content="Content B."),
        ]
        retriever.index_documents(docs)
        assert retriever.document_count == 2
        assert retriever.chunk_count >= 2


class TestScopedShortTermMemory:

    def test_store_and_recall_with_scope(self):
        mem = ScopedShortTermMemory()

        scope_a = MemoryScope(tenant_id="t1", user_id="u1")
        scope_b = MemoryScope(tenant_id="t2", user_id="u2")

        mem.store({"key": "a1"}, scope=scope_a)
        mem.store({"key": "b1"}, scope=scope_b)
        mem.store({"key": "a2"}, scope=scope_a)

        results_a = mem.recall(limit=10, scope=scope_a)
        assert len(results_a) == 2
        assert all(r["key"].startswith("a") for r in results_a)

        results_b = mem.recall(limit=10, scope=scope_b)
        assert len(results_b) == 1
        assert results_b[0]["key"] == "b1"

    def test_recall_without_scope_returns_all(self):
        mem = ScopedShortTermMemory()
        scope_a = MemoryScope(tenant_id="t1")
        scope_b = MemoryScope(tenant_id="t2")

        mem.store("a", scope=scope_a)
        mem.store("b", scope=scope_b)

        results = mem.recall(limit=10)
        assert len(results) == 2

    def test_recall_with_limit(self):
        mem = ScopedShortTermMemory()
        for i in range(5):
            mem.store(f"item-{i}")

        results = mem.recall(limit=3)
        assert len(results) == 3
        assert results == ["item-2", "item-3", "item-4"]

    def test_clear_scope(self):
        mem = ScopedShortTermMemory()
        scope_a = MemoryScope(tenant_id="t1")
        scope_b = MemoryScope(tenant_id="t2")

        mem.store("a1", scope=scope_a)
        mem.store("b1", scope=scope_b)

        mem.clear(scope=scope_a)

        results_a = mem.recall(scope=scope_a)
        assert len(results_a) == 0

        results_b = mem.recall(scope=scope_b)
        assert len(results_b) == 1

    def test_clear_all(self):
        mem = ScopedShortTermMemory()
        mem.store("a")
        mem.store("b")
        mem.clear()
        assert len(mem.recall(limit=10)) == 0


class TestContextProviderWithScope:

    @pytest.mark.asyncio
    async def test_enrich_run_context_populates_memory_and_rag(self):
        from enterprise_harness.runtime.context import RunContext

        retriever = InMemoryRetriever()
        retriever.add_document(
            Document(
                doc_id="doc-1",
                title="API Reference",
                content="The Chat API supports streaming responses. Use the stream parameter.",
            )
        )

        memory_manager = MemoryManager(
            short_term=ScopedShortTermMemory()
        )

        await memory_manager.store(
            {"role": "user", "content": "previous question"},
            scope=MemoryScope(tenant_id="t1", agent_id="agent-1"),
        )

        provider = ContextProvider(
            memory_manager=memory_manager,
            retriever=retriever,
        )

        run_ctx = RunContext(
            run_id="run-001",
            agent_id="agent-1",
            tenant_id="t1",
            task="How does the Chat API work?",
        )

        await provider.enrich_run_context(run_ctx)

        assert "memory_results" in run_ctx.context
        assert "rag_results" in run_ctx.context

        rag_results = run_ctx.context["rag_results"]
        assert len(rag_results) >= 1
        assert any("Chat API" in r.content for r in rag_results)


class TestContextBuilderIntegration:

    def test_context_builder_with_all_sources(self):
        builder = ContextBuilder()

        items = builder.build(
            system_policy="Be helpful",
            task="What is Python?",
            tool_results=["result1"],
            rag_results=["rag1", "rag2"],
            memory_results=["mem1"],
            conversation=["Hello", "Hi there"],
        )

        sources = {item.source for item in items}
        assert "system_policy" in sources
        assert "task" in sources
        assert "tool" in sources
        assert "rag" in sources
        assert "memory" in sources
        assert "conversation" in sources

    def test_context_builder_applies_budget(self):
        builder = ContextBuilder(token_budget=TokenBudget(max_tokens=50))

        items = builder.build(
            system_policy="This is a system policy that is very important.",
            task="Solve the problem.",
            rag_results=["A very long RAG result that takes a lot of space."],
        )

        total_tokens = sum(
            len(str(item.content)) for item in items
        )
        assert total_tokens <= 50

    def test_build_from_run_context_with_enriched_data(self):
        from enterprise_harness.runtime.context import RunContext

        builder = ContextBuilder(
            token_budget=TokenBudget(max_tokens=500)
        )

        run_ctx = RunContext(
            run_id="run-001",
            agent_id="agent-1",
            tenant_id="t1",
            task="Analyze data",
        )
        run_ctx.context["memory_results"] = [
            {"type": "memory", "content": "past interaction"}
        ]
        run_ctx.context["rag_results"] = [
            RetrievalResult(
                content="Data analysis techniques",
                score=0.9,
                source="docs",
            )
        ]
        run_ctx.context["system_policy"] = "Be accurate"
        run_ctx.context["conversation"] = [{"role": "user", "content": "Hi"}]

        built = builder.build_from_run_context(run_ctx)
        assert len(built) >= 3

        sources = {item.source for item in built}
        assert "system_policy" in sources
        assert "task" in sources
        assert "memory" in sources
        assert "rag" in sources


class TestTokenBudgetCompression:

    def test_fit_prioritizes_high_priority(self):
        budget = TokenBudget(max_tokens=60)

        items = [
            ContextItem(priority=0, source="system", content="SYS:" + "x" * 20),
            ContextItem(priority=1, source="task", content="TASK:" + "x" * 20),
            ContextItem(priority=5, source="conv", content="CONV:" + "x" * 20),
        ]

        result = budget.fit(items)
        sources = {item.source for item in result}
        assert "system" in sources
        assert "task" in sources

    def test_truncate_large_item(self):
        budget = TokenBudget(max_tokens=50)

        items = [
            ContextItem(priority=0, source="system", content="SYS"),
            ContextItem(priority=2, source="tool", content="TOOL:" + "x" * 200),
        ]

        result = budget.fit(items)
        assert len(result) == 1

    def test_fit_with_compression_fallback(self):
        budget = TokenBudget(max_tokens=60)

        items = [
            ContextItem(priority=0, source="system", content="SYS:" + "x" * 20),
            ContextItem(priority=1, source="task", content="TASK:" + "x" * 20),
            ContextItem(priority=3, source="rag", content="RAG:" + "x" * 200),
        ]

        result = budget.fit_with_compression(items)
        assert len(result) >= 2