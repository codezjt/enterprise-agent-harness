import pytest

from enterprise_harness.agent import AgentConfig
from enterprise_harness.agent.deepagent_runtime import DeepAgentRuntime
from enterprise_harness.context import (
    ContextBuilder,
    ContextProvider,
)
from enterprise_harness.memory import MemoryManager
from enterprise_harness.rag import Retriever
from enterprise_harness.runtime.context import RunContext


class FakeRetriever(Retriever):

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[str]:
        return [f"RAG: {query}"]

@pytest.mark.asyncio
async def test_runtime_context_provider():
    memory = MemoryManager()

    await memory.store("历史订单：1001")

    provider = ContextProvider(
        memory_manager=memory,
        retriever=FakeRetriever(),
    )

    config = AgentConfig(
        agent_id="test-agent",
        name="Test Agent",
        model="test-model",
    )

    runtime = DeepAgentRuntime(
        config=config,
        context_builder=ContextBuilder(),
        context_provider=provider,
    )

    run_context = RunContext(
        run_id="run-1",
        agent_id="test-agent",
        task="分析订单 1001",
    )

    memory_results = await runtime.context_provider.get_memory(
        query=run_context.task,
    )

    rag_results = await runtime.context_provider.get_rag(
        query=run_context.task,
    )

    run_context.context["memory_results"] = memory_results
    run_context.context["rag_results"] = rag_results

    items = runtime.context_builder.build_from_run_context(
        run_context
    )

    assert items[0].source == "task"

    assert any(
        item.source == "memory"
        and item.content == "历史订单：1001"
        for item in items
    )

    assert any(
        item.source == "rag"
        and item.content == "RAG: 分析订单 1001"
        for item in items
    )