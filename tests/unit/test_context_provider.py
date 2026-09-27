import pytest

from enterprise_harness.context import ContextProvider
from enterprise_harness.memory import MemoryManager
from enterprise_harness.rag import Retriever


class FakeRetriever(Retriever):

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[str]:
        return [
            f"RAG result for: {query}",
        ][:top_k]

@pytest.mark.asyncio
async def test_context_provider_memory():
    memory = MemoryManager()

    await memory.store("历史订单信息")

    provider = ContextProvider(
        memory_manager=memory,
    )

    result = await provider.get_memory(
        query="查询订单",
    )

    assert result == ["历史订单信息"]

@pytest.mark.asyncio
async def test_context_provider_rag():
    provider = ContextProvider(
        retriever=FakeRetriever(),
        rag_top_k=3,
    )

    result = await provider.get_rag(
        query="订单处理规范",
    )

    assert result == [
        "RAG result for: 订单处理规范",
    ]