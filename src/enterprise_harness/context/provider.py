from typing import Any

from enterprise_harness.memory import MemoryManager
from enterprise_harness.rag import Retriever


class ContextProvider:
    def __init__(
        self,
        memory_manager: MemoryManager | None = None,
        retriever: Retriever | None = None,
        rag_top_k: int = 5,
    ):
        self.memory_manager = memory_manager
        self.retriever = retriever
        self.rag_top_k = rag_top_k

    async def get_memory(
        self,
        *,
        query: str,
    ) -> list[Any]:
        if self.memory_manager is None:
            return []

        return await self.memory_manager.recall()

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