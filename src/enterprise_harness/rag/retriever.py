from abc import ABC, abstractmethod
from typing import Any


class Retriever(ABC):

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[Any]:
        raise NotImplementedError