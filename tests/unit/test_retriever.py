import pytest

from enterprise_harness.rag import Retriever


class FakeRetriever(Retriever):

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[str]:
        results = [
            "订单 1001 的状态为已发货",
            "订单 1002 的状态为待付款",
            "订单 1003 的状态为已完成",
        ]
        return results[:top_k]

@pytest.mark.asyncio
async def test_retriever():
    retriever = FakeRetriever()

    results = await retriever.retrieve(
        query="查询订单状态",
        top_k=2,
    )

    assert results == [
        "订单 1001 的状态为已发货",
        "订单 1002 的状态为待付款",
    ]