import pytest

from enterprise_harness.memory import (
    LongTermMemory,
    MemoryManager,
    ShortTermMemory,
)


def test_short_term_memory():
    memory = ShortTermMemory()

    memory.store("a")
    memory.store("b")

    assert memory.recall() == ["a", "b"]
    assert memory.recall(limit=1) == ["b"]

def test_long_term_memory():
    memory = LongTermMemory()

    memory.store("user preference")

    assert memory.recall() == ["user preference"]

@pytest.mark.asyncio
async def test_memory_manager():
    manager = MemoryManager()

    await manager.store("short")

    await manager.store(
        "long",
        long_term=True,
    )

    assert await manager.recall() == ["short"]

    assert await manager.recall(
        long_term=True,
    ) == ["long"]

@pytest.mark.asyncio
async def test_clear_short_term():
    manager = MemoryManager()

    await manager.store("a")
    manager.clear_short_term()

    assert await manager.recall() == []