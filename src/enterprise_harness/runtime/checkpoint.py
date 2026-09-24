from typing import Any

from pydantic import BaseModel, Field


class Checkpoint(BaseModel):
    """Harness 层的运行时检查点。"""

    checkpoint_id: str
    run_id: str

    state: dict[str, Any] = Field(
        default_factory=dict
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class CheckpointStore:
    """Checkpoint 持久化抽象。

    第一阶段使用内存实现。
    后续可以替换成 Redis / DB / LangGraph Checkpointer。
    """

    def __init__(self):
        self._checkpoints: dict[str, Checkpoint] = {}

    async def save(
        self,
        checkpoint: Checkpoint,
    ) -> Checkpoint:
        self._checkpoints[
            checkpoint.checkpoint_id
        ] = checkpoint

        return checkpoint

    async def get(
        self,
        checkpoint_id: str,
    ) -> Checkpoint:
        checkpoint = self._checkpoints.get(
            checkpoint_id
        )

        if checkpoint is None:
            raise KeyError(
                f"Checkpoint not found: {checkpoint_id}"
            )

        return checkpoint

    async def delete(
        self,
        checkpoint_id: str,
    ) -> None:
        if checkpoint_id not in self._checkpoints:
            raise KeyError(
                f"Checkpoint not found: {checkpoint_id}"
            )

        del self._checkpoints[checkpoint_id]