from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from enterprise_harness.agent.model import resolve_model

from .planner import Plan, Planner


class LLMPlanner(Planner):
    """
    基于 LangChain Chat Model 的 Planner。

    LLM 只负责生成结构化 Plan，
    不负责执行 Task。
    """

    def __init__(
        self,
        model: str | BaseChatModel,
    ) -> None:
        resolved_model = resolve_model(model)

        if not isinstance(resolved_model, BaseChatModel):
            raise TypeError(
                "LLMPlanner requires a chat model"
            )

        self.model = resolved_model

        self.structured_model = (
            self.model.with_structured_output(Plan)
        )

    async def plan(self, task: str) -> Plan:
        messages = [
            (
                "system",
                self.system_prompt(),
            ),
            (
                "human",
                task,
            ),
        ]

        result = await self.structured_model.ainvoke(
            messages
        )

        if isinstance(result, Plan):
            return result

        return Plan.model_validate(result)

    @staticmethod
    def system_prompt() -> str:
        return """
你是企业级 Agent Harness 的任务规划器。

你的职责是把用户任务拆分成可执行的任务计划。

要求：

1. 为每个任务生成唯一 task_id。
2. 每个任务必须具有清晰的 name。
3. 每个任务必须具有具体的 description。
4. 使用 dependencies 表示任务之间的依赖关系。
5. 没有依赖关系的任务可以并行执行。
6. 只有依赖任务完成后，当前任务才能执行。
7. 不要执行任务。
8. 不要调用工具。
9. 不要生成任务之外的解释。

任务应该尽可能保持独立，并能够由 Agent Runtime 单独执行。

最终只输出符合 Plan 数据结构的结构化结果。
""".strip()