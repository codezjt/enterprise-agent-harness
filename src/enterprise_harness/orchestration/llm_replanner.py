from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from enterprise_harness.agent.model import resolve_model

from .plan_validator import PlanValidator
from .replanner import ReplanRequest, ReplanResult, Replanner
from .task_graph import TaskGraph


class LLMReplanner(Replanner):
    """
    基于 LLM 的任务重新规划器。

    职责：
    1. 接收失败任务、已完成任务、剩余任务以及失败原因。
    2. 调用 LLM 生成新的 ReplanResult。
    3. 对 LLM 生成的任务计划进行结构校验。
    4. 将合法结果转换为 TaskGraph。

    注意：
    LLMReplanner 不负责执行任务。
    实际执行仍然由 Scheduler 完成。
    """

    def __init__(
        self,
        model: str | BaseChatModel,
        validator: PlanValidator | None = None,
    ) -> None:
        resolved_model = resolve_model(model)

        if not isinstance(resolved_model, BaseChatModel):
            raise TypeError("LLMReplanner requires a chat model")

        self.model = resolved_model
        self.structured_model = self.model.with_structured_output(
            ReplanResult
        )
        self.validator = validator or PlanValidator()

    async def replan(
        self,
        request: ReplanRequest,
    ) -> ReplanResult:
        messages = [
            (
                "system",
                self.system_prompt(),
            ),
            (
                "human",
                self._build_request_prompt(request),
            ),
        ]

        result = await self.structured_model.ainvoke(messages)

        if not isinstance(result, ReplanResult):
            result = ReplanResult.model_validate(result)

        self._validate_result(result)

        return result

    def to_task_graph(
        self,
        result: ReplanResult,
    ) -> TaskGraph:
        """
        将经过校验的 ReplanResult 转换为 TaskGraph。
        """
        self._validate_result(result)
        return result.to_task_graph()

    def _validate_result(
        self,
        result: ReplanResult,
    ) -> None:
        """
        复用 PlanValidator 的校验能力。

        ReplanResult 与 Plan 都最终描述一个 TaskGraph，
        因此这里把 ReplanResult 转换成 Plan 再复用已有校验逻辑。
        """
        from .planner import Plan

        plan = Plan(
            task=result.reason or "replan",
            tasks=result.tasks,
        )

        self.validator.validate(plan)

    @staticmethod
    def system_prompt() -> str:
        return """
你是企业级 Agent Harness 的任务重新规划器。

你的职责是在任务执行失败后，根据当前执行状态重新生成可执行任务计划。

你会获得：

1. 用户原始任务。
2. 执行失败的任务。
3. 已经成功完成的任务。
4. 尚未完成的任务。
5. 失败原因。

你的目标不是重复执行失败任务，而是分析失败原因，
在必要时选择替代任务、替代执行路径或调整任务依赖关系。

要求：

1. 每个任务必须生成唯一 task_id。
2. 每个任务必须具有清晰的 name。
3. 每个任务必须具有具体的 description。
4. 使用 dependencies 表示任务之间的依赖关系。
5. 已经成功完成的任务如果仍然是后续任务的前置条件，应保留。
6. 不应重复执行已经成功完成且没有必要重新执行的任务。
7. 对失败任务，如果存在替代方案，应优先生成替代任务。
8. 如果失败原因是暂时性问题且仍适合重试，可以重新生成对应任务。
9. 新任务必须能够形成合法的 DAG。
10. 不允许出现循环依赖。
11. 不要执行任务。
12. 不要调用工具。
13. 最终只输出符合 ReplanResult 数据结构的结构化结果。

重新规划必须围绕原始任务目标，
不能因为某一个子任务失败而改变用户最终目标。
""".strip()

    @staticmethod
    def _build_request_prompt(
        request: ReplanRequest,
    ) -> str:
        def task_text(task) -> str:
            return (
                f"- task_id: {task.task_id}\n"
                f"  name: {task.name}\n"
                f"  description: {task.description}\n"
                f"  status: {task.status.value}\n"
                f"  dependencies: {task.dependencies}\n"
                f"  retry_count: {task.retry_count}\n"
                f"  max_retries: {task.max_retries}\n"
                f"  error: {task.error}"
            )

        completed = "\n".join(
            task_text(task)
            for task in request.completed_tasks
        ) or "无"

        remaining = "\n".join(
            task_text(task)
            for task in request.remaining_tasks
        ) or "无"

        return f"""
用户原始任务：

{request.original_task}

失败任务：

{task_text(request.failed_task)}

已经成功完成的任务：

{completed}

尚未完成的任务：

{remaining}

请根据以上信息生成新的任务计划。
""".strip()