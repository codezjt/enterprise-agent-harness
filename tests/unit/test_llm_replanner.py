import pytest
from langchain_core.language_models.fake_chat_models import FakeChatModel

from enterprise_harness.orchestration.llm_replanner import LLMReplanner
from enterprise_harness.orchestration.replanner import (
    ReplanRequest,
)
from enterprise_harness.orchestration.task import (
    Task,
    TaskStatus,
)
from enterprise_harness.orchestration.plan_validator import (
    PlanValidationError,
)


class FakeStructuredModel:
    def __init__(self, result):
        self.result = result

    async def ainvoke(self, messages):
        return self.result


class FakeModel(FakeChatModel):
    def __init__(self):
        super().__init__(responses=["ok"])

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        return "ok"


def build_request() -> ReplanRequest:
    failed_task = Task(
        task_id="T2",
        name="查询库存",
        description="查询当前库存",
        status=TaskStatus.FAILED,
        error="permission denied",
    )

    completed_task = Task(
        task_id="T1",
        name="查询订单",
        description="查询订单信息",
        status=TaskStatus.SUCCESS,
    )

    remaining_task = Task(
        task_id="T3",
        name="更新订单",
        description="更新订单库存信息",
        dependencies=["T2"],
        status=TaskStatus.PENDING,
    )

    return ReplanRequest(
        original_task="查询订单并更新库存",
        failed_task=failed_task,
        completed_tasks=[completed_task],
        remaining_tasks=[remaining_task],
    )


@pytest.mark.asyncio
async def test_llm_replanner_validates_result():
    replanner = LLMReplanner.__new__(LLMReplanner)

    from enterprise_harness.orchestration.replanner import ReplanResult

    result = ReplanResult(
        reason="库存查询接口权限失败，改用备用接口",
        tasks=[
            Task(
                task_id="T1",
                name="查询订单",
                description="查询订单信息",
                status=TaskStatus.SUCCESS,
            ),
            Task(
                task_id="T2_ALT",
                name="调用备用库存接口",
                description="通过备用库存接口查询库存",
                dependencies=["T1"],
            ),
            Task(
                task_id="T3",
                name="更新订单",
                description="根据库存结果更新订单",
                dependencies=["T2_ALT"],
            ),
        ],
    )

    from enterprise_harness.orchestration.plan_validator import (
        PlanValidator,
    )

    replanner.validator = PlanValidator()

    replanner._validate_result(result)


@pytest.mark.asyncio
async def test_llm_replanner_rejects_cyclic_plan():
    replanner = LLMReplanner.__new__(LLMReplanner)

    from enterprise_harness.orchestration.replanner import ReplanResult
    from enterprise_harness.orchestration.plan_validator import (
        PlanValidator,
    )

    replanner.validator = PlanValidator()

    result = ReplanResult(
        reason="invalid plan",
        tasks=[
            Task(
                task_id="T1",
                name="任务1",
                description="任务1",
                dependencies=["T2"],
            ),
            Task(
                task_id="T2",
                name="任务2",
                description="任务2",
                dependencies=["T1"],
            ),
        ],
    )

    with pytest.raises(PlanValidationError):
        replanner._validate_result(result)