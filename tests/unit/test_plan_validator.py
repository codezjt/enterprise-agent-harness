import pytest

from enterprise_harness.orchestration.plan_validator import (
    PlanValidationError,
    PlanValidator,
)
from enterprise_harness.orchestration.planner import Plan
from enterprise_harness.orchestration.task import Task


def create_valid_plan() -> Plan:
    return Plan(
        task="处理订单 1001",
        tasks=[
            Task(
                task_id="T1",
                name="查询订单",
                description="查询订单 1001",
            ),
            Task(
                task_id="T2",
                name="查询库存",
                description="查询库存 SKU-001",
            ),
            Task(
                task_id="T3",
                name="综合分析",
                description="综合订单和库存",
                dependencies=["T1", "T2"],
            ),
        ],
    )


def test_validate_valid_plan():
    validator = PlanValidator()

    graph = validator.validate(
        create_valid_plan()
    )

    assert len(graph.tasks()) == 3

    assert {
        task.task_id
        for task in graph.get_ready_tasks()
    } == {"T1", "T2"}


def test_reject_empty_plan_task():
    validator = PlanValidator()

    plan = Plan(
        task="",
        tasks=[
            Task(
                task_id="T1",
                name="执行任务",
            )
        ],
    )

    with pytest.raises(
        PlanValidationError,
        match="Plan task must not be empty",
    ):
        validator.validate(plan)


def test_reject_empty_tasks():
    validator = PlanValidator()

    plan = Plan(
        task="处理订单",
        tasks=[],
    )

    with pytest.raises(
        PlanValidationError,
        match="Plan must contain at least one task",
    ):
        validator.validate(plan)


def test_reject_duplicate_task_id():
    validator = PlanValidator()

    plan = Plan(
        task="处理订单",
        tasks=[
            Task(
                task_id="T1",
                name="查询订单",
            ),
            Task(
                task_id="T1",
                name="查询库存",
            ),
        ],
    )

    with pytest.raises(
        PlanValidationError,
        match="Duplicate task id: T1",
    ):
        validator.validate(plan)


def test_reject_unknown_dependency():
    validator = PlanValidator()

    plan = Plan(
        task="处理订单",
        tasks=[
            Task(
                task_id="T1",
                name="查询订单",
                dependencies=["UNKNOWN"],
            ),
        ],
    )

    with pytest.raises(
        PlanValidationError,
        match="depends on unknown task: UNKNOWN",
    ):
        validator.validate(plan)


def test_reject_cyclic_dependency():
    validator = PlanValidator()

    plan = Plan(
        task="处理订单",
        tasks=[
            Task(
                task_id="T1",
                name="任务一",
                dependencies=["T2"],
            ),
            Task(
                task_id="T2",
                name="任务二",
                dependencies=["T1"],
            ),
        ],
    )

    with pytest.raises(
        PlanValidationError,
        match="cycle",
    ):
        validator.validate(plan)