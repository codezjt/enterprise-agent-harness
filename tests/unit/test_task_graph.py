import pytest

from enterprise_harness.orchestration.task import (
    Task,
    TaskStatus,
)
from enterprise_harness.orchestration.task_graph import TaskGraph


def build_graph() -> TaskGraph:
    graph = TaskGraph()

    graph.add_task(
        Task(
            task_id="T1",
            name="查询订单",
        )
    )

    graph.add_task(
        Task(
            task_id="T2",
            name="查询库存",
        )
    )

    graph.add_task(
        Task(
            task_id="T3",
            name="查询物流",
        )
    )

    graph.add_task(
        Task(
            task_id="T4",
            name="综合分析",
        )
    )

    graph.add_task(
        Task(
            task_id="T5",
            name="更新订单",
        )
    )

    graph.add_dependency("T4", "T1")
    graph.add_dependency("T4", "T2")
    graph.add_dependency("T4", "T3")

    graph.add_dependency("T5", "T4")

    return graph


def test_initial_ready_tasks():
    graph = build_graph()

    ready_tasks = graph.get_ready_tasks()

    assert {task.task_id for task in ready_tasks} == {
        "T1",
        "T2",
        "T3",
    }


def test_dependency_unlocks_task():
    graph = build_graph()

    graph.mark_success("T1", {"order_id": "1001"})
    graph.mark_success("T2", {"stock": 10})
    graph.mark_success("T3", {"status": "SHIPPED"})

    ready_tasks = graph.get_ready_tasks()

    assert [task.task_id for task in ready_tasks] == ["T4"]


def test_next_task_after_analysis():
    graph = build_graph()

    graph.mark_success("T1")
    graph.mark_success("T2")
    graph.mark_success("T3")

    graph.mark_success(
        "T4",
        output={"decision": "UPDATE"},
    )

    ready_tasks = graph.get_ready_tasks()

    assert [task.task_id for task in ready_tasks] == ["T5"]


def test_graph_completion():
    graph = build_graph()

    for task_id in ["T1", "T2", "T3", "T4", "T5"]:
        graph.mark_success(task_id)

    assert graph.is_completed() is True


def test_duplicate_task_rejected():
    graph = TaskGraph()

    graph.add_task(
        Task(
            task_id="T1",
            name="任务1",
        )
    )

    with pytest.raises(ValueError, match="already exists"):
        graph.add_task(
            Task(
                task_id="T1",
                name="任务1",
            )
        )


def test_self_dependency_rejected():
    graph = TaskGraph()

    graph.add_task(
        Task(
            task_id="T1",
            name="任务1",
        )
    )

    with pytest.raises(ValueError, match="cannot depend on itself"):
        graph.add_dependency("T1", "T1")


def test_cycle_dependency_rejected():
    graph = TaskGraph()

    graph.add_task(Task(task_id="T1", name="任务1"))
    graph.add_task(Task(task_id="T2", name="任务2"))
    graph.add_task(Task(task_id="T3", name="任务3"))

    graph.add_dependency("T2", "T1")
    graph.add_dependency("T3", "T2")

    with pytest.raises(ValueError, match="contains a cycle"):
        graph.add_dependency("T1", "T3")


def test_failed_task():
    graph = build_graph()

    graph.mark_failed(
        "T1",
        "query order failed",
    )

    assert graph.get_task("T1").status == TaskStatus.FAILED
    assert graph.has_failed() is True