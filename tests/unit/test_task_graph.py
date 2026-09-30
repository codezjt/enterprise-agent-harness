from __future__ import annotations

import pytest

from enterprise_harness.orchestration import (
    Task,
    TaskGraph,
    TaskStatus,
)


def create_task(
    task_id: str,
    name: str | None = None,
    dependencies: list[str] | None = None,
) -> Task:
    return Task(
        task_id=task_id,
        name=name or task_id,
        dependencies=dependencies or [],
    )


def test_add_task():
    graph = TaskGraph()

    task = create_task("task-1", "Query Order")

    graph.add_task(task)

    assert graph.get_task("task-1") is task
    assert graph.tasks["task-1"].name == "Query Order"


def test_add_duplicate_task_rejected():
    graph = TaskGraph()

    graph.add_task(
        create_task("task-1")
    )

    with pytest.raises(ValueError, match="Task already exists"):
        graph.add_task(
            create_task("task-1")
        )


def test_add_dependency():
    graph = TaskGraph()

    graph.add_task(create_task("task-1"))
    graph.add_task(create_task("task-2"))

    graph.add_dependency(
        task_id="task-2",
        dependency_id="task-1",
    )

    assert graph.get_task("task-2").dependencies == [
        "task-1"
    ]


def test_add_dependency_unknown_task_rejected():
    graph = TaskGraph()

    graph.add_task(create_task("task-1"))

    with pytest.raises(
        KeyError,
        match="Dependency task not found",
    ):
        graph.add_dependency(
            task_id="task-1",
            dependency_id="task-999",
        )


def test_self_dependency_rejected():
    graph = TaskGraph()

    graph.add_task(create_task("task-1"))

    with pytest.raises(
        ValueError,
        match="cannot depend on itself",
    ):
        graph.add_dependency(
            task_id="task-1",
            dependency_id="task-1",
        )


def test_duplicate_dependency_is_ignored():
    graph = TaskGraph()

    graph.add_task(create_task("task-1"))
    graph.add_task(create_task("task-2"))

    graph.add_dependency("task-2", "task-1")
    graph.add_dependency("task-2", "task-1")

    assert graph.get_task("task-2").dependencies == [
        "task-1"
    ]


def test_validate_simple_dag():
    graph = TaskGraph()

    graph.add_task(create_task("task-1"))
    graph.add_task(
        create_task(
            "task-2",
            dependencies=["task-1"],
        )
    )
    graph.add_task(
        create_task(
            "task-3",
            dependencies=["task-2"],
        )
    )

    graph.validate()


def test_validate_unknown_dependency_rejected():
    graph = TaskGraph()

    graph.add_task(
        create_task(
            "task-1",
            dependencies=["task-999"],
        )
    )

    with pytest.raises(
        ValueError,
        match="depends on unknown task",
    ):
        graph.validate()


def test_validate_self_dependency_rejected():
    graph = TaskGraph()

    graph.add_task(
        create_task(
            "task-1",
            dependencies=["task-1"],
        )
    )

    with pytest.raises(
        ValueError,
        match="cannot depend on itself",
    ):
        graph.validate()


def test_validate_cycle_rejected():
    graph = TaskGraph()

    graph.add_task(
        create_task(
            "task-1",
            dependencies=["task-3"],
        )
    )
    graph.add_task(
        create_task(
            "task-2",
            dependencies=["task-1"],
        )
    )
    graph.add_task(
        create_task(
            "task-3",
            dependencies=["task-2"],
        )
    )

    with pytest.raises(
        ValueError,
        match="contains a cycle",
    ):
        graph.validate()


def test_ready_tasks_initial_parallel_tasks():
    graph = TaskGraph()

    graph.add_task(create_task("task-1"))
    graph.add_task(create_task("task-2"))

    graph.add_task(
        create_task(
            "task-3",
            dependencies=["task-1", "task-2"],
        )
    )

    graph.validate()

    ready = graph.get_ready_tasks()

    assert [task.task_id for task in ready] == [
        "task-1",
        "task-2",
    ]


def test_ready_tasks_after_first_dependency_completed():
    graph = TaskGraph()

    graph.add_task(create_task("task-1"))
    graph.add_task(create_task("task-2"))

    graph.add_task(
        create_task(
            "task-3",
            dependencies=["task-1", "task-2"],
        )
    )

    graph.mark_running("task-1")
    graph.mark_completed(
        "task-1",
        result="result-1",
    )

    ready = graph.get_ready_tasks()

    assert [task.task_id for task in ready] == [
        "task-2"
    ]


def test_ready_tasks_after_all_dependencies_completed():
    graph = TaskGraph()

    graph.add_task(create_task("task-1"))
    graph.add_task(create_task("task-2"))

    graph.add_task(
        create_task(
            "task-3",
            dependencies=["task-1", "task-2"],
        )
    )

    graph.mark_running("task-1")
    graph.mark_completed(
        "task-1",
        result="result-1",
    )

    graph.mark_running("task-2")
    graph.mark_completed(
        "task-2",
        result="result-2",
    )

    ready = graph.get_ready_tasks()

    assert [task.task_id for task in ready] == [
        "task-3"
    ]


def test_completed_task_is_not_ready():
    graph = TaskGraph()

    graph.add_task(create_task("task-1"))

    graph.mark_running("task-1")
    graph.mark_completed(
        "task-1",
        result="completed",
    )

    assert graph.get_ready_tasks() == []


def test_running_task_is_not_ready():
    graph = TaskGraph()

    graph.add_task(create_task("task-1"))

    graph.mark_running("task-1")

    assert graph.get_ready_tasks() == []


def test_failed_dependency_blocks_dependent_task():
    graph = TaskGraph()

    graph.add_task(create_task("task-1"))

    graph.add_task(
        create_task(
            "task-2",
            dependencies=["task-1"],
        )
    )

    graph.mark_running("task-1")
    graph.mark_failed(
        "task-1",
        error="query failed",
    )

    assert graph.get_ready_tasks() == []


def test_mark_running_requires_pending():
    graph = TaskGraph()

    graph.add_task(create_task("task-1"))

    graph.mark_running("task-1")

    with pytest.raises(
        ValueError,
        match="cannot transition to RUNNING",
    ):
        graph.mark_running("task-1")


def test_mark_completed_requires_running():
    graph = TaskGraph()

    graph.add_task(create_task("task-1"))

    with pytest.raises(
        ValueError,
        match="cannot transition to COMPLETED",
    ):
        graph.mark_completed(
            "task-1",
            result="completed",
        )


def test_mark_failed_requires_running():
    graph = TaskGraph()

    graph.add_task(create_task("task-1"))

    with pytest.raises(
        ValueError,
        match="cannot transition to FAILED",
    ):
        graph.mark_failed(
            "task-1",
            error="failed",
        )


def test_task_result_and_error_are_updated():
    graph = TaskGraph()

    graph.add_task(create_task("task-1"))

    graph.mark_running("task-1")
    graph.mark_completed(
        "task-1",
        result={
            "order_id": "1001",
            "updated": True,
        },
    )

    task = graph.get_task("task-1")

    assert task.status == TaskStatus.COMPLETED
    assert task.result == {
        "order_id": "1001",
        "updated": True,
    }
    assert task.error is None


def test_mark_failed_sets_error():
    graph = TaskGraph()

    graph.add_task(create_task("task-1"))

    graph.mark_running("task-1")
    graph.mark_failed(
        "task-1",
        error="tool execution failed",
    )

    task = graph.get_task("task-1")

    assert task.status == TaskStatus.FAILED
    assert task.error == "tool execution failed"


def test_diamond_graph():
    """
    验证典型 DAG：

          task-1
         /      \
      task-2   task-3
         \      /
          task-4
    """

    graph = TaskGraph()

    graph.add_task(create_task("task-1"))

    graph.add_task(
        create_task(
            "task-2",
            dependencies=["task-1"],
        )
    )

    graph.add_task(
        create_task(
            "task-3",
            dependencies=["task-1"],
        )
    )

    graph.add_task(
        create_task(
            "task-4",
            dependencies=[
                "task-2",
                "task-3",
            ],
        )
    )

    graph.validate()

    ready = graph.get_ready_tasks()

    assert [task.task_id for task in ready] == [
        "task-1"
    ]

    graph.mark_running("task-1")
    graph.mark_completed(
        "task-1",
        result="result-1",
    )

    ready = graph.get_ready_tasks()

    assert [task.task_id for task in ready] == [
        "task-2",
        "task-3",
    ]

    graph.mark_running("task-2")
    graph.mark_completed(
        "task-2",
        result="result-2",
    )

    ready = graph.get_ready_tasks()

    assert [task.task_id for task in ready] == [
        "task-3"
    ]

    graph.mark_running("task-3")
    graph.mark_completed(
        "task-3",
        result="result-3",
    )

    ready = graph.get_ready_tasks()

    assert [task.task_id for task in ready] == [
        "task-4"
    ]