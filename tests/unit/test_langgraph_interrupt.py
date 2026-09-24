from typing import TypedDict

import pytest

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ApprovalState(TypedDict):
    message: str
    approved: bool | None
    result: str | None


def approval_node(state: ApprovalState):
    approved = interrupt(
        {
            "type": "approval",
            "message": state["message"],
        }
    )

    if approved:
        return {
            "approved": True,
            "result": "approved",
        }

    return {
        "approved": False,
        "result": "rejected",
    }


def build_graph():
    graph = StateGraph(ApprovalState)

    graph.add_node(
        "approval",
        approval_node,
    )

    graph.add_edge(
        START,
        "approval",
    )

    graph.add_edge(
        "approval",
        END,
    )

    return graph.compile(
        checkpointer=InMemorySaver(),
    )


@pytest.mark.asyncio
async def test_interrupt_and_resume():
    graph = build_graph()

    config = {
        "configurable": {
            "thread_id": "run-001",
        }
    }

    result = await graph.ainvoke(
        {
            "message": "是否执行订单更新？",
            "approved": None,
            "result": None,
        },
        config=config,
    )

    assert "__interrupt__" in result

    interrupts = result["__interrupt__"]

    assert len(interrupts) == 1

    interrupt_value = interrupts[0].value

    assert interrupt_value["type"] == "approval"

    resumed = await graph.ainvoke(
        Command(resume=True),
        config=config,
    )

    assert resumed["approved"] is True
    assert resumed["result"] == "approved"