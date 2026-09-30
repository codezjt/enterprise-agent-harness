# from .context import RunContext
# from .context_builder import RunContextBuilder
# from .context_factory import RunContextFactory
# from .manager import RunManager
# from .models import Run, RunStatus
# from .contract import Runtime
# from .result import RuntimeResult, RuntimeStatus
# from .agent_runtime_adapter import AgentRuntimeAdapter
#
# def __getattr__(name: str):
#     if name == "RunManager":
#         from .manager import RunManager
#
#         return RunManager
#     raise AttributeError(
#         f"module {__name__!r} has no attribute {name!r}"
#     )
#
#
# __all__ = [
#     "Run",
#     "RunStatus",
#     "RunContext",
#     "RunContextFactory",
#     "RunContextBuilder",
#     "RunManager",
#     "RuntimeResult",
#     "RuntimeStatus",
#     "AgentRuntimeAdapter",
# ]

from .contract import Runtime
from .result import RuntimeResult, RuntimeStatus

__all__ = [
    "AgentRuntimeAdapter",
    "Runtime",
    "RuntimeResult",
    "RuntimeStatus",
    "RunManager"
]


def __getattr__(name: str):
    if name == "AgentRuntimeAdapter":
        from .agent_runtime_adapter import AgentRuntimeAdapter

        return AgentRuntimeAdapter
    if name == "RunManager":
        from .manager import RunManager

        return RunManager
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )