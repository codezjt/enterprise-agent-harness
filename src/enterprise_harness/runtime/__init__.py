import importlib

from .contract import Runtime
from .result import RuntimeResult, RuntimeStatus


def __getattr__(name: str):
    lazy = {
        "AgentRuntimeAdapter": ".agent_runtime_adapter",
        "LangGraphRuntime": ".langgraph_runtime",
        "LangGraphRuntimeAdapter": ".langgraph_runtime_adapter",
        "RunManager": ".manager",
        "Run": ".models",
        "RunStatus": ".models",
        "RunContext": ".context",
        "RunContextBuilder": ".context_builder",
        "RunContextFactory": ".context_factory",
        "Checkpoint": ".checkpoint",
        "CheckpointStore": ".checkpoint",
        "RecoveryManager": ".recovery",
        "RetryPolicy": ".retry",
        "FailureType": ".retry",
        "RetryDecision": ".retry",
    }
    if name in lazy:
        mod = importlib.import_module(lazy[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Runtime",
    "RuntimeResult",
    "RuntimeStatus",
    "AgentRuntimeAdapter",
    "LangGraphRuntime",
    "LangGraphRuntimeAdapter",
    "RunManager",
    "Run",
    "RunStatus",
    "RunContext",
    "RunContextBuilder",
    "RunContextFactory",
    "Checkpoint",
    "CheckpointStore",
    "RecoveryManager",
    "RetryPolicy",
    "FailureType",
    "RetryDecision",
]
