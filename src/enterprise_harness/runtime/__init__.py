from .context import RunContext
from .context_builder import RunContextBuilder
from .context_factory import RunContextFactory
from .models import Run, RunStatus


def __getattr__(name: str):
    if name == "RunManager":
        from .manager import RunManager

        return RunManager
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__ = [
    "Run",
    "RunStatus",
    "RunContext",
    "RunContextFactory",
    "RunContextBuilder",
    "RunManager",
]