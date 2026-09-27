from dataclasses import dataclass


@dataclass(frozen=True)
class ContextRouting:
    use_tool_results: bool = True
    use_rag_results: bool = True
    use_memory_results: bool = True
    use_conversation: bool = True

    def enabled_sources(self) -> set[str]:
        sources: set[str] = {"system_policy", "task"}

        if self.use_tool_results:
            sources.add("tool")

        if self.use_rag_results:
            sources.add("rag")

        if self.use_memory_results:
            sources.add("memory")

        if self.use_conversation:
            sources.add("conversation")

        return sources