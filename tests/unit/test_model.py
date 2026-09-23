from langchain_core.language_models import BaseChatModel

from enterprise_harness.agent.model import resolve_model


def test_resolve_string_model():
    model = resolve_model("test-model")

    assert model == "test-model"


def test_resolve_chat_model():
    class FakeChatModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "fake"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise NotImplementedError

    model = FakeChatModel()

    result = resolve_model(model)

    assert result is model