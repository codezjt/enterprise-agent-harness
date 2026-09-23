from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from enterprise_harness.config import get_env


def resolve_model(
    model: str | BaseChatModel,
) -> str | BaseChatModel:
    """
    将 Harness 中的模型配置解析成
    DeepAgents 可以使用的模型。
    """

    if isinstance(model, BaseChatModel):
        return model

    provider = get_env("MODEL_PROVIDER")

    if provider == "dashscope":
        api_key = get_env("DASHSCOPE_API_KEY")

        if not api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY is not configured"
            )

        model_name = get_env("MODEL_NAME", model)
        base_url = get_env("MODEL_BASE_URL")

        if not base_url:
            raise ValueError(
                "MODEL_BASE_URL is not configured"
            )

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
        )

    return model