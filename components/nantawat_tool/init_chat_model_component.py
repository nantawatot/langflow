from langchain.chat_models import init_chat_model
from langflow.field_typing import LanguageModel
from langflow.io import (
    BoolInput,
    DictInput,
    FloatInput,
    IntInput,
    MessageInput,
    SecretStrInput,
    StrInput,
)

from lfx.base.constants import STREAM_INFO_TEXT
from lfx.base.models.model import LCModelComponent
from lfx.inputs.inputs import BoolInput, MessageInput, MultilineInput


class InitChatModelComponent(LCModelComponent):
    display_name = "InitChatModel"
    description = "`init_chat_model` collection of large language models."
    documentation = "https://docs.langchain.com/oss/python/langgraph/streaming#init-chat-model"
    icon = "Globe"

    inputs = [
        MessageInput(name="input_value", display_name="Input"),
        StrInput(
            name="model",
            display_name="Model name",
            advanced=False,
            required=True,
            info="The name of the model to use. For example, `gpt-3.5-turbo`.",
        ),
        SecretStrInput(name="api_key", display_name="API Key", advanced=False, required=False, value=None),
        FloatInput(
            name="temperature",
            display_name="Temperature",
            advanced=False,
            required=False,
            value=0.7,
        ),
        DictInput(
            name="kwargs",
            display_name="Kwargs",
            advanced=True,
            required=False,
            is_list=True,
            value={},
        ),
        FloatInput(name="top_p", display_name="Top p", advanced=True, required=False, value=0.5),
        IntInput(
            name="max_tokens",
            display_name="Max tokens",
            advanced=False,
            value=256,
            info="The maximum number of tokens to generate for each chat completion.",
        ),
        BoolInput(
            name="stream",
            display_name="Stream",
            info=STREAM_INFO_TEXT,
            advanced=True,
        ),
        MultilineInput(
            name="system_message",
            display_name="System Message",
            info="System message to pass to the model.",
            advanced=False,
        ),
    ]

    def build_model(self) -> LanguageModel:  # type: ignore[type-var]
        output = init_chat_model(
            model=self.model,
            client=None,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            **self.kwargs,
        )
        return output
