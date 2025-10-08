import base64
import re
from collections.abc import Callable, Sequence
from typing import (
    Any,
    Literal,
    TypeVar,
    cast,
)

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput

# message
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    merge_message_runs,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.messages.tool import tool_call as create_tool_call
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import TypeBaseModel
from pydantic import Field


def _str_if_single_text_block(
    anthropic_content: list[dict[str, Any]],
) -> str | list[dict[str, Any]]:
    if len(anthropic_content) == 1 and anthropic_content[0]["type"] == "text":
        return anthropic_content[0]["text"]
    return anthropic_content


def _camel_to_snake(text: str) -> str:
    pattern = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
    return pattern.sub("_", text).lower()


_T = TypeVar("_T")


def _camel_to_snake_keys(obj: _T) -> _T:
    if isinstance(obj, list):
        return cast(_T, [_camel_to_snake_keys(e) for e in obj])
    if isinstance(obj, dict):
        return cast(_T, {_camel_to_snake(k): _camel_to_snake_keys(v) for k, v in obj.items()})
    return obj


def _extract_response_metadata(response: dict[str, Any]) -> dict[str, Any]:
    response_metadata = response
    # response_metadata only supports string, list or dict
    if "metrics" in response and "latencyMs" in response["metrics"]:
        response_metadata["metrics"]["latencyMs"] = [response["metrics"]["latencyMs"]]

    return response_metadata


def _bedrock_to_anthropic(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anthropic_content = []
    for block in _camel_to_snake_keys(content):
        if "text" in block:
            anthropic_content.append({"type": "text", "text": block["text"]})
        elif "tool_use" in block:
            block["tool_use"]["id"] = block["tool_use"].pop("tool_use_id", None)
            anthropic_content.append({"type": "tool_use", **block["tool_use"]})
        elif "image" in block:
            anthropic_content.append(
                {
                    "type": "image",
                    "source": {
                        "media_type": f"image/{block['image']['format']}",
                        "type": "base64",
                        "data": _bytes_to_b64_str(block["image"]["source"]["bytes"]),
                    },
                }
            )
        elif "tool_result" in block:
            anthropic_content.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block["tool_result"]["tool_use_id"],
                    "is_error": block["tool_result"].get("status") == "error",
                    "content": _bedrock_to_anthropic(block["tool_result"]["content"]),
                }
            )
        # Only occurs in content blocks of a tool_result:
        elif "json" in block:
            anthropic_content.append({"type": "json", **block})
        elif "guard_content" in block:
            anthropic_content.append(
                {
                    "type": "guard_content",
                    "guard_content": {
                        "type": "text",
                        "text": block["guard_content"]["text"]["text"],
                    },
                }
            )
        else:
            raise ValueError(
                "Unexpected content block type in content. Expected to have one of "
                "'text', 'tool_use', 'image', or 'tool_result' keys. Received:\n\n"
                f"{block}"
            )
    return anthropic_content


def _parse_response(response: dict[str, Any]) -> AIMessage:
    anthropic_content = _bedrock_to_anthropic(response.pop("output")["message"]["content"])
    tool_calls = _extract_tool_calls(anthropic_content)
    usage = UsageMetadata(_camel_to_snake_keys(response.pop("usage")))  # type: ignore[misc]
    return AIMessage(
        content=_str_if_single_text_block(anthropic_content),  # type: ignore[arg-type]
        usage_metadata=usage,
        response_metadata=_extract_response_metadata(response),
        tool_calls=tool_calls,
    )


def _snake_to_camel(text: str) -> str:
    split = text.split("_")
    return "".join(split[:1] + [s.title() for s in split[1:]])


def _snake_to_camel_keys(obj: _T, excluded_keys: set = set()) -> _T:
    if isinstance(obj, list):
        return cast(_T, [_snake_to_camel_keys(e, excluded_keys=excluded_keys) for e in obj])
    if isinstance(obj, dict):
        _dict = {}
        for k, v in obj.items():
            if k in excluded_keys:
                _dict[k] = v
            else:
                _dict[_snake_to_camel(k)] = _snake_to_camel_keys(v, excluded_keys=excluded_keys)
        return cast(_T, _dict)
    return obj


def _b64str_to_bytes(base64_str: str) -> bytes:
    return base64.b64decode(base64_str.encode("utf-8"))


def _bytes_to_b64_str(bytes_: bytes) -> str:
    return base64.b64encode(bytes_).decode("utf-8")


def _extract_tool_calls(anthropic_content: list[dict]) -> list[ToolCall]:
    tool_calls = []
    for block in anthropic_content:
        if block["type"] == "tool_use":
            tool_calls.append(create_tool_call(name=block["name"], args=block["input"], id=block["id"]))
    return tool_calls


def _anthropic_to_bedrock(
    content: str | list[str | dict[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(content, str):
        content = [{"text": content}]
    bedrock_content: list[dict[str, Any]] = []
    for block in _snake_to_camel_keys(content):
        if isinstance(block, str):
            bedrock_content.append({"text": block})
        # Assume block is already in bedrock format.
        elif "type" not in block:
            bedrock_content.append(block)
        elif block["type"] == "text":
            bedrock_content.append({"text": block["text"]})
        elif block["type"] == "image":
            # Assume block is already in bedrock format.
            if "image" in block:
                bedrock_content.append({"image": block["image"]})
            else:
                bedrock_content.append(
                    {
                        "image": {
                            "format": block["source"]["mediaType"].split("/")[1],
                            "source": {"bytes": _b64str_to_bytes(block["source"]["data"])},
                        }
                    }
                )
        elif block["type"] == "image_url":
            # Support OpenAI image format as well.
            bedrock_content.append({"image": _format_openai_image_url(block["imageUrl"]["url"])})
        elif block["type"] == "document":
            # Assume block in bedrock document format
            bedrock_content.append({"document": block["document"]})
        elif block["type"] == "tool_use":
            bedrock_content.append(
                {
                    "toolUse": {
                        "toolUseId": block["id"],
                        "input": block["input"],
                        "name": block["name"],
                    }
                }
            )
        elif block["type"] == "tool_result":
            bedrock_content.append(
                {
                    "toolResult": {
                        "toolUseId": block["toolUseId"],
                        "content": _anthropic_to_bedrock(block["content"]),
                        "status": "error" if block.get("isError") else "success",
                    }
                }
            )
        # Only needed for tool_result content blocks.
        elif block["type"] == "json":
            bedrock_content.append({"json": block["json"]})
        elif block["type"] == "guard_content":
            bedrock_content.append({"guardContent": {"text": {"text": block["text"]}}})
        else:
            raise ValueError(f"Unsupported content block type:\n{block}")
    # drop empty text blocks
    return [block for block in bedrock_content if block.get("text", True)]


def _upsert_tool_calls_to_bedrock_content(
    content: list[dict[str, Any]], tool_calls: list[ToolCall]
) -> list[dict[str, Any]]:
    existing_tc_blocks = [block for block in content if "toolUse" in block]
    for tool_call in tool_calls:
        if tool_call["id"] in [block["toolUse"]["toolUseId"] for block in existing_tc_blocks]:
            tc_block = next(block for block in existing_tc_blocks if block["toolUse"]["toolUseId"] == tool_call["id"])
            tc_block["toolUse"]["input"] = tool_call["args"]
            tc_block["toolUse"]["name"] = tool_call["name"]
        else:
            content.append(
                {
                    "toolUse": {
                        "toolUseId": tool_call["id"],
                        "input": tool_call["args"],
                        "name": tool_call["name"],
                    }
                }
            )
    return content


def _format_openai_image_url(image_url: str) -> dict:
    """Formats an image of format data:image/jpeg;base64,{b64_string}
    to a dict for bedrock api.

    And throws an error if url is not a b64 image.
    """
    regex = r"^data:image/(?P<media_type>.+);base64,(?P<data>.+)$"
    match = re.match(regex, image_url)
    if match is None:
        raise ValueError(
            "Bedrock does not currently support OpenAI-format image URLs, only "
            "base64-encoded images. Example: data:image/png;base64,'/9j/4AAQSk'..."
        )
    return {
        "format": match.group("media_type"),
        "source": {"bytes": _b64str_to_bytes(match.group("data"))},
    }


def _messages_to_bedrock(
    messages: list[BaseMessage],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Handle Bedrock converse and Anthropic style content blocks"""
    bedrock_messages: list[dict[str, Any]] = []
    bedrock_system: list[dict[str, Any]] = []
    # Merge system, human, ai message runs because Anthropic expects (at most) 1
    # system message then alternating human/ai messages.
    messages = merge_message_runs(messages)
    for msg in messages:
        content = _anthropic_to_bedrock(msg.content)
        if isinstance(msg, HumanMessage):
            # If there's a human, tool, human message sequence, the
            # tool message will be merged with the first human message, so the second
            # human message will now be preceded by a human message and should also
            # be merged with it.
            if bedrock_messages and bedrock_messages[-1]["role"] == "user":
                bedrock_messages[-1]["content"].extend(content)
            else:
                bedrock_messages.append({"role": "user", "content": content})
        elif isinstance(msg, AIMessage):
            content = _upsert_tool_calls_to_bedrock_content(content, msg.tool_calls)
            bedrock_messages.append({"role": "assistant", "content": content})
        elif isinstance(msg, SystemMessage):
            bedrock_system.extend(content)
        elif isinstance(msg, ToolMessage):
            if bedrock_messages and bedrock_messages[-1]["role"] == "user":
                curr = bedrock_messages.pop()
            else:
                curr = {"role": "user", "content": []}

            curr["content"].append(
                {
                    "toolResult": {
                        "content": content,
                        "toolUseId": msg.tool_call_id,
                        "status": msg.status,
                    }
                }
            )
            bedrock_messages.append(curr)
        else:
            raise ValueError(f"Unsupported message type {type(msg)}")
    return bedrock_messages, bedrock_system


def _format_tool_choice(
    tool_choice: dict[str, dict] | Literal["auto", "any"] | str,
) -> dict[str, dict[str, str]]:
    if isinstance(tool_choice, dict):
        return tool_choice
    if tool_choice in ("auto", "any"):
        return {tool_choice: {}}
    return {"tool": {"name": tool_choice}}


def _format_tools(
    tools: Sequence[dict[str, Any] | TypeBaseModel | Callable | BaseTool,],
) -> list[dict[Literal["toolSpec"], dict[str, dict[str, Any] | str]]]:
    formatted_tools: list = []
    for tool in tools:
        if isinstance(tool, dict) and "toolSpec" in tool:
            formatted_tools.append(tool)
        else:
            spec = convert_to_openai_tool(tool)["function"]
            spec["inputSchema"] = {"json": spec.pop("parameters")}
            formatted_tools.append({"toolSpec": spec})
    return formatted_tools


def _drop_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        new = {k: _drop_none(v) for k, v in obj.items() if _drop_none(v) is not None}
        return new
    return obj


class CustomLLM(BaseChatModel):
    """Custom LLM that simulates a language model with a fixed response."""

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    model_id: str = Field(alias="model")
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = Field(default=None, alias="stop")
    additional_model_request_fields: dict[str, Any] | None = None
    additional_model_response_field_paths: list[str] | None = None
    guardrail_config: dict[str, Any] | None = Field(default=None, alias="guardrails")

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "amazon_bedrock_converse_chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate responses for the given messages."""
        bedrock_messages, system = _messages_to_bedrock(messages)
        print(bedrock_messages)
        print("-------------------")
        print(system)
        params = self._converse_params(stop=stop, **_snake_to_camel_keys(kwargs, excluded_keys={"inputSchema"}))
        print(params)
        response = self.client.converse(messages=bedrock_messages, system=system, **params)

        response_message = _parse_response(response)
        return ChatResult(generations=[ChatGeneration(message=response_message)])

    def _converse_params(
        self,
        *,
        stop: list[str] | None = None,
        stopSequences: list[str] | None = None,
        maxTokens: list[str] | None = None,
        temperature: float | None = None,
        topP: float | None = None,
        tools: list | None = None,
        toolChoice: dict | None = None,
        modelId: str | None = None,
        inferenceConfig: dict | None = None,
        toolConfig: dict | None = None,
        additionalModelRequestFields: dict | None = None,
        additionalModelResponseFieldPaths: list[str] | None = None,
        guardrailConfig: dict | None = None,
    ) -> dict[str, Any]:
        if not inferenceConfig:
            inferenceConfig = {
                "maxTokens": maxTokens or self.max_tokens,
                "temperature": temperature or self.temperature,
                "topP": self.top_p or topP,
                "stopSequences": stop or stopSequences or self.stop_sequences,
            }
        if not toolConfig and tools:
            toolChoice = _format_tool_choice(toolChoice) if toolChoice else None
            toolConfig = {"tools": _format_tools(tools), "toolChoice": toolChoice}

        return _drop_none(
            {
                "modelId": modelId or self.model_id,
                "inferenceConfig": inferenceConfig,
                "toolConfig": toolConfig,
                "additionalModelRequestFields": additionalModelRequestFields or self.additional_model_request_fields,
                "additionalModelResponseFieldPaths": additionalModelResponseFieldPaths
                or self.additional_model_response_field_paths,
                "guardrailConfig": guardrailConfig or self.guardrail_config,
            }
        )

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | TypeBaseModel | Callable | BaseTool],
        *,
        tool_choice: dict | str | Literal["auto", "any"] | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        try:
            formatted_tools: list[dict] = [convert_to_openai_tool(tool) for tool in tools]
        except Exception:
            formatted_tools = _format_tools(tools)
        if tool_choice:
            tool_choice = _format_tool_choice(tool_choice)
            tool_choice_type = list(tool_choice.keys())[0]
            if tool_choice_type not in list(self.supports_tool_choice_values or []):
                if self.supports_tool_choice_values:
                    supported = (
                        f"Model {self.model_id} does not currently support tool_choice "
                        f"of type {tool_choice_type}. The following tool_choice types "
                        f"are supported: {self.supports_tool_choice_values}."
                    )
                else:
                    supported = f"Model {self.model_id} does not currently support tool_choice."

                raise ValueError(
                    f"{supported} Please see "
                    f"https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html "
                    f"for the latest documentation on models that support tool choice."
                )
            kwargs["tool_choice"] = _format_tool_choice(tool_choice)
        print("bind tools", formatted_tools)
        return self.bind(tools=formatted_tools, **kwargs)
