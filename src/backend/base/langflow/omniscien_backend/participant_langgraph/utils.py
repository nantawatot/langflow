"""Utility functions and helpers for the Deep Research agent."""

import asyncio
import os
from datetime import datetime
from typing import Annotated, Literal

import pandas as pd
from dotenv import load_dotenv

# from tavily import AsyncTavilyClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    MessageLikeRepresentation,
    SystemMessage,
    ToolMessage,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    InjectedToolArg,
    tool,
)
from pydantic import BaseModel, Field

from langflow.omniscien_backend.agent_langgraph.configuration import Configuration

load_dotenv()

##########################
# Tavily Search Tool Utils
##########################
TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)


##########################
# Reflection Tool Utils
##########################
class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""

    status: bool = Field(description="Whether the research is complete.")


@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"


##########################
# Tool Utils
##########################


async def get_all_tools(config: RunnableConfig = None) -> list[tool]:
    """Assemble complete toolkit including research, and search tools.

    Args:
        config: Configuration instance with model and tool settings
    Returns:
        List of all configured and available tools for research operations
    """
    # Start with core research tools
    tools = [tool(ResearchComplete), think_tool]

    # Add configured search tools
    # search_tools = await get_search_tool()
    # tools.extend(search_tools)

    # Add any additional tools from configuration
    if config:
        lang_tool = await get_tool_lang(config)
        tools.extend(lang_tool)
    return tools


async def get_tool_lang(config: RunnableConfig) -> tool:
    """Get the tool from Langflow.

    Returns:
        The tool from Langflow.
    """
    configurable = Configuration.from_runnable_config(config)
    tools = configurable.tools or []

    return tools


def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """Extract notes from tool call messages."""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]


##########################
# Token Limit Exceeded Utils
##########################


def is_token_limit_exceeded(exception: Exception) -> bool:
    """Determine if an exception indicates a token/context limit was exceeded.

    Args:
        exception: The exception to analyze

    Returns:
        True if the exception indicates a token limit was exceeded, False otherwise
    """
    error_str = str(exception).lower()

    return (
        _check_openai_token_limit(exception, error_str)
        or _check_anthropic_token_limit(exception, error_str)
        or _check_gemini_token_limit(exception, error_str)
        or _check_bedrock_token_limit(exception, error_str)
    )


def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates OpenAI token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, "__module__", "")

    # Check if this is an OpenAI exception
    is_openai_exception = "openai" in exception_type.lower() or "openai" in module_name.lower()

    # Check for typical OpenAI token limit error types
    is_request_error = class_name in ["BadRequestError", "InvalidRequestError"]

    if is_openai_exception and is_request_error:
        # Look for token-related keywords in error message
        token_keywords = ["token", "context", "length", "maximum context", "reduce"]
        if any(keyword in error_str for keyword in token_keywords):
            return True

    # Check for specific OpenAI error codes
    if hasattr(exception, "code") and hasattr(exception, "type"):
        error_code = getattr(exception, "code", "")
        error_type = getattr(exception, "type", "")

        if error_code == "context_length_exceeded" or error_type == "invalid_request_error":
            return True
    return False


def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates Anthropic token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, "__module__", "")

    # Check if this is an Anthropic exception
    is_anthropic_exception = "anthropic" in exception_type.lower() or "anthropic" in module_name.lower()

    # Check for Anthropic-specific error patterns
    is_bad_request = class_name == "BadRequestError"

    if is_anthropic_exception and is_bad_request:
        # Anthropic uses specific error messages for token limits
        if "prompt is too long" in error_str:
            return True

    return False


def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates Google/Gemini token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, "__module__", "")

    # Check if this is a Google/Gemini exception
    is_google_exception = "google" in exception_type.lower() or "google" in module_name.lower()

    # Check for Google-specific resource exhaustion errors
    is_resource_exhausted = class_name in ["ResourceExhausted", "GoogleGenerativeAIFetchError"]

    if is_google_exception and is_resource_exhausted:
        return True

    # Check for specific Google API resource exhaustion patterns
    if "google.api_core.exceptions.resourceexhausted" in exception_type.lower():
        return True

    return False


def _check_bedrock_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates AWS Bedrock token limit exceeded."""
    # Analyze exception metadata
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, "__module__", "")

    # Check if this is an AWS/Bedrock exception
    is_bedrock_exception = (
        "bedrock" in exception_type.lower()
        or "bedrock" in module_name.lower()
        or "boto3" in module_name.lower()
        or "botocore" in module_name.lower()
    )

    # Bedrock often uses ValidationException for bad inputs
    is_validation_error = class_name == "ValidationException"

    # Look for known context/token length error patterns
    token_keywords = ["maximum context length", "reduce the length of the prompt", "context length", "tokens"]

    if is_bedrock_exception and is_validation_error:
        if any(keyword in error_str for keyword in token_keywords):
            return True

    # Fallback: If the error message itself is clearly a Bedrock context limit issue
    if "maximum context length" in error_str and "reduce the length of the prompt" in error_str:
        return True

    return False


def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    """Truncate message history by removing up to the last AI message.

    This is useful for handling token limit exceeded errors by removing recent context.

    Args:
        messages: List of message objects to truncate

    Returns:
        Truncated message list up to (but not including) the last AI message
    """
    # Search backwards through messages to find the last AI message
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            # Return everything up to (but not including) the last AI message
            return messages[:i]

    # No AI messages found, return original list
    return messages


##########################
# Misc Utils
##########################


def get_today_str() -> str:
    """Get current date formatted for display in prompts and outputs.

    Returns:
        Human-readable date string in format like 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"


async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {e!s}"


def normalize_message(researcher_messages):
    new_research_msg = []
    for message in researcher_messages:
        if isinstance(message, ToolMessage):
            message = converse_tool_result_to_message(message)
        new_research_msg.append(message)

    researcher_messages = new_research_msg
    return researcher_messages


def converse_tool_result_to_message(tool_result: ToolMessage) -> dict:
    """Convert a tool result into a ToolMessage."""
    if isinstance(tool_result.content, dict):
        object_content = [{"json": replace_nan_with_none(tool_result.content)}]
    elif isinstance(tool_result.content, list):
        object_content = [{"json": replace_nan_with_none(content_block)} for content_block in tool_result.content]

    else:
        object_content = [{"text": tool_result.content}]
    return {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "content": object_content,  # {"text": json.dumps(tool_result.content)},  # object_content
                    "toolUseId": tool_result.tool_call_id,
                }
            }
        ],
    }


def replace_nan_with_none(obj):
    if isinstance(obj, dict):
        return {k: replace_nan_with_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_nan_with_none(v) for v in obj]
    return None if pd.isna(obj) else obj


def compress_filter(researcher_messages: list, system_message: str, compress_model: BaseChatModel):
    filtered_messages = []
    for message in researcher_messages:
        if hasattr(message, "tool_calls") and message.tool_calls:
            # Convert AIMessage with tool calls to plain AIMessage
            # Include the text content but ignore tool calls
            if message.content:
                filtered_messages.append(AIMessage(content=str(message.content)))
        elif message.__class__.__name__ == "ToolMessage":
            # Convert ToolMessage to HumanMessage to preserve information
            filtered_messages.append(HumanMessage(content=f"Tool result: {message.content}"))
        else:
            # Keep other message types as-is
            filtered_messages.append(message)

    messages = [SystemMessage(content=system_message)] + filtered_messages
    response = compress_model.invoke(messages)
    return response


@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: list[str],
    max_results: Annotated[int, InjectedToolArg] = 1,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None,
) -> str:
    """Fetch and summarize search results from Tavily search API.

    Args:
        queries: List of search queries to execute
        max_results: Maximum number of results to return per query
        topic: Topic filter for search results (general, news, or finance)
        config: Runtime configuration for API keys and model settings

    Returns:
        Formatted string containing summarized search results
    """
    # Step 1: Execute search queries asynchronously
    search_results = await tavily_search_async(
        queries, max_results=max_results, topic=topic, include_raw_content=True, config=config
    )
    return str(search_results)


async def tavily_search_async(
    search_queries,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
    config: RunnableConfig = None,
):
    """Execute multiple Tavily search queries asynchronously.

    Args:
        search_queries: List of search query strings to execute
        max_results: Maximum number of results per query
        topic: Topic category for filtering results
        include_raw_content: Whether to include full webpage content
        config: Runtime configuration for API key access

    Returns:
        List of search result dictionaries from Tavily API
    """
    # Initialize the Tavily client with API key from config
    tavily_client = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    # Create search tasks for parallel execution
    search_tasks = [
        tavily_client.search(query, max_results=max_results, include_raw_content=include_raw_content, topic=topic)
        for query in search_queries
    ]

    # Execute all search queries in parallel and return results
    search_results = await asyncio.gather(*search_tasks)
    return search_results
