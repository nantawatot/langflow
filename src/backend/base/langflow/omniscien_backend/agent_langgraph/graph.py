import asyncio
import operator
from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage, filter_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    tool,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from loguru import logger
from typing_extensions import TypedDict

from langflow.omniscien_backend.agent_langgraph.configuration import (
    Configuration,
)
from langflow.omniscien_backend.deep_research.utils import (
    ResearchComplete,
    execute_tool_safely,
    normalize_message,
)


class ResearcherOutputState(TypedDict):
    """Output state for the research agent containing final research results.

    This represents the final output of the research process with compressed
    research findings and all raw notes from the research process.
    """

    compressed_research: str
    raw_notes: Annotated[list[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]


class ResearcherState(TypedDict):
    """State for the research agent containing message history and research metadata.

    This state tracks the researcher's conversation, iteration count for limiting
    tool calls, the research topic being investigated, compressed findings,
    and raw research notes for detailed analysis.
    """

    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], operator.add]


async def get_all_tools(config: RunnableConfig = None) -> list[tool]:
    """Assemble complete toolkit including research, and search tools.

    Args:
        config: Configuration instance with model and tool settings
    Returns:
        List of all configured and available tools for research operations
    """
    # Start with core research tools
    tools = [tool(ResearchComplete), think_tool]
    if config:
        lang_tool = await get_tool_lang(config)
        tools.extend(lang_tool)
    return tools


async def get_tool_lang(config: RunnableConfig) -> list[tool]:
    """Get the tool from Langflow.

    Returns:
        The tool from Langflow.
    """
    configurable = Configuration.from_runnable_config(config)
    tools = configurable.tools or []

    return tools


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


async def llm_call(state: ResearcherState, config: RunnableConfig) -> dict:
    """Analyze current state and decide on next actions.

    The model analyzes the current conversation state and decides whether to:
    1. Call search tools to gather more information
    2. Provide a final answer based on gathered information

    Returns updated state with the model's response.
    """
    configuration: Configuration = Configuration.from_runnable_config(config)

    tools = await get_all_tools(config)

    tool_available = "\n\n".join([f"- **{tool.name}**: {tool.description}" for tool in tools])
    tool_prompt = f"""<Available Tools>
    You have access to tools:
    {tool_available}

    **CRITICAL: Use think_tool after each tool call to reflect on results and plan next step s**
    </Available Tools>"""
    model_with_tools = configuration.get_model().bind_tools(tools)
    research_agent_prompt = configuration.research_agent_prompt + tool_prompt
    convert_research_msg = normalize_message(state.get("researcher_messages"))
    logger.debug("Call LLM")
    convert_research_msg = tool_result_and_human_message(convert_research_msg)
    message = [SystemMessage(content=research_agent_prompt)] + convert_research_msg

    logger.info("Invoke")
    wrap_call = {
        "researcher_messages": [model_with_tools.invoke(message)],  # state["researcher_messages"]
        "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
    }
    logger.info("Invoke Success")

    return wrap_call


def tool_result_and_human_message(messages: BaseMessage | Sequence[BaseMessage]) -> list[BaseMessage]:
    """Add Assistance Message if tool result and human message are adjacent.

    Args:
        messages: The original message
    Returns:
        List containing the ToolMessage and a follow-up HumanMessage
    """
    for i in range(len(messages) - 1):
        logger.debug(f"Message {i} content: {get_role(messages[i])}")
        # logger.debug(f"Message {i}: {messages[i].type}")
        if get_role(messages[i]) == "tool" and get_role(messages[i + 1]) == "user":
            messages.insert(i + 1, AIMessage(content="acknowledged."))
            logger.info("Added acknowledged message")

    return messages


def get_role_from_basemessage(message):
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, ToolMessage):
        return "tool"
    return getattr(message, "role", "unknown")


def get_role_from_dict(message_dict):
    role = message_dict.get("role", "unknown")
    if role == "user":
        for block in message_dict.get("content", []):
            if isinstance(block, dict) and block.get("toolResult"):
                return "tool"
    return role


def get_role(messages: BaseMessage | dict) -> str:
    """Get Role From Message.

    Args:
        messages: Messages
    Returns:
        Role of the messages
    roles = [user, assistant, system, tool, unknown]
    """
    if isinstance(messages, BaseMessage):
        return get_role_from_basemessage(messages)
    if isinstance(messages, dict):
        return get_role_from_dict(messages)
    return "unknown"


async def tool_node(state: ResearcherState, config: RunnableConfig) -> dict:
    """Execute all tool calls from the previous LLM response.

    Executes all tool calls from the previous LLM responses.
    Returns updated state with tool execution results.
    """
    configuration: Configuration = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]

    # Early exit if no tool calls were made
    # has_tool_calls = bool(most_recent_message.tool_calls)
    if not isinstance(most_recent_message, AIMessage):
        logger.debug("Most recent message is not an AIMessage")
        return Command(goto="compress_research")
    if not hasattr(
        most_recent_message, "tool_calls"
    ):  # and not most_recent_message.tool_calls:  # and not has_tool_calls:
        logger.debug("No tool calls found")
        return Command(goto="compress_research")

    add_tools = await get_all_tools(config)
    tools_by_name = configuration.get_tools_by_name(add_tools)
    tool_calls = most_recent_message.tool_calls

    # Execute all tool calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    tool_outputs = [
        ToolMessage(content=observation, name=tool_call["name"], tool_call_id=tool_call["id"])
        for observation, tool_call in zip(observations, tool_calls, strict=False)
    ]

    # tool_outputs = [converse_tool_result_to_message(tool_out) for tool_out in tool_outputs]
    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configuration.max_react_tool_calls
    research_complete_called = any(tool_call["name"] == "ResearchComplete" for tool_call in tool_calls)
    if exceeded_iterations or research_complete_called:
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs},
        )
    return Command(
        goto="llm_call",
        update={"researcher_messages": tool_outputs},
    )


async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress research findings into a concise summary.

    Takes all the research messages and tool outputs and creates
    a compressed summary suitable for the supervisor's decision-making.
    """
    configuration: Configuration = Configuration.from_runnable_config(config)
    compress_research_system_prompt = configuration.compress_research_system_prompt
    compress_research_human_message = configuration.compress_research_human_message

    # tools = await get_all_tools(config)
    compress_model = configuration.get_model()
    system_message = compress_research_system_prompt + f"For context, today's date is {get_today_str}."
    researcher_messages = state.get("researcher_messages", [])
    # researcher_messages.append(AIMessage(content="acknowledged."))
    researcher_messages.append(HumanMessage(content=compress_research_human_message))

    synthesis_attempts = 0
    max_attempts = 123

    while synthesis_attempts < max_attempts:
        try:
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
            messages = [SystemMessage(content=system_message)] + filtered_messages  # researcher_messages
            response = compress_model.invoke(messages)

            raw_notes_content = "\n".join(
                [str(message.content) for message in filter_messages(researcher_messages, include_types=["tool", "ai"])]
            )

            return Command(
                goto="END",
                update={
                    "compressed_research": str(response.content),
                    "raw_notes": [raw_notes_content],
                    "researcher_messages": researcher_messages,
                },
            )
        except Exception as e:
            synthesis_attempts += 1
            logger.error(f"Compression failed with exception {e}", exc_info=True)
            continue

    # Extract raw notes from tool and AI messages
    raw_notes_content = "\n".join(
        [str(message.content) for message in filter_messages(researcher_messages, include_types=["tool", "ai"])]
    )

    logger.debug("Compression failed after maximum retries")
    return Command(
        goto="END",
        update={
            "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
            "raw_notes": [raw_notes_content],
            "researcher_messages": researcher_messages,
        },
    )


def should_continue(state: ResearcherState, config: RunnableConfig) -> Literal["tool_node", "compress_research"]:
    """Determine whether to continue research or provide final answer.

    Determines whether the agent should continue the research loop or provide
    a final answer based on whether the LLM made tool calls.

    Returns:
        "tool_node": Continue to tool execution
        "compress_research": Stop and compress research
    """
    messages = state.get("researcher_messages", [])
    last_message = messages[-1]

    # If the LLM makes a tool call, continue to tool execution
    if last_message.tool_calls:  # asattr(last_message, "tool_calls"): # :
        return "tool_node"
    # Otherwise, we have a final answer
    return "compress_research"


def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")


agent_builder = StateGraph(
    state_schema=ResearcherState, output_schema=ResearcherOutputState, context_schema=Configuration
)

# Add nodes to the graph
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",  # Continue research loop
        "compress_research": "compress_research",  # Provide final answer
    },
)
agent_builder.add_edge("tool_node", "llm_call")  # Loop back for more research
agent_builder.add_edge("compress_research", END)

# Compile the agent
researcher_agent = agent_builder.compile()
