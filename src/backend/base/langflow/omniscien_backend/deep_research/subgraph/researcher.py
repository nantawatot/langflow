import asyncio
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, filter_messages
from langchain_core.runnables import RunnableConfig
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command

from langflow.logging import logger
from langflow.omniscien_backend.deep_research.configuration import Configuration
from langflow.omniscien_backend.deep_research.state import ResearcherOutputState, ResearcherState
from langflow.omniscien_backend.deep_research.utils import (
    execute_tool_safely,
    get_all_tools,
    get_today_str,
    is_token_limit_exceeded,
    normalize_message,
    remove_up_to_last_ai_message,
)


async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.

    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.

    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability

    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    logger.debug("Starting researcher")
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])

    # Get all available research tools (search, think_tool)
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError("No tools found to conduct research: Please configure either your search API")

    tool_available = "\n\n".join([f"- **{tool.name}**: {tool.description}" for tool in tools])

    # Step 2: Configure the researcher model with tools
    researcher_prompt = configurable.research_system_prompt.format(
        date=get_today_str(), tool_available_options=tool_available
    )

    # Configure model with tools, retry logic, and settings
    research_model = (
        configurable.get_model()
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    researcher_messages = normalize_message(researcher_messages)
    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)

    # Step 4: Update state and proceed to tool execution
    logger.debug("Researcher response generated, proceeding to tool execution")
    return Command(
        goto="researcher_tools",
        update={"researcher_messages": [response], "tool_call_iterations": state.get("tool_call_iterations", 0) + 1},
    )


async def researcher_tools(
    state: ResearcherState, config: RunnableConfig
) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.

    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, web_search) - Information gathering
    3. ResearchComplete - Signals completion of individual research task

    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings

    Returns:
        Command to either continue research loop or proceed to compression
    """
    logger.debug("Starting researcher")
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]

    # Early exit if no tool calls were made
    has_tool_calls = bool(most_recent_message.tool_calls)

    if not has_tool_calls:
        logger.debug("No tool calls found")
        return Command(goto="compress_research")

    # Step 2: Handle other tool calls (search, etc.)
    tools = await get_all_tools(config)
    tools_by_name = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool for tool in tools}

    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)

    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(content=observation, name=tool_call["name"], tool_call_id=tool_call["id"])
        for observation, tool_call in zip(observations, tool_calls, strict=False)
    ]

    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls
    )

    if exceeded_iterations or research_complete_called:
        logger.debug("Exiting researcher: max iterations reached or ResearchComplete called")
        command = Command(goto="compress_research", update={"researcher_messages": tool_outputs})
    else:
        logger.debug("No researcher reached or ResearchComplete called")
        command = Command(goto="researcher", update={"researcher_messages": tool_outputs})

    return command


async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.

    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.

    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with model settings

    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    logger.debug("Starting compress research")
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable.get_model()
    # Step 2: Prepare messages for compression
    researcher_messages = state.get("researcher_messages", [])

    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(HumanMessage(content=configurable.compress_research_human_message))

    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3

    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = configurable.compress_research_system_prompt.format(date=get_today_str())

            # Filter and convert messages to remove tool calls and tool messages
            # since compression model doesn't use tools
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

            messages = [SystemMessage(content=compression_prompt)] + filtered_messages
            logger.debug(f"Compression attempt {synthesis_attempts + 1} with {len(messages)} messages")

            # Execute compression
            response = await synthesizer_model.ainvoke(messages)

            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join(
                [str(message.content) for message in filter_messages(researcher_messages, include_types=["tool", "ai"])]
            )

            # Return successful compression result
            logger.debug("Compression successful")
            return {"compressed_research": str(response.content), "raw_notes": [raw_notes_content]}

        except Exception as e:
            synthesis_attempts += 1
            logger.error(f"Compression failed with exception {e}", exc_info=True)
            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue

            # For other errors, continue retrying
            continue

    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join(
        [str(message.content) for message in filter_messages(researcher_messages, include_types=["tool", "ai"])]
    )

    logger.debug("Compression failed after maximum retries")
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content],
    }


# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(ResearcherState, output=ResearcherOutputState, config_schema=Configuration)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)  # Main researcher logic
researcher_builder.add_node("researcher_tools", researcher_tools)  # Tool execution handler
researcher_builder.add_node("compress_research", compress_research)  # Research compression

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")  # Entry point to researcher
researcher_builder.add_edge("compress_research", END)  # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()
