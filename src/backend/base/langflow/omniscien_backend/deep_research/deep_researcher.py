"""Main LangGraph implementation for the Deep Research agent."""

from typing import Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from langflow.logging import logger
from langflow.omniscien_backend.deep_research.configuration import (
    Configuration,
)
from langflow.omniscien_backend.deep_research.state import (
    AgentInputState,
    AgentState,
    ResearchQuestion,
)
from langflow.omniscien_backend.deep_research.subgraph.supervisor import supervisor_subgraph
from langflow.omniscien_backend.deep_research.utils import (
    get_today_str,
    is_token_limit_exceeded,
)


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor.

    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.

    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings

    Returns:
        Command to proceed to research supervisor with initialized context
    """
    logger.debug("Starting write_research_brief")

    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)

    # Configure model for structured research question generation
    research_model = (
        configurable.get_model()
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    # Step 2: Generate structured research brief from user messages
    prompt_content = configurable.transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])), date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])

    # Step 3: Initialize supervisor with research brief and instructions
    supervisor_system_prompt = configurable.lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations,
    )

    logger.debug("Research brief generated, moving to research_supervisor")

    return Command(
        goto="research_supervisor",
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief),
                ],
            },
        },
    )


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.

    This function takes all collected research findings and synthesizes them into a
    well-structured, comprehensive final report using the configured report generation model.

    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys

    Returns:
        Dictionary containing the final report and cleared state
    """
    logger.debug("Starting final_report_generation")

    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)

    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    writer_model = configurable.get_model()

    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None

    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            if configurable.output_type == "text":
                final_report_prompt = configurable.final_report_generation_prompt.format(
                    research_brief=state.get("research_brief", ""),
                    messages=get_buffer_string(state.get("messages", [])),
                    findings=findings,
                    date=get_today_str(),
                )
            else:
                final_report_prompt = configurable.final_json_generation_prompt.format(
                    research_brief=state.get("research_brief", ""),
                    messages=get_buffer_string(state.get("messages", [])),
                    findings=findings,
                    date=get_today_str(),
                )

            # Generate the final report
            final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])
            print(final_report.content)
            # Return successful report generation
            return {"final_report": final_report.content, "messages": [final_report], **cleared_state}

        except Exception as e:
            print(f"Error during final report generation: {e}")
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e):
                current_retry += 1

                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = configurable.context_window
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state,
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)

                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                logger.debug("Token limit exceeded, trying again")
                continue
            # Non-token-limit error: return error immediately
            logger.debug("Non-token-limit error, aborting final report generation")
            return {
                "final_report": f"Error generating final report: {e}",
                "messages": [AIMessage(content="Report generation failed.")],
                **cleared_state,
            }

    # Step 4: Return failure result if all retries exhausted
    logger.debug("Maximum retries exceeded, aborting final report generation")
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed.")],
        **cleared_state,
    }


# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(AgentState, input=AgentInputState, config_schema=Configuration)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node("write_research_brief", write_research_brief)  # Research planning phase
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)  # Research execution phase
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase

# Define main workflow edges for sequential execution
deep_researcher_builder.add_edge(START, "write_research_brief")  # Entry point
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")  # Research to report
deep_researcher_builder.add_edge("final_report_generation", END)  # Final exit point

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile(checkpointer=MemorySaver())
