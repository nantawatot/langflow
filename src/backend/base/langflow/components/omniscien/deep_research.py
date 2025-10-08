import uuid

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from langflow.custom.custom_component.component import Component
from langflow.inputs import HandleInput, IntInput
from langflow.io import MultilineInput, Output
from langflow.omniscien_backend.deep_research.deep_researcher import deep_researcher
from langflow.schema import Message


class DeepResearch(Component):
    display_name = "Deep Research"
    description = "Deep Research custom component"
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "Omniscien"
    name = "DeepResearch"

    inputs = [
        MultilineInput(
            name="query",
            display_name="Query",
            info="The research query to be answered.",
            value="When is Hollow Knight Silksong releasing? Provide details on the release date, platforms, and any notable features or changes from the original Hollow Knight game.",
            required=True,
        ),
        IntInput(
            name="max_concurrent_research_units",
            display_name="Max Concurrent Research Units",
            info="Maximum number of research units to run concurrently.",
            value=5,
            advanced=True,
        ),
        IntInput(
            name="max_researcher_iterations",
            display_name="Max Researcher Iterations",
            info="Maximum number of iterations the researcher can perform.",
            value=6,
            advanced=True,
        ),
        IntInput(
            name="max_react_tool_calls",
            display_name="Max REACT Tool Calls",
            info="Maximum number of REACT tool calls allowed.",
            value=10,
            advanced=True,
        ),
        HandleInput(
            name="llm",
            display_name="Language Model",
            input_types=["LanguageModel"],
            info="The LLM used to run the summarization chain.",
            required=True,
        ),
        HandleInput(
            name="tools",
            display_name="Tools",
            input_types=["Tool"],
            is_list=True,
            required=False,
            info="These are the tools that the agent can use to help with tasks.",
        ),
    ]

    outputs = [
        Output(display_name="Output", name="output", method="build_output"),
    ]

    async def build_output(self) -> Message:
        config: RunnableConfig = {
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "max_concurrent_research_units": self.max_concurrent_research_units,
                "max_researcher_iterations": self.max_researcher_iterations,
                "max_react_tool_calls": self.max_react_tool_calls,
                "llm": self.llm,
                "tools": self.tools or [],
            }
        }

        result = await deep_researcher.ainvoke(
            {
                "messages": [
                    HumanMessage(self.query),
                ]
            },
            config,
        )

        return Message(text=result["final_report"])
