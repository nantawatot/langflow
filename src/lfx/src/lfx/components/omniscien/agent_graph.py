import uuid

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langflow.custom.custom_component.component import Component
from langflow.inputs.inputs import MultilineInput
from langflow.io import HandleInput, IntInput
from langflow.omniscien_backend.agent_langgraph.graph import researcher_agent
from langflow.schema import Message
from langflow.template.field.base import Output


class GraphAgent(Component):
    display_name = "Graph Agent"
    description = "Agent with Langgraph"
    documentation = "https://docs.langchain.com/oss/python/langgraph/streaming#init-chat-model"
    icon = "Globe"

    inputs = [
        MultilineInput(name="input_value", display_name="Input"),
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
        MultilineInput(
            name="research_agent_prompt",
            display_name="Research Agent Prompt",
            info="Prompt template to use for the research agent.",
            advanced=False,
        ),
        MultilineInput(
            name="compress_research_system_prompt",
            display_name="Compress Research System Prompt",
            info="System message for research compress.",
            advanced=False,
        ),
        MultilineInput(
            name="compress_research_human_message",
            display_name="Compress Research Human Message",
            info="Human message for research compress.",
            advanced=False,
        ),
        IntInput(
            name="max_recursion_limit",
            display_name="Max Recursion Limit",
            advanced=False,
            value=25,
            info="The maximum number of Recursion.",
        ),
        IntInput(
            name="max_react_tool_calls",
            display_name="Max REACT Tool Calls",
            advanced=False,
            value=10,
            info="Maximum number of REACT tool calls allowed.",
        ),
    ]
    outputs = [
        Output(
            display_name="Output Message",
            name="message",
            method="message_response",
        ),
    ]

    async def message_response(self) -> Message:  # type: ignore[type-var]
        config: RunnableConfig = {
            "configurable": {
                "thread_id": str(uuid.uuid4()),
                "max_react_tool_calls": self.max_react_tool_calls,
                "llm": self.llm,
                "tools": self.tools or [],
                "research_agent_prompt": self.research_agent_prompt,
                "compress_research_system_prompt": self.compress_research_system_prompt,
                "compress_research_human_message": self.compress_research_human_message,
            },
            "recursion_limit": self.max_recursion_limit,
        }

        result = await researcher_agent.ainvoke(
            {
                "researcher_messages": [
                    HumanMessage(self.input_value),
                ]
            },
            config,
        )

        return Message(text=result["compressed_research"])
