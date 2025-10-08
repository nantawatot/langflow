"""Configuration management for the Open Deep Research system."""

import os
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""

    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        description="Maximum number of retries for structured output calls from models",
    )
    max_concurrent_research_units: int = Field(
        default=5,
        description="Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits.",
    )
    max_researcher_iterations: int = Field(
        default=6,
        description="Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions.",
    )
    max_react_tool_calls: int = Field(
        default=10,
        description="Maximum number of tool calls for the Researcher agent in a single iteration.",
    )
    max_content_length: int = Field(
        default=50000,
        description="Maximum character length for webpage content before summarization is applied.",
    )

    # LLM Configuration
    llm: BaseChatModel = Field(
        default=init_chat_model(
            model="us.meta.llama4-maverick-17b-instruct-v1:0",
            model_provider="bedrock_converse",
            temperature=0.1,
            max_tokens=8192,
        ),
        description="The language model to use for research operations.",
    )
    context_window: int = Field(
        default=1048576,
        description="Context window (input) size of the LLM in tokens.",
    )

    tools: list[Any] | None = Field(
        default=None,
        description="List of tools available to the Researcher agent.",
    )

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
            if field_name != "llm"  # Skip LLM field for environment variable lookup
        }
        # Handle LLM separately - it comes from configurable, not environment
        if "llm" in configurable:
            values["llm"] = configurable["llm"]

        return cls(**{k: v for k, v in values.items() if v is not None})

    def get_model(self) -> BaseChatModel:
        """Get the configured LLM model or default fallback."""
        return self.llm

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
