from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    """Main configuration class for the system."""

    # LLM Configuration
    llm: BaseChatModel = Field(
        description="The language model to use for fact-checking operations.",
    )

    search_tools: list[BaseTool] = Field(
        default_factory=list,
        description="List of search/retrieval tools - can be web search, RAG systems, or any callable tool.",
    )

    def get_language_model(self) -> BaseChatModel:
        """Get the configured language model."""
        return self.llm

    def get_search_tools(self) -> list[BaseTool]:
        """Get the configured search tools."""
        return self.search_tools

    @classmethod
    def from_runnable_config(cls, runnable_config: RunnableConfig | None = None) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig.

        This method is flexible and can handle any configuration structure.
        """
        configurable_params = runnable_config.get("configurable", {}) if runnable_config else {}

        # Start with default values
        configuration_values: dict[str, Any] = {}

        # Handle LLM - it comes from configurable, not environment
        if "llm" in configurable_params:
            configuration_values["llm"] = configurable_params["llm"]

        # Handle search_tools - flexible to accept any tool type
        if "search_tools" in configurable_params:
            configuration_values["search_tools"] = configurable_params["search_tools"]

        return cls(**configuration_values)
