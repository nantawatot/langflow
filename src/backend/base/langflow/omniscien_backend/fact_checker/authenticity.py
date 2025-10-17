from datetime import UTC, datetime
from urllib.parse import urlparse

import whois
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from langflow.omniscien_backend.fact_checker.configuration import Configuration
from langflow.omniscien_backend.fact_checker.models import AuthenticityScore, WhoisInfo
from langflow.omniscien_backend.fact_checker.prompts import INVESTIGATION_PROMPT, SCORING_PROMPT


@tool
def get_whois_info(domain: str) -> str:
    """Performs a WHOIS lookup for a given domain to get registration details like creation date, registrar, and expiration date.
    Use this to assess the age and legitimacy of a domain. For example, a domain created very recently might be less trustworthy.
    """
    try:
        w = whois.whois(domain)

        # Calculate domain age if creation_date is available
        domain_age_years = None
        if w.creation_date:
            # Handle both single date and list of dates
            creation_date = w.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]

            if creation_date:
                domain_age_years = (datetime.now() - creation_date).days // 365

        info = WhoisInfo(
            domain_name=str(w.domain_name) if w.domain_name else domain,
            registrar=str(w.registrar) if w.registrar else "Unknown",
            creation_date=creation_date if isinstance(creation_date, datetime) else None,
            expiration_date=w.expiration_date if isinstance(w.expiration_date, datetime) else None,
            updated_date=w.updated_date if isinstance(w.updated_date, datetime) else None,
            name_servers=[str(ns) for ns in w.name_servers] if w.name_servers else [],
        )

        result = info.model_dump_json(indent=2)
        if domain_age_years is not None:
            result += f"\nDomain Age: approximately {domain_age_years} years"

        return result

    except Exception as e:
        print(f"WHOIS lookup failed for {domain}: {e}")
        return f"Error performing WHOIS lookup for {domain}: {e}"


def get_tool_descriptions(tools) -> str:
    """Format tool names and descriptions into a readable string for the prompt."""
    tool_texts = []
    for t in tools:
        tool_texts.append(f"- **{t.name}**: {t.description.strip()}")
    return "\n".join(tool_texts)


async def assess_source_authenticity(url: str, content: str | None, config: RunnableConfig) -> AuthenticityScore:
    """Perform a comprehensive, agent-based authenticity assessment for a given URL.
    This function uses an LLM with tools (web search, whois) to dynamically investigate the source.
    """
    configurable = Configuration.from_runnable_config(config)
    llm = configurable.get_language_model()
    search_tools = configurable.get_search_tools()

    if not llm:
        raise ValueError("LLM not provided in config for authenticity assessment.")
    if not search_tools:
        print("Warning: No search tools provided for authenticity assessment. Results will be limited.")

    all_tools = search_tools + [get_whois_info]

    print(
        f"Starting agentic authenticity assessment for {url} using {llm.__class__.__name__} with tools: {[authenticity_tool.name for authenticity_tool in all_tools]}"
    )

    domain = urlparse(url).netloc
    domain = domain.removeprefix("www.")

    try:
        # Method 1: Manual tool execution approach
        return await _assess_with_manual_tool_execution(llm, all_tools, url, domain, content)

        # Alternative Method 2: If you prefer the agent approach, uncomment the line below
        # return await _assess_with_agent_approach(llm, all_tools, url, domain, content)

    except Exception as e:
        print(f"Error during agentic authenticity assessment for {url}: {e!s}")
        return AuthenticityScore(
            overall_score=0.0,
            assessment_reasoning=f"Agentic assessment failed due to an error: {e}. Authenticity is uncertain.",
            assessed_at=datetime.now(UTC).isoformat(),
        )


async def _assess_with_manual_tool_execution(
    llm: BaseChatModel, all_tools: list, url: str, domain: str, content: str | None
) -> AuthenticityScore:
    """Approach 1: Manually execute tools and then use structured output
    This is more reliable and gives you control over the investigation flow.
    """
    # Step 1: Execute investigation using tools
    llm_with_tools = llm.bind_tools(all_tools)

    investigation_chain = INVESTIGATION_PROMPT | llm_with_tools

    # Get the initial investigation response
    investigation_response = await investigation_chain.ainvoke(
        {"url": url, "domain": domain, "content": content if content else "No content available."}
    )

    # Execute any tool calls
    investigation_results = [f"Initial Assessment: {investigation_response.content}"]

    if investigation_response.tool_calls:
        for tool_call in investigation_response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # Find and execute the tool
            tool_to_execute = next(
                (authenticity_tool for authenticity_tool in all_tools if authenticity_tool.name == tool_name), None
            )
            if tool_to_execute:
                try:
                    tool_result = await tool_to_execute.ainvoke(tool_args)
                    investigation_results.append(f"{tool_name} results: {tool_result}")
                except Exception as e:
                    investigation_results.append(f"{tool_name} failed: {e!s}")

    # Step 2: Use the investigation results to generate structured assessment
    scoring_chain = SCORING_PROMPT | llm.with_structured_output(AuthenticityScore)

    result = await scoring_chain.ainvoke({"url": url, "investigation_results": "\n\n".join(investigation_results)})

    # Add timestamp
    authenticity_score = AuthenticityScore.model_validate(result)
    authenticity_score.assessed_at = datetime.now(UTC).isoformat()

    # Calculate overall score
    authenticity_score.overall_score = authenticity_score.calculate_overall_score()

    return authenticity_score
