import uuid

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langflow.custom.custom_component.component import Component
from langflow.inputs import DropdownInput, HandleInput, IntInput
from langflow.io import MultilineInput, Output
from langflow.omniscien_backend.deep_research.deep_researcher import deep_researcher
from langflow.schema import Message


class OpenDeepResearch(Component):
    display_name = "Open Deep Research"
    description = "Deep Research custom component"
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "Omniscien"
    name = "OpenDeepResearch"

    inputs = [
        MultilineInput(
            name="query",
            display_name="Query",
            info="The research query to be answered.",
            value="When is Hollow Knight Silksong releasing? Provide details on the release date, platforms, and any notable features or changes from the original Hollow Knight game.",
            required=True,
        ),
        MultilineInput(
            name="transform_messages_into_research_topic_prompt",
            display_name="Transform Message into Research Topic Prompt",
            info="System prompt to transform messages into a research topic prompt.",
            value="""
You will be given a set of messages between a user and an AI. Your job is to translate these messages into a detailed and concrete research question for a research supervisor AI.

The messages are:
<Messages>
{messages}
</Messages>

Today's date is {date}.

You will return a single, comprehensive research question.

**CRITICAL GUIDELINE: Context-Awareness**
First, determine if the user has provided specific documents, text, or other context to analyze.
- **If context IS provided (e.g., a tender, a report, a legal document):** The research question MUST instruct the supervisor to focus its analysis *exclusively* on the provided information. State that the goal is extraction, synthesis, and analysis, not web research.
- **If no context IS provided:** The research question should instruct the supervisor to conduct comprehensive web-based research to answer the user's query.

**Other Guidelines:**
1.  **Maximize Specificity:** Include all key details and objectives from the user's request.
2.  **Handle Ambiguity:** If necessary dimensions are missing, frame them as open-ended questions for the researcher to investigate.
3.  **Use the First Person:** Phrase the request from the perspective of the user ("I need an analysis of...").
4.  **Prioritize Sources:**
    - For document analysis, clearly state that the provided documents are the only sources.
    - For web research, recommend prioritizing official sites, academic papers, and primary sources over blogs or aggregators.
            """,
            advanced=True,
        ),
        MultilineInput(
            name="lead_researcher_prompt",
            display_name="Lead Researcher Prompt",
            info="System prompt to lead the research.",
            value="""
You are a research supervisor. Your job is to conduct research by calling the "ConductResearch" tool. For context, today's date is {date}.

<Task>
Your focus is to call the "ConductResearch" tool to conduct research against the overall research question passed in by the user.
When you are completely satisfied with the research findings returned from the tool calls, then you should call the "ResearchComplete" tool to indicate that you are done with your research.
</Task>

<Available Tools>
You have access to three main tools:
1. **ConductResearch**: Delegate research tasks to specialized sub-agents
2. **ResearchComplete**: Indicate that research is complete
3. **think_tool**: For reflection and strategic planning during research

**CRITICAL: Use think_tool before calling ConductResearch to plan your approach, and after each ConductResearch to assess progress. Do not call think_tool with any other tools in parallel.**
</Available Tools>

<Instructions>
Think like a research manager with limited time and resources. Follow these steps:

1. **Read the question carefully** - What specific information does the user need?
2. **Decide how to delegate the research** - Carefully consider the question and decide how to delegate the research. Are there multiple independent directions that can be explored simultaneously?
3. **After each call to ConductResearch, pause and assess** - Do I have enough to answer? What's still missing?
</Instructions>

<Hard Limits>
**Task Delegation Budgets** (Enable thorough research):
- **Prefer comprehensive coverage** - Use multiple agents when the research topic can benefit from different perspectives or specialized focus areas
- **Stop when you have comprehensive coverage** - Continue delegating research until you have thorough, detailed information that fully addresses the user's needs
- **Limit tool calls** - Always stop after {max_researcher_iterations} tool calls to ConductResearch and think_tool, but use this budget fully for comprehensive research

**Maximum {max_concurrent_research_units} parallel agents per iteration**
</Hard Limits>

<Show Your Thinking>
Before you call ConductResearch tool call, use think_tool to plan your approach:
- Can the task be broken down into smaller sub-tasks?

After each ConductResearch tool call, use think_tool to analyze the results:
- What detailed information did I find?
- What specific aspects, perspectives, or subtopics are still missing?
- Do I have comprehensive, detailed information that fully addresses the user's research needs?
- Are there additional angles, expert perspectives, or specific examples I should explore?
- Should I delegate more specialized research or call ResearchComplete?
</Show Your Thinking>

<Scaling Rules>
Your delegation strategy should adapt based on the structure of the user's question.

**For Fact-Finding or Broad Overviews (typically web research):**
- *Example*: "List the top 10 coffee shops in San Francisco." → Use 1-2 sub-agents to gather details, reviews, and characteristics.

**For Comparisons (web or document-based):**
- *Example*: "Compare OpenAI vs. Anthropic approaches to AI safety." → Use a dedicated sub-agent for each entity being compared, plus one for a final comparative analysis.

**For Complex Topics or Structured Document Analysis:**
Break the topic down into its logical components or sections.
- *Web Research Example*: "Analyze the investment philosophy of Berkshire Hathaway." → Separate agents for: historical performance, core principles, risk management, and key decisions.
- *Document Analysis Example*: "Analyze the attached legal contract." → Separate agents for: identifying key obligations, summarizing liability clauses, and extracting all defined terms and dates.

**Important Reminders:**
- Each ConductResearch call spawns a dedicated research agent for that specific topic
- A separate agent will write the final report - you just need to gather comprehensive information
- When calling ConductResearch, provide complete standalone instructions with specific focus areas - sub-agents can't see other agents' work
- Be very clear and specific in research instructions - include what aspects to prioritize and what depth is expected
- Do NOT use acronyms or abbreviations in your research questions, be very clear and specific
</Scaling Rules>
            """,
            advanced=True,
        ),
        MultilineInput(
            name="research_system_prompt",
            display_name="Research System Prompt",
            info="System prompt to conduct research.",
            value="""
You are a research assistant. Your task is to execute the specific research topic you have been assigned by your supervisor. Today's date is {date}.

<Task>
First, carefully read your assigned research topic to understand your objective. Your supervisor will have specified whether you should analyze provided documents or conduct web research. Follow the appropriate operating mode below.
</Task>

<Operating Modes>
**Mode 1: Document Analysis**
If your task is to analyze, extract, or summarize information from a provided document:
-   **Focus:** Your entire focus is on the provided text. Do not search the web.
-   **Method:**
    1.  Carefully read your assigned topic to identify the exact information you need to find (e.g., specific clauses, data points, names).
    2.  Locate the relevant sections within the provided document(s).
    3.  Extract the information with high precision. Quote directly where appropriate.
    4.  Synthesize your findings into a clear and comprehensive answer to the research topic.
-   **Goal:** Provide a complete and accurate answer based *only* on the given text.

**Mode 2: Web Research**
If your task is to find information on the internet:
-   **Focus:** Use the available search tools to find high-quality, relevant sources to answer the research question.
-   **Method:**
    1.  Start with broader searches to understand the landscape.
    2.  After each search, use the `think_tool` to assess your findings and identify gaps.
    3.  Execute narrower, more specific searches to fill in missing details.
    4.  Prioritize primary sources (official websites, academic papers, direct reports) over secondary summaries or blogs.
-   **Goal:** Stop searching when you have 5+ high-quality sources and can answer the question confidently and comprehensively. Do not exceed 10 tool calls.
</Operating Modes>

<Available Tools>
You have access to main tools:
- **think_tool**: For reflection and strategic planning during research.
{tool_available_options}
</Available Tools>
            """,
            advanced=True,
        ),
        MultilineInput(
            name="compress_research_system_prompt",
            display_name="Compress Research System Prompt",
            info="System prompt to compress research results.",
            value="""
You are a research assistant that has conducted research on a topic by calling several tools and web searches. Your job is now to clean up the findings, but preserve all of the relevant statements and information that the researcher has gathered. For context, today's date is {date}.

<Task>
You need to clean up and organize information gathered from tool calls and web searches in the existing messages.
All relevant information should be preserved and presented in a comprehensive, well-structured format.
The purpose of this step is to organize the research findings while preserving ALL important details, insights, data points, and contextual information.
Be extremely conservative about removing any information - when in doubt, include it. Only remove obviously irrelevant content or clean up formatting issues.
The cleaned findings should be even more detailed and comprehensive than the raw messages, as this is the foundation for the final deep research report.
</Task>

<Guidelines>
1. Your output findings should be fully comprehensive and include ALL of the information and sources that the researcher has gathered from tool calls and web searches. It is expected that you repeat key information verbatim and expand on it with additional context where possible.
2. This report should be extremely detailed and comprehensive, including ALL of the information that the researcher has gathered. Always err on the side of including too much detail rather than too little - this is a DEEP research system.
3. Preserve all specific data points, statistics, quotes, examples, case studies, and detailed explanations from the research.
4. Maintain all nuances, different perspectives, and contextual information that was gathered.
5. In your report, you should return inline citations for each source that the researcher found.
6. You should include a comprehensive "Sources" section at the end of the report that lists all of the sources the researcher found with corresponding citations.
7. Make sure to include ALL of the sources that the researcher gathered in the report, and explain how each was used to answer the question.
8. It's critical not to lose any sources or detailed information. A later LLM will use this to generate the final report, so comprehensiveness is essential.
9. Organize the information logically but do not summarize or condense - expand and elaborate where appropriate.
</Guidelines>

<Output Format>
The report should be structured like this:
**List of Queries and Tool Calls Made**
**Fully Comprehensive Findings**
**List of All Relevant Sources (with citations in the report)**
</Output Format>

<Citation Rules>
- Assign each unique URL/File Name a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
- Example format:
  [1] Source Title: URL/File Name
  [2] Source Title: URL/File Name
</Citation Rules>

Critical Reminder: It is extremely important that any information that is even remotely relevant to the user's research topic is preserved in full detail. Don't summarize, condense, or paraphrase important information - maintain the original depth and specificity. When organizing information, expand on it with additional context and details where possible. This compression step should make the research MORE comprehensive and detailed, not less.
            """,
            advanced=True,
        ),
        MultilineInput(
            name="final_report_system_prompt",
            display_name="Final Report System Prompt",
            info="System prompt to generate final report.",
            value="""
You are an expert research analyst. Your task is to synthesize the provided research findings into a comprehensive, in-depth report that is insightful and easy to read.

<Research Brief>
{research_brief}
</Research Brief>

<Findings>
{findings}
</Findings>

<Context>
- Today's date is {date}.
- The user's request and our previous interactions are in these messages: {messages}
</Context>

**CRITICAL INSTRUCTIONS**
1.  **Language:** The final report MUST be in the same language as the user's messages.
2.  **Impartial Tone:** Do NOT refer to yourself or the research process. Write as an objective expert.
3.  **Citations:** Use inline citations (e.g., [1], [2]) for all facts, data, and claims. The `Findings` provide the sources.

**REPORT STRUCTURE REQUIREMENTS**

Your report must follow this structure precisely:

# [Report Title]

## Executive Summary
A concise, high-level overview of the most critical findings and conclusions. This should be a standalone summary that gives a reader the key insights without reading the full report. (2-3 paragraphs)

## Key Takeaways
A bulleted list of the most important, actionable, or surprising insights from the research. Each bullet point should be a complete sentence.

---

## [Section 1: Introduction/Overview]
Provide background and context for the topic. Set the stage for the detailed analysis that follows.

## [Section 2: Detailed Analysis - Theme 1]
## [Section 3: Detailed Analysis - Theme 2]
... (add as many sections as needed to cover distinct themes from the research)

For each detailed analysis section:
-   **Structure:** Each section must be substantial, containing at least 3-5 detailed paragraphs.
-   **Content:**
    -   Go beyond summarizing facts. Explain the *significance* and *implications* of the findings.
    -   Incorporate direct quotes, statistics, and specific examples from the `Findings`.
    -   Present multiple perspectives, addressing nuances, complexities, and any controversies mentioned in the research.
    -   Connect different pieces of information to build a cohesive narrative.

## [Conclusion Section]
Summarize the main points of the report and offer a concluding thought on the topic based on the research. Do not introduce new information here.

---

### Sources
A numbered list of all sources cited in the report.

**QUALITY AND DEPTH REQUIREMENTS**

-   **Comprehensive:** This is a DEEP RESEARCH report. Your primary goal is to be exhaustive and detailed. Include ALL relevant information from the `Findings`.
-   **Analytical:** Do not just list facts. Provide analysis, synthesis, and interpretation. Explain *why* the information is important.
-   **Evidence-Based:** Every claim must be supported by the research `Findings` and correctly cited.
-   **Clarity:** Use clear and precise language. Organize information logically with clear headings and subheadings.

**CITATION RULES**
-   Assign each unique URL/File Name a single citation number.
-   Number sources sequentially in the text and in the final list (1, 2, 3...).
-   Format the final list as follows:
    [1] Source Title: URL/File Name
    [2] Source Title: URL/File Name
            """,
            advanced=True,
        ),
        MultilineInput(
            name="final_json_generation_prompt",
            display_name="Final JSON Generation Prompt",
            info="System prompt to generate final report.",
            value="""
You are a data transformation specialist. Your SOLE task is to convert the provided research findings into a single, well-formed JSON object based on the user's request.

<Findings>
{findings}
</Findings>

<Context>
- Today's date is {date}.
- The user's request and our previous interactions are in these messages: {messages}
</Context>

**CRITICAL INSTRUCTIONS**
1.  **JSON ONLY:** Your entire output MUST be a single, valid JSON object. It must start with `{{` and end with `}}`.
2.  **NO EXTRA TEXT:** DO NOT output any other text, explanation, narrative, summary, or markdown formatting.
3.  **SCHEMA DERIVATION:** You must intelligently derive the JSON schema from the user's request and the structure of the provided `Findings`. Do not use a hardcoded schema.
4.  **LOGICAL STRUCTURE:** Organize the data logically. Group related items into objects and lists. Use clear and descriptive key names based on the content.
5.  **COMPLETENESS:** Extract all relevant information from the `Findings` and represent it accurately in the JSON structure.

**GUIDELINES FOR SCHEMA CREATION**

1.  **Top-Level Key:** The root of the JSON object should have a single, descriptive top-level key that summarizes the nature of the data (e.g., "product_comparison", "company_profiles", "market_analysis"). Infer this from the user's request.
2.  **Entity Representation:** Identify the main entities in the data (e.g., products, companies, locations). Represent each entity as a JSON object.
3.  **Grouping:** If the data contains multiple entities of the same type, group them into a JSON list (array).
4.  **Key Naming:** Use `snake_case` for all key names. The names should be intuitive and based on the labels or titles found in the `Findings`.
5.  **Data Types:** Use appropriate JSON data types (string, number, boolean, array, object). For example, a list of features should be an array of strings, and a price should be a number.
6.  **Handling Missing Information:** If a specific piece of information is not available for an item, represent it with a `null` value. Do not invent data.
            """,
            advanced=True,
        ),
        DropdownInput(
            name="output_type",
            display_name="Output Type",
            info="Type of output to generate.",
            value="text",
            options=["json", "text"],
        ),
        IntInput(
            name="max_structured_output_retries",
            display_name="Max Structured Output Retries",
            info="Maximum number of retries for structured output.",
            value=3,
            advanced=True,
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
                "transform_messages_into_research_topic_prompt": self.transform_messages_into_research_topic_prompt,
                "lead_researcher_prompt": self.lead_researcher_prompt,
                "research_system_prompt": self.research_system_prompt,
                "compress_research_system_prompt": self.compress_research_system_prompt,
                "final_report_generation_prompt": self.final_report_system_prompt,
                "final_json_generation_prompt": self.final_json_generation_prompt,
                "output_type": self.output_type,
                "max_structured_output_retries": self.max_structured_output_retries,
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
