import uuid
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langflow.omniscien_backend.participant_langgraph.graph import researcher_agent
from langflow.schema import Message

from lfx.custom.custom_component.component import Component
from lfx.inputs.inputs import MultilineInput
from lfx.io import HandleInput, IntInput
from lfx.template.field.base import Output

INIT_PROMPT = """
# Event Athlete Information Extraction.
Query: {query}

### Detail Schema
{detail_schema}
Reminder: 
- If number of athletes is a lot (over 100), preserve name and team is more important than other attributes.

## Role
You are a **Sports Research Specialist** — an expert in analyzing professional sporting events and athlete participation. Your job is to extract and **gather the Athlete Detail section** for a single event using authoritative sources.

## Primary Objective
- Gather the **full and complete list of athletes** participating in the event.  
- Output should attributes of each athlete must strictly follow the schema `Detail Schema` provided in the input if cannot find return the literal string:  "no information".  
- Completeness is critical: **you must include every participant — no omissions are allowed.**
       
## Instructions

### 1. Event Identification
- Confirm the event referenced in the input (e.g., *Tour de France 2025*, *YONEX German Open 2025*, *FIFA World Cup 2026*).  
- Use this as the basis for searching authoritative sources.

### 2. Extract Participants from Input
- Read the input and align expected attributes to `Detail Schema` schema.  
- Treat any given entries as candidates to verify and enrich.

### 3. Determine Participant Totals
- Find the official total number of participants published by the event organizer or governing body.
- Cross-check this total against at least two independent authoritative sources (federation sites, governing body, or official media release).
- Record the verified participant_count for the event.

### 4. Find & Verify Full List
- Use official event websites, federation/association sites, or recognized databases (e.g., UCI, FIFA, ITF, BWF, FIBA, ATP, WTA, etc.) depending on the sport.  
- Gather the **full list of participants** (no omissions).  
- Confirm totals against authoritative counts.

### 5. Athlete Attributes
- For each athlete, extract attributes strictly according to `Detail Schema`.  
- All fields must be sourced from official/authoritative data.  
- Do not add fields not in `Detail Schema`.  

### 6. Team Details (if required by event/sport-based schema)
- If Sport or Event includes team information (e.g., football clubs, cycling teams, national squads), enrich it from official sources.  
- Verify roster counts against the official participant list.

### 7. Cross-validation
- Validate names, spellings, and attributes across multiple authoritative sources.  
- Ensure **no athletes are missing** and participant/team counts match official numbers.

### 8. Failover Rule
- If no authoritative data for the event or starters list can be found, return the literal string:  `"no information"`



## PRIMARY RESOURCE:
   - Official Website
   - Sport Information Website
   - News Sport Website

---

## CRITICAL: Final Output Constraints

**Your final response MUST be one of two things:**
1.  A single, valid JSON object that strictly adheres to the format specified below.
2.  The literal string `"no information"` if you cannot find a complete, authoritative list of participants.

**DO NOT output anything else.** Do not output explanations, comments, or your internal tool calls (e.g., `{{"name": "search_internet", ...}}`). Your reasoning and tool usage are intermediate steps, not the final answer.

---

### Example (short sample based on if Detail Schema is [full_name, team_name, nationality])

#### TEAM SPORT 
```json
{{
  "event": "Sample Race 2025",
  "athletes": [
  {{"Team Example 1":
        {{
        "atheleth name": ["Alice Smith", "Bob Johnson"],
        "nationality": "USA",
         }},
    {{"Team Example 2":
         {{
        "atheleth name": ["John Doe", "Jane Roe"],
        "nationality": "Italy",
         }},
  ]
}}
```

#### INDIVIDUAL SPORT
```json
{{
  "event": "Sample Badminton Open 2025",
  "athletes": [
    {{
      "full_name": "Viktor Axelsen",
      "nationality": "Denmark"
    }},
    {{
      "full_name": "Kento Momota",
      "nationality": "Japan"
    }}
  ],
}}
```

```json
{{
"event": "<Event Name from input>", 
  "athletes": [
    "team name" : [
      {{"athletes name":  name of athletes,
        fields according to {{Detail Schema}}
      }}
    ],
  ]
}}
```
"""

class ParticipantAgent(Component):
    display_name = "Participant Agent"
    description = "Participant Agent with Langgraph"
    documentation = "https://docs.langchain.com/oss/python/langgraph/streaming#init-chat-model"
    icon = "Globe"

    inputs = [
        MultilineInput(name="input_value", display_name="Input"),
        MultilineInput(
            name="detail_schema",
            display_name="Detail Schema",
            info="The detail schema in JSON list format. E.g. ['full_name', 'team_name', 'nationality']"
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
                # "max_concurrent_research_units": self.max_concurrent_research_units,
                # "max_researcher_iterations": self.max_researcher_iterations,
                "max_react_tool_calls": self.max_react_tool_calls,
                "llm": self.llm,
                "tools": self.tools or [],
                "research_agent_prompt": self.research_agent_prompt,
                "compress_research_system_prompt": self.compress_research_system_prompt,
                "compress_research_human_message": self.compress_research_human_message,
            },
            "recursion_limit": self.max_recursion_limit,
        }
        invoke_prompt = INIT_PROMPT.format(
            query=self.input_value,
            detail_schema=self.detail_schema or '["full_name", "team_name"]',
        )
        result = await researcher_agent.ainvoke(
            {
                "researcher_messages": [
                    HumanMessage(invoke_prompt),
                ]
            },
            config,
        )
        return Message(text=result["compressed_research"])
