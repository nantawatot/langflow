from datetime import datetime

import pandas as pd
from langchain_core.messages import (
    HumanMessage,
)
from lfx.base.langchain_utilities.model import LCToolComponent
from lfx.io import (
    HandleInput,
    MessageTextInput,
    Output,
)
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame
from pydantic import BaseModel

summarize_webpage_prompt = """You are tasked with summarizing the raw content of a webpage retrieved from a web search. Your goal is to create a summary that preserves the most important information from the original web page. This summary will be used by a downstream research agent, so it's crucial to maintain the key details without losing essential information.

Here is the raw content of the webpage:

<webpage_content>
{webpage_content}
</webpage_content>

Please follow these guidelines to create your summary:

1. Identify and preserve the main topic or purpose of the webpage.
2. Retain key facts, statistics, and data points that are central to the content's message.
3. Keep important quotes from credible sources or experts.
4. Maintain the chronological order of events if the content is time-sensitive or historical.
5. Preserve any lists or step-by-step instructions if present.
6. Include relevant dates, names, and locations that are crucial to understanding the content.
7. Summarize lengthy explanations while keeping the core message intact.

When handling different types of content:

- For news articles: Focus on the who, what, when, where, why, and how.
- For scientific content: Preserve methodology, results, and conclusions.
- For opinion pieces: Maintain the main arguments and supporting points.
- For product pages: Keep key features, specifications, and unique selling points.

Your summary should be significantly shorter than the original content but comprehensive enough to stand alone as a source of information. Aim for about 25-30 percent of the original length, unless the content is already concise.

Present your summary in the following format:

```
{{
   "summary": "Your summary here, structured with appropriate paragraphs or bullet points as needed",
   "key_excerpts": "First important quote or excerpt, Second important quote or excerpt, Third important quote or excerpt, ...Add more excerpts as needed, up to a maximum of 5"
}}
```

Here are two examples of good summaries:

Example 1 (for a news article):
```json
{{
   "summary": "On July 15, 2023, NASA successfully launched the Artemis II mission from Kennedy Space Center. This marks the first crewed mission to the Moon since Apollo 17 in 1972. The four-person crew, led by Commander Jane Smith, will orbit the Moon for 10 days before returning to Earth. This mission is a crucial step in NASA's plans to establish a permanent human presence on the Moon by 2030.",
   "key_excerpts": "Artemis II represents a new era in space exploration, said NASA Administrator John Doe. The mission will test critical systems for future long-duration stays on the Moon, explained Lead Engineer Sarah Johnson. We're not just going back to the Moon, we're going forward to the Moon, Commander Jane Smith stated during the pre-launch press conference."
}}
```

Example 2 (for a scientific article):
```json
{{
   "summary": "A new study published in Nature Climate Change reveals that global sea levels are rising faster than previously thought. Researchers analyzed satellite data from 1993 to 2022 and found that the rate of sea-level rise has accelerated by 0.08 mm/year² over the past three decades. This acceleration is primarily attributed to melting ice sheets in Greenland and Antarctica. The study projects that if current trends continue, global sea levels could rise by up to 2 meters by 2100, posing significant risks to coastal communities worldwide.",
   "key_excerpts": "Our findings indicate a clear acceleration in sea-level rise, which has significant implications for coastal planning and adaptation strategies, lead author Dr. Emily Brown stated. The rate of ice sheet melt in Greenland and Antarctica has tripled since the 1990s, the study reports. Without immediate and substantial reductions in greenhouse gas emissions, we are looking at potentially catastrophic sea-level rise by the end of this century, warned co-author Professor Michael Green."
}}
```

Remember, your goal is to create a summary that can be easily understood and utilized by a downstream research agent while preserving the most critical information from the original webpage.

Today's date is {date}.
"""


def get_today_str() -> str:
    """Get current date formatted for display in prompts and outputs.

    Returns:
        Human-readable date string in format like 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"


class Summary(BaseModel):
    """Research summary with key findings."""

    summary: str
    key_excerpts: str


class SearchLLMTool(LCToolComponent):
    display_name = "Custom Component222"
    description = "Use as a template to create your own component."
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "code"
    name = "CustomComponent222"

    inputs = [
        MessageTextInput(
            name="input_value",
            display_name="Input Value",
            info="This is a custom component Input",
            value="Hello, World!",
            tool_mode=True,
        ),
        HandleInput(
            name="llm",
            display_name="Language Model",
            input_types=["LanguageModel"],
            info="The LLM used to run the summarization chain.",
            required=True,
        ),
    ]

    outputs = [
        Output(display_name="Output", name="output", method="build_output"),
    ]

    def build_output(self) -> Data:
        data = Data(value=self.input_value)
        self.status = data
        return data

    def wrap_search_llm_tool(self, search_result: DataFrame):
        """Wrap Search.
        DataFrame input
        {
            "text": "The search query you want to execute with Tavily.",
            "title": "Title of the search result",
            "url": "URL of the search result",
            "content": "Content of the search result",
            "score": "Relevance score of the search result"
            "raw_content": "cleaned and parsed HTML of each search result"
        }
        """
        # search_result_converted = safe_convert(search_result, pd.DataFrame)
        summary_result = []
        for idx, row in search_result.iterrows():
            summary_result.append(self.reformat_content(row["raw_content"]))

        summary_result = search_result["raw_content_new"].apply(self.process_df)
        # summary_result = asyncio.run(*[
        #     self.reformat_content(row['raw_content']) for index, row in search_result.iterrows()
        # ])

        format_output = "Search Results: \n\n"
        for (index, row), summ in zip(search_result.iterrows(), summary_result.tolist(), strict=False):
            format_output += f"Title: {row['title']}\n"
            format_output += f"URL: {row['url']}\n"
            format_output += f"Summary: {summ.summary}\n"

    def process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        task = [self.reformat_content(row) for row in df[["raw_content"]]]
        results = task
        df["raw_content_new"] = results
        return df

    def reformat_content(self, webpage_content: str) -> str:
        """Reformat Content and Pass to LLM"""
        prompt_content = summarize_webpage_prompt.format(webpage_content=webpage_content, date=get_today_str())
        llm_with_structure = self.llm.with_structured_output(Summary)
        summary = llm_with_structure.invoke([HumanMessage(content=prompt_content)])
        return summary


if __name__ == "__main__":
    pass
