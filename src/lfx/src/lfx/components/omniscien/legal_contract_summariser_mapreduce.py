import asyncio
import json
from dataclasses import dataclass
from functools import partial
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langflow.custom import Component
from langflow.inputs import DataInput, HandleInput, IntInput, MessageInput
from langflow.schema import Message
from langflow.template import Output
from langgraph.constants import END
from langgraph.graph import StateGraph
from pydantic import BaseModel

# Constants
DEFAULT_MAX_TOKENS = 128000
LEAF_NODE_PREFIX = "Leaf_"
MERGE_NODE_PREFIX = "Merge_"


@dataclass
class ProcessingBatch:
    """Represents a batch of data to be processed together."""

    content: str
    children: list[str]
    token_count: int


class GraphState(BaseModel):
    """State management for the contract summarization workflow."""

    docs: list[Document]
    json_list: list[dict]
    final_summary_json: dict
    parent_child_map: dict[str, list[str]]
    node_data: dict[str, Any]


class JSONProcessor:
    """Handles JSON template extraction and validation operations."""

    def __init__(self, logger_func):
        self.log = logger_func

    def extract_template_from_prompt(self, prompt: str) -> dict:
        """Extracts the JSON template structure from a prompt string.

        Args:
            prompt: The prompt containing a JSON template

        Returns:
            The extracted JSON template as a dictionary

        Raises:
            ValueError: If no valid JSON template is found
        """
        try:
            start_pos = prompt.find("{")
            if start_pos == -1:
                raise ValueError("No JSON object found in the prompt.")

            decoder = json.JSONDecoder()
            json_template, _ = decoder.raw_decode(prompt[start_pos:])
            return json_template
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(
                f"Failed to parse JSON template from prompt. Ensure it's valid, properly escaped JSON. Error: {e}"
            )

    def compare_json_structure(self, template: dict, response: dict) -> bool:
        """Recursively compares two JSON objects to ensure they have matching key structures.

        Args:
            template: The reference JSON structure
            response: The JSON to validate against the template

        Returns:
            True if structures match, False otherwise
        """
        template_keys = set(template.keys())
        response_keys = set(response.keys())

        if template_keys != response_keys:
            missing_keys = template_keys - response_keys
            extra_keys = response_keys - template_keys

            if missing_keys:
                self.log(f"Missing keys in response: {missing_keys}")
            if extra_keys:
                self.log(f"Extra keys in response: {extra_keys}")

            return False

        # Recursively check nested structures
        for key in template_keys:
            if isinstance(template[key], dict) and isinstance(response[key], dict):
                if not self.compare_json_structure(template[key], response[key]):
                    return False
            elif isinstance(template[key], list) and isinstance(response[key], list):
                # Check dictionary structures within lists
                for i in range(min(len(template[key]), len(response[key]))):
                    if isinstance(template[key][i], dict) and isinstance(response[key][i], dict):
                        if not self.compare_json_structure(template[key][i], response[key][i]):
                            return False

        return True

    def create_default_structure(self, template: dict) -> dict:
        """Creates a default JSON structure with None values based on a template.

        Args:
            template: The template to base the structure on

        Returns:
            A new dictionary with the same structure but None values
        """
        if isinstance(template, dict):
            result = {}
            for key, value in template.items():
                if isinstance(value, dict):
                    result[key] = self.create_default_structure(value)
                else:
                    result[key] = None
            return result
        if isinstance(template, list):
            return [
                self.create_default_structure(item) if isinstance(item, (dict, list)) else None for item in template
            ]
        return None

    def validate_and_fix_json(self, template: dict, response: dict) -> dict:
        """Validates a JSON response against a template and fixes it if invalid.

        Args:
            template: The reference JSON structure
            response: The JSON to validate

        Returns:
            The validated JSON or a default structure if validation fails
        """
        try:
            if self.compare_json_structure(template, response):
                return response
            self.log("JSON structure validation failed. Using default structure.")
            return self.create_default_structure(template)
        except Exception as e:
            self.log(f"Error during JSON validation: {e}")
            self.log(f"Problematic JSON: {response}")
            return self.create_default_structure(template)


class TokenBatchProcessor:
    """Handles batching of content based on token limits."""

    def __init__(self, llm, max_tokens: int, logger_func):
        self.llm = llm
        self.max_tokens = max_tokens
        self.log = logger_func

    def create_batches(
        self, items: list[Any], content_extractor, prompt_overhead: int, item_name_extractor=None
    ) -> list[ProcessingBatch]:
        """Creates batches of items that fit within token limits.

        Args:
            items: List of items to batch
            content_extractor: Function to extract content string from an item
            prompt_overhead: Number of tokens used by the prompt template
            item_name_extractor: Optional function to extract item names

        Returns:
            List of ProcessingBatch objects
        """
        batches = []
        current_batch_items = []
        current_batch_tokens = 0
        current_children = []

        for item in items:
            content = content_extractor(item)
            token_count = self.llm.get_num_tokens(content)
            item_name = item_name_extractor(item) if item_name_extractor else None

            if current_batch_items and current_batch_tokens + token_count + prompt_overhead > self.max_tokens:
                # Save current batch
                batch_content = self._format_batch_content(current_batch_items, content_extractor)
                batches.append(
                    ProcessingBatch(
                        content=batch_content, children=current_children.copy(), token_count=current_batch_tokens
                    )
                )

                # Start new batch
                current_batch_items = [item]
                current_batch_tokens = token_count
                current_children = [item_name] if item_name else []
            else:
                current_batch_items.append(item)
                current_batch_tokens += token_count
                if item_name:
                    current_children.append(item_name)

        # Add final batch if it exists
        if current_batch_items:
            batch_content = self._format_batch_content(current_batch_items, content_extractor)
            batches.append(
                ProcessingBatch(
                    content=batch_content, children=current_children.copy(), token_count=current_batch_tokens
                )
            )

        self.log(f"Created {len(batches)} batches for processing")
        return batches

    def _format_batch_content(self, items: list[Any], content_extractor) -> str:
        """Formats batch items into a single content string."""
        if len(items) == 1 and hasattr(items[0], "page_content"):
            # For document chunks
            return items[0].page_content
        # For JSON objects or other content
        contents = [content_extractor(item) for item in items]
        return "\n\n".join(str(c) for c in contents)


class ContractSummarizerMapReduce(Component):
    """Extracts and summarizes contract documents using a map-reduce LLM chain."""

    display_name = "Legal Contract Summarizer (Map-Reduce)"
    description = "Extracts and summarizes from contract documents using a multi-step map-reduce LLM chain."
    icon = "Omniscien"
    name = "ContractSummarizerMapReduce"

    inputs = [
        DataInput(
            name="contract_chunks",
            display_name="Contract Chunks",
            info="A list of LangChain-compatible Document chunks representing a legal contract.",
            required=True,
        ),
        MessageInput(
            name="map_prompt",
            display_name="Map Prompt",
            info="Prompt used to summarize each chunk. Must contain a sample JSON structure.",
            required=True,
        ),
        MessageInput(
            name="combine_prompt",
            display_name="Combine Prompt",
            info="Prompt used to combine summaries. Must contain the same JSON structure as the map prompt.",
            required=True,
        ),
        MessageInput(
            name="validation_prompt",
            display_name="Validation Prompt",
            info="Prompt used to validate the final summary against the original contract chunks.",
            required=True,
        ),
        MessageInput(
            name="consolidate_prompt",
            display_name="Consolidation Prompt",
            info="Prompt used to consolidate the final summary JSON after validation checks.",
            required=True,
        ),
        HandleInput(
            name="llm",
            display_name="Language Model",
            input_types=["LanguageModel"],
            info="The LLM used to run the summarization chain.",
            required=True,
        ),
        IntInput(
            name="max_input_tokens",
            display_name="Max Input Tokens",
            info="The maximum number of input tokens allowed for the LLM.",
            required=True,
            value=DEFAULT_MAX_TOKENS,
        ),
    ]

    outputs = [
        Output(
            name="response",
            display_name="Summary JSON",
            method="map_reduce",
        )
    ]

    batch_size = 5  # fixed batch size for throttling

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.json_processor = JSONProcessor(self.log)
        self.batch_processor = None  # Initialized in map_reduce method
        self.json_processor = None
        self.batch_processor = None

    async def _batched_llm_call(self, chain, inputs, batch_size=5, sleep=0.2):
        """Calls an LLM chain in small batches to avoid throttling."""
        all_results = []
        for i in range(0, len(inputs), batch_size):
            chunk = inputs[i : i + batch_size]
            # run this slice
            results = await chain.abatch(chunk)  # async version of batch
            all_results.extend(results)
            if sleep:
                await asyncio.sleep(sleep)  # polite pause
        return all_results

    def _find_node_name(self, json_obj: dict, node_data: dict[str, Any]) -> str | None:
        """Finds the node name associated with a JSON object."""
        for name, data in node_data.items():
            if data == json_obj:
                return name
        return None

    async def _map_node(self, state: GraphState) -> dict[str, Any]:
        """Extracts JSON from each document chunk and validates it.

        Returns:
            Updated state with json_list and node_data
        """
        self.log(f"Processing {len(state.docs)} document chunks in map phase...")

        # Set up the mapping chain
        map_prompt_template = PromptTemplate(template=self.map_prompt.template, input_variables=["text"])
        map_chain = map_prompt_template | self.llm | JsonOutputParser()

        # Extract JSON template for validation
        formatted_prompt = map_prompt_template.format(text="")
        json_template = self.json_processor.extract_template_from_prompt(formatted_prompt)
        self.log(f"Extracted JSON template for validation: {json_template}")

        # # Process document chunks
        # map_results = map_chain.batch([
        #     {"text": chunk.page_content} for chunk in state.docs
        # ])
        # Process document chunks
        map_inputs = [{"text": chunk.page_content} for chunk in state.docs]
        map_results = await self._batched_llm_call(map_chain, map_inputs, batch_size=5)

        # Validate and collect results
        partial_jsons = []
        node_data = {}

        for i, result in enumerate(map_results):
            validated_result = self.json_processor.validate_and_fix_json(json_template, result)
            partial_jsons.append(validated_result)

            node_name = f"{LEAF_NODE_PREFIX}{i}"
            node_data[node_name] = validated_result

        self.log(f"Successfully processed {len(partial_jsons)} JSON objects")
        return {"json_list": partial_jsons, "node_data": node_data}

    async def _reduce_node(self, state: GraphState) -> dict[str, Any]:
        """Combines JSON objects using the reduce prompt.

        Returns:
            Updated state with reduced json_list, node_data, and parent_child_map
        """
        self.log(f"Starting reduce phase with {len(state.json_list)} JSON objects...")

        json_list = state.json_list
        node_data = state.node_data
        parent_child_map = state.parent_child_map

        # Set up the reduction chain
        reduce_prompt_template = PromptTemplate(
            template=self.combine_prompt.template, input_variables=["json_list_str"]
        )
        reduce_chain = reduce_prompt_template | self.llm | JsonOutputParser()

        # Extract JSON template
        formatted_prompt = reduce_prompt_template.format(json_list_str="")
        json_template = self.json_processor.extract_template_from_prompt(formatted_prompt)

        # Calculate prompt overhead
        prompt_overhead = self.llm.get_num_tokens(formatted_prompt)
        self.log(f"Prompt overhead: {prompt_overhead} tokens")

        # Create batches for processing
        batches = self.batch_processor.create_batches(
            items=json_list,
            content_extractor=lambda x: json.dumps(x),
            prompt_overhead=prompt_overhead,
            item_name_extractor=lambda x: self._find_node_name(x, node_data),
        )

        # # Process batches
        # batch_contents = [batch.content for batch in batches]
        # results = reduce_chain.batch([{"json_list_str": content} for content in batch_contents])
        batch_contents = [batch.content for batch in batches]
        reduce_inputs = [{"json_list_str": content} for content in batch_contents]
        results = await self._batched_llm_call(reduce_chain, reduce_inputs, batch_size=5)

        # Process results and update state
        next_level_jsons = []
        for idx, (batch, merged_json) in enumerate(zip(batches, results, strict=False)):
            validated_json = self.json_processor.validate_and_fix_json(json_template, merged_json)
            next_level_jsons.append(validated_json)

            # Update node tracking
            parent_name = f"{MERGE_NODE_PREFIX}{len(json_list)}_{idx}"
            node_data[parent_name] = validated_json
            parent_child_map[parent_name] = [name for name in batch.children if name]

        self.log(f"Reduce phase complete. New list size: {len(next_level_jsons)}")
        return {"json_list": next_level_jsons, "node_data": node_data, "parent_child_map": parent_child_map}

    def _should_continue_reducing(self, state: GraphState) -> bool:
        """Determines whether the reduction process should continue."""
        should_continue = len(state.json_list) > 1
        self.log(f"Continue reducing? {should_continue} (current list size: {len(state.json_list)})")
        return should_continue

    async def _finalize_node(self, state: GraphState, docs: list[Document]) -> dict[str, Any]:
        """Performs final validation and consolidation of the summary.

        Returns:
            Updated state with final_summary_json
        """
        self.log("Starting finalization phase...")

        # Get the final summary candidate
        if len(state.json_list) != 1:
            self.log(f"Warning: Expected 1 final JSON, got {len(state.json_list)}. Using first.")

        final_summary = state.json_list[0] if state.json_list else {}
        self.log(f"Final summary candidate: {final_summary}")

        # Set up validation chain
        validation_prompt = PromptTemplate(
            template=self.validation_prompt.template, input_variables=["text", "summary"]
        )
        validation_chain = validation_prompt | self.llm | JsonOutputParser()

        # Calculate validation prompt overhead
        prompt_overhead = self.llm.get_num_tokens(validation_prompt.format(text="", summary=json.dumps(final_summary)))

        # Create batches for validation
        validation_batches = self.batch_processor.create_batches(
            items=docs, content_extractor=lambda doc: doc.page_content, prompt_overhead=prompt_overhead
        )

        # Validate against document batches
        self.log(f"Validating against {len(validation_batches)} document batches...")
        # validation_results = validation_chain.batch([
        #     {"text": batch.content, "summary": json.dumps(final_summary)}
        #     for batch in validation_batches
        # ])
        validation_inputs = [
            {"text": batch.content, "summary": json.dumps(final_summary)} for batch in validation_batches
        ]
        validation_results = await self._batched_llm_call(validation_chain, validation_inputs, batch_size=5)

        # Set up consolidation chain
        consolidate_prompt = PromptTemplate(
            template=self.consolidate_prompt.template, input_variables=["initial_summary", "validation_list_str"]
        )
        consolidation_chain = consolidate_prompt | self.llm | JsonOutputParser()

        # Perform final consolidation
        self.log("Performing final consolidation...")
        consolidated_result = consolidation_chain.invoke(
            {
                "initial_summary": json.dumps(final_summary),
                "validation_list_str": json.dumps(validation_results),
            }
        )

        # Final validation
        json_template = self.json_processor.extract_template_from_prompt(self.map_prompt.template.format(text=""))
        final_result = self.json_processor.validate_and_fix_json(json_template, consolidated_result)

        self.log(f"Finalization complete. Final result: {final_result}")
        return {"final_summary_json": final_result}

    # async def map_reduce(self) -> Message:
    #     """
    #     Main execution method that orchestrates the map-reduce summarization process.

    #     Returns:
    #         Message containing the final summarized JSON
    #     """
    #     # Initialize batch processor
    #     # Initialize processors
    #     self.json_processor = JSONProcessor(self.log)
    #     self.batch_processor = TokenBatchProcessor(
    #         self.llm, self.max_input_tokens, self.log
    #     )

    #     # Define the workflow graph
    #     workflow = StateGraph(GraphState)

    #     # Add nodes
    #     workflow.add_node("map", self._map_node)
    #     workflow.add_node("reduce", self._reduce_node)
    #     workflow.add_node("finalize", lambda state: self._finalize_node(
    #         state, [chunk.to_lc_document() for chunk in self.contract_chunks]
    #     ))

    #     # Define edges
    #     workflow.set_entry_point("map")
    #     workflow.add_edge("map", "reduce")
    #     workflow.add_conditional_edges(
    #         "reduce",
    #         self._should_continue_reducing,
    #         {
    #             True: "reduce",
    #             False: "finalize",
    #         }
    #     )
    #     workflow.add_edge("finalize", END)

    #     # Compile and execute
    #     app = workflow.compile()
    #     self.log("Workflow graph compiled successfully")

    #     try:
    #         # Prepare initial state
    #         docs = [chunk.to_lc_document() for chunk in self.contract_chunks]
    #         initial_state = {
    #             "docs": docs,
    #             "json_list": [],
    #             "final_summary_json": {},
    #             "parent_child_map": {},
    #             "node_data": {}
    #         }

    #         # Execute workflow
    #         self.log("Starting map-reduce execution...")
    #         final_state = await app.ainvoke(initial_state)  # <-- note 'await' and 'ainvoke'
    #         result = final_state.get('final_summary_json', {})

    #         self.log("Map-reduce execution completed successfully")
    #         return Message(text=json.dumps(result, indent=2))

    #     except Exception as e:
    #         self.log(f"Error during map-reduce execution: {e}")
    #         raise e

    async def map_reduce(self) -> Message:
        """Main execution method that orchestrates the map-reduce summarization process.
        Fully async-ready.
        """
        # Initialize processors
        self.json_processor = JSONProcessor(self.log)
        self.batch_processor = TokenBatchProcessor(self.llm, self.max_input_tokens, self.log)

        # Define the workflow graph
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("map", self._map_node)  # async node
        workflow.add_node("reduce", self._reduce_node)  # async node
        workflow.add_node(
            "finalize", partial(self._finalize_node, docs=[chunk.to_lc_document() for chunk in self.contract_chunks])
        )  # async node with docs

        # Define edges
        workflow.set_entry_point("map")
        workflow.add_edge("map", "reduce")
        workflow.add_conditional_edges(
            "reduce",
            self._should_continue_reducing,
            {
                True: "reduce",
                False: "finalize",
            },
        )
        workflow.add_edge("finalize", END)

        # Compile workflow
        app = workflow.compile()
        self.log("Workflow graph compiled successfully")

        # Prepare initial state
        docs = [chunk.to_lc_document() for chunk in self.contract_chunks]
        initial_state = {
            "docs": docs,
            "json_list": [],
            "final_summary_json": {},
            "parent_child_map": {},
            "node_data": {},
        }

        # Execute asynchronously
        self.log("Starting map-reduce execution...")
        final_state = await app.ainvoke(initial_state)
        result = final_state.get("final_summary_json", {})

        self.log("Map-reduce execution completed successfully")
        return Message(text=json.dumps(result, indent=2))
