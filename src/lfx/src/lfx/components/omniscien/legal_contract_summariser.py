import json
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


class ContractSummarizerMapReduce(Component):
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
            value=128000,
        ),
    ]

    outputs = [
        Output(
            name="response",
            display_name="Summary JSON",
            method="map_reduce",
        )
    ]

    # --- NEW DYNAMIC HELPER FUNCTIONS ---

    def extract_json_from_template(self, prompt: str) -> dict:
        """Dynamically extracts the template JSON object from a prompt string.
        This template is used as the 'golden standard' for validation.
        """
        try:
            # Format with a dummy value to handle template variables like {text}
            # and unescape the double curly braces for the JSON.
            formatted_prompt = prompt.format(text="dummy_text", json_list_str="[]")
        except KeyError:
            # If formatting fails, it might just be a raw JSON string.
            formatted_prompt = prompt

        try:
            # Find the first opening brace of the JSON object.
            start_pos = formatted_prompt.find("{")
            if start_pos == -1:
                raise ValueError("No JSON object found in the prompt.")

            # Use the JSON decoder to parse one complete object from the string.
            decoder = json.JSONDecoder()
            json_template, _ = decoder.raw_decode(formatted_prompt[start_pos:])
            return json_template
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(
                f"Failed to parse the JSON template from the prompt. "
                f"Please ensure it's a valid, properly escaped JSON structure. Error: {e}"
            )

    def json_key_compare(self, original, response):
        """Recursively compares two JSON objects to ensure they have the same keys.

        Args:
            original (dict): The first JSON object (or dictionary).
            response (dict): The second JSON object (or dictionary).

        Returns:
            bool: True if both objects have the same keys at all nested levels,
                  False otherwise.
        """
        # Use sets to get the keys from both dictionaries
        original_keys = set(original.keys())
        response_keys = set(response.keys())

        # Check if the keys at the current level are the same
        if original_keys != response_keys:
            # Find which keys are different for more helpful output
            missing_in_response = original_keys - response_keys
            extra_in_response = response_keys - original_keys

            if missing_in_response:
                self.log(f"Keys from the first JSON are missing in the second: {missing_in_response}")
            if extra_in_response:
                self.log(f"Extra keys were found in the second JSON: {extra_in_response}")

            return False

        # Recursively check the keys for nested dictionaries
        for key in original_keys:
            # If a value in both is a dictionary, recurse into it
            if isinstance(original[key], dict) and isinstance(response[key], dict):
                if not self.json_key_compare(original[key], response[key]):
                    return False
            # If a value in both is a list, iterate and check for nested dictionaries
            elif isinstance(original[key], list) and isinstance(response[key], list):
                # This example assumes you want to compare dicts at the same index in each list
                for i in range(min(len(original), len(response))):
                    if isinstance(original[key][i], dict) and isinstance(response[key][i], dict):
                        if not self.json_key_compare(original[key][i], response[key][i]):
                            return False

        # If all key checks pass, return True
        return True

    def populate_default_json(self, original):
        """Recursively traverses a JSON object. It replaces all primitive values
        and all lists with the string "undefined". It only traverses into
        nested dictionaries. This modification happens in-place.

        Args:
            original (dict or list): The JSON object to process.

        Returns:
            The same object that was passed in, now modified.
        """
        # If the object is a dictionary, process its keys.
        if isinstance(original, dict):
            # Iterate over a copy of the keys, as we are modifying the dictionary.
            for key in list(original.keys()):
                value = original[key]

                # If the value is a nested dictionary, recurse into it.
                if isinstance(value, dict):
                    self.populate_default_json(value)

                # Otherwise, replace the value with the null.
                # This handles lists, strings, numbers, booleans, etc.
                else:
                    original[key] = None

        # If the top-level object is a list, process its items.
        # This is to handle cases like [{"a": 1}, {"b": 2}].
        elif isinstance(original, list):
            for item in original:
                # We only need to recurse if an item in the list is a dictionary
                # or a list that might contain dictionaries.
                if isinstance(item, (dict, list)):
                    self.populate_default_json(item)

        return original

    def check_json(self, original: dict, response: dict):
        valid = False
        try:
            valid = self.json_key_compare(original, response)
        except Exception as e:
            self.log(f"Error occurred during JSON validation: {e}")
            self.log(f"JSON object extracted from document chunk {response}")

        if not valid:
            return self.populate_default_json(original)

        return response

    # --- MAIN EXECUTION METHOD ---

    def map_reduce(self) -> Message:
        class GraphState(BaseModel):
            docs: list[Document]
            json_list: list[dict]
            final_summary_json: dict
            parent_child_map: dict[str, list[str]]
            node_data: dict[str, Any]

        def map_node(state: GraphState):
            """Extracts JSON from each document chunk and validates it dynamically."""
            self.log(f"Extracting JSON from {len(state.docs)} document chunks...")
            doc_chunks = state.docs
            map_prompt_template = PromptTemplate(template=self.map_prompt.template, input_variables=["text"])
            map_chain = map_prompt_template | self.llm | JsonOutputParser()
            map_results = map_chain.batch([{"text": chunk.page_content} for chunk in doc_chunks])

            # Dynamically get the required JSON structure from the prompt.
            json_template = self.extract_json_from_template(self.map_prompt.template)
            self.log(f"Dynamically extracted JSON template for validation.{json_template}")

            partial_jsons = []
            node_data = {}
            for i, res in enumerate(map_results):
                res = self.check_json(json_template, res)
                partial_jsons.append(res)
                node_name = f"Leaf_{i}"
                node_data[node_name] = res

            self.log(f"Extracted and sanitized {len(partial_jsons)} JSON objects and created leaf nodes.")
            return {"json_list": partial_jsons, "node_data": node_data}

        def reduce_node(state: GraphState):
            self.log(f"Performing reduce step (current list size: {len(state.json_list)})...")
            json_list = state.json_list
            node_data = state.node_data
            parent_child_map = state.parent_child_map

            json_template = self.extract_json_from_template(self.combine_prompt.template)
            self.log("Dynamically extracted JSON template for validation.")

            reduce_prompt_template = PromptTemplate(
                template=self.combine_prompt.template, input_variables=["json_list_str"]
            )
            reduce_chain = reduce_prompt_template | self.llm | JsonOutputParser()

            prompt_overhead_tokens = self.llm.get_num_tokens(reduce_prompt_template.format(json_list_str=""))
            self.log(f"Calculated prompt overhead: {prompt_overhead_tokens} tokens.")

            batches = []
            batch_node_children = []  # Keep track of which child nodes go with each batch

            current_batch = []
            current_batch_tokens = 0
            current_children = []

            def find_node_name(json_obj, node_data_dict):
                for name, data in node_data_dict.items():
                    if data == json_obj:
                        return name
                return None

            for json_obj in json_list:
                json_str = json.dumps(json_obj)
                token_count = self.llm.get_num_tokens(json_str)

                if current_batch and (
                    current_batch_tokens + token_count + prompt_overhead_tokens > self.max_input_tokens
                ):
                    # Save current batch for later LLM call
                    batches.append(json.dumps(current_batch))
                    batch_node_children.append(current_children)

                    current_batch = [json_obj]
                    current_batch_tokens = token_count
                    current_children = [find_node_name(json_obj, node_data)]
                else:
                    current_batch.append(json_obj)
                    current_batch_tokens += token_count
                    current_children.append(find_node_name(json_obj, node_data))

            if current_batch:
                batches.append(json.dumps(current_batch))
                batch_node_children.append(current_children)

            self.log(f"Prepared {len(batches)} batches for parallel reduction.")

            # Call LLM in parallel
            results = reduce_chain.batch([{"json_list_str": b} for b in batches])

            # Process results
            next_level_jsons = []
            for idx, merged_json in enumerate(results):
                checked_json = self.check_json(json_template, merged_json)
                next_level_jsons.append(checked_json)

                parent_name = f"Merge_{len(json_list)}_{idx}"
                node_data[parent_name] = checked_json
                parent_child_map[parent_name] = [name for name in batch_node_children[idx] if name]

            self.log(f"Reduction step complete. New list size: {len(next_level_jsons)}")
            return {"json_list": next_level_jsons, "node_data": node_data, "parent_child_map": parent_child_map}

        def should_continue_reducing(state: GraphState):
            """Determines whether the reduction process should continue."""
            self.log(f"Checking if reduction should continue (current list size: {len(state.json_list)})...")
            return len(state.json_list) > 1

        def finalize_node(state: GraphState):
            """Takes the final single-item list and extracts the dictionary."""
            self.log("Finalizing output...")
            if len(state.json_list) != 1:
                # This check is now more of a safeguard; the logic should prevent this.
                self.log(f"Warning: Final list size is {len(state.json_list)}, expected 1. Taking the first element.")
                final_summary_json = state.json_list[0] if state.json_list else {}
            else:
                final_summary_json = state.json_list[0]
            return {"final_summary_json": final_summary_json}

        # --- Graph Definition ---
        workflow = StateGraph(GraphState)

        workflow.add_node("map", map_node)
        workflow.add_node("reduce", reduce_node)
        workflow.add_node("finalize", finalize_node)

        workflow.set_entry_point("map")
        workflow.add_edge("map", "reduce")
        workflow.add_conditional_edges(
            "reduce",
            should_continue_reducing,
            {
                True: "reduce",
                False: "finalize",
            },
        )
        workflow.add_edge("finalize", END)

        app = workflow.compile()
        self.log("Graph compiled successfully.")

        result = {}
        try:
            docs = [chunk.to_lc_document() for chunk in self.contract_chunks]
            initial_state = {
                "docs": docs,
                "json_list": [],
                "final_summary_json": {},
                "parent_child_map": {},
                "node_data": {},
            }
            final_state = app.invoke(initial_state)
            result = final_state.get("final_summary_json", {})
            self.log(f"Final state: {result}")
        except Exception as e:
            self.log(f"Error occurred during graph execution: {e}")
            raise e

        return Message(text=json.dumps(result, indent=2))
