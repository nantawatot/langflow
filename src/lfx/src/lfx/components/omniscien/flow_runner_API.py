import ast
import asyncio
import builtins
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langflow.custom import Component
from langflow.inputs.inputs import (
    BoolInput,
    DictInput,
    DropdownInput,
    # CustomInput,
    IntInput,
    MessageTextInput,
    MultilineInput,
)
from langflow.omniscien_backend.flowrunner.FlowAPI import FlowAPI
from langflow.schema import Data, dotdict
from langflow.template import Output


def merge_string_to_dict(string: str | dict, dict: dict) -> dict:
    """Parses a JSON-like string into a dictionary and merges it with another dictionary.

    Args:
    string (str): String to convert into a dictionary.
    dict (Dict): Dictionary to merge into.

    Returns:
    Dict: Merged dictionary.

    Raises:
    RuntimeError: If the string cannot be safely parsed.
    """
    # if the string is empty, return the Original dict
    if not string:
        return dict

    # attempt to convert string into Dictionary structure
    try:
        string_dict = ast.literal_eval(string)

    except Exception:
        if isinstance(string, str):
            # Cast to string
            string_dict = {"Result": string}

        elif isinstance(string, builtins.dict):
            string_dict = string
        else:
            string_dict = {}
    # merge dicts
    dict = string_dict | dict
    return dict


class FlowRunnerAPIComponent(Component):
    display_name = "Flow Runner API"
    description = "Component to run multiple flows sequentially."
    documentation: str = "https://docs.langflow.org/components-custom-components"
    name = "FlowRunnerAPI"
    icon = "Omniscien"

    inputs = [
        BoolInput(
            name="seqParallel",
            display_name="Enable Parallel running?",
            value=False,
            dynamic=True,
            real_time_refresh=True,
            advanced=False,
        ),
        IntInput(
            name="num_parallel_workers",
            display_name="Set Max Number of Parallel Requests (Min. 2",
            value=3,
            show=True,
            dynamic=True,
            advanced=False,
            real_time_refresh=True,
        ),
        CustomInput(
            name="flows",
            display_name="Select Flows",
            modal="https://devdemo.languagestudio.com/langflow/popup",
            info="Click here to input Flows to run",
            dynamic=True,
            real_time_refresh=True,
            advanced=False,
        ),
        MessageTextInput(
            name="flows_to_run",
            display_name="Flows to Run",
            info="This is input accepts only valid flows that exist in the LangFlow server. \n Removes inputs that are invalid Flows or duplicate Flows.",
            value=[],
            is_list=True,
            tool_mode=True,
            real_time_refresh=True,
            input_types=[],
            advanced=False,
        ),
        MultilineInput(
            name="flow_value",
            display_name="Flow Input",
            value="",
            dynamic=True,
            real_time_refresh=True,
            info=".",
            advanced=False,
        ),
        DropdownInput(
            name="mode",
            display_name="On Error:",
            options=["continue", "break"],
            value="continue",
            advanced=True,
            info="On Error, either continue to cascade the flows or break the flows from running.",
            show=True,
        ),
        MessageTextInput(
            name="LangflowAPI",
            display_name="Langflow API key (optional)",
            value="",
            info="API key to use for the Langflow API",
            advanced=True,
            show=False,
        ),
        MessageTextInput(
            name="API_URL",
            display_name="URL of the API to call",
            value="https://devdemo.languagestudio.com:3000/",
            info="URL of the Langflow API to call",
            advanced=True,
            show=True,
        ),
        DictInput(
            name="flow_dict",
            display_name="Language Studio Flow Data",
            value={},
            dynamic=True,
            real_time_refresh=True,
            is_list=True,
            advanced=True,
            show=True,
        ),
        IntInput(
            name="retries",
            display_name="Number of API call attempts (Minimum: 1)",
            value=1,
            advanced=True,
            show=True,
        ),
    ]

    field_order = ["seqParallel"]

    outputs = [
        Output(display_name="Output", name="output", method="build_output"),
    ]

    async def get_flow_names(self) -> list[str]:
        """Retrieves the list of all available flow names on the Langflow server.

        Returns:
            list[str]: A list of flow names.
        """
        flow_data = await self.alist_flows()
        return [flow_data.data["name"] for flow_data in flow_data]

    async def get_flow(self, flow_name_selected: str) -> Data | None:
        """Fetches a specific flow's data by name.

        Args:
            flow_name_selected (str): The name of the flow.

        Returns:
            Data | None: The flow's data if found, otherwise None.
        """
        flow_datas = await self.alist_flows()
        for flow_data in flow_datas:
            if flow_data.data["name"] == flow_name_selected:
                self.log(flow_data, "Flow Data:")
                return flow_data

        return None

    async def get_value_from_str_dict(self, str_dict: str, key: str) -> str | None:
        """Attempts to retrieve the value associated with a given key from a dictionary represented as a string or a dict object.
        str_dict (str | dict): The dictionary, either as a string (e.g., "{'key': 'value'}") or a dict object.
        key (str): The key whose value should be retrieved.
        str | None: The value associated with the key if found, otherwise an empty string. Returns an empty string if parsing fails or input type is unsupported.
        """
        parsed_dict = None
        try:
            # Convert the string to a dictionary
            if isinstance(str_dict, str):
                self.log("Is string")
                parsed_dict = ast.literal_eval(str_dict)
            elif isinstance(str_dict, dict):
                parsed_dict = str_dict
            else:
                self.log(f"Unsupported input type: {type(str_dict)}", "Popup input")
                return ""

        except Exception as e:
            self.log(str(e), "Exception while parsing string to dict:")
            return ""

        # Check if it's a dictionary and contains the key
        if key in parsed_dict:
            self.log(f"Key '{key}' found in parsed dict")
            return parsed_dict[key]
        return ""

    async def build_list_of_flows(self, flows_to_run: dict, input_value) -> list:
        """Constructs a list of FlowAPI instances based on the provided flows and input value.

        Args:
            flows_to_run (dict): A dictionary where keys are flow names and values are flow IDs.
            input_value: The input value to be added to each flow's payload.

        Returns:
            List: A list of FlowAPI objects, each configured with the specified flow name, flow ID, input value, retry count, and default payload.

        Notes:
            - Skips any flow entry where the flow name or flow ID is missing or empty.
            - Logs relevant information during flow creation and addition.
        """
        list_of_flows = []
        print("flows to run is of type", type(flows_to_run).__name__)

        for idx, (flow_name, flow_id) in enumerate(flows_to_run.items()):
            if not flow_name or not flow_id:
                continue  # Skip if either is missing or empty

            self.log(flow_name + ": ID : " + str(flow_id), str(idx))

            new_flow = FlowAPI(self._attributes.get("API_URL"))
            new_flow.set_flow_name(flow_name)
            new_flow.set_flow_id(flow_id)
            new_flow.add_payload("input_value", input_value)
            new_flow.set_retry_count(int(self._attributes.get("retries")))
            new_flow.prepare_default_payload()
            list_of_flows.append(new_flow)
            self.log(flow_name, "Added to flows to run")

        return list_of_flows

    def _handle_error(self, errored_flow: str, flows: list[FlowAPI]):
        """Handles errors encountered during flow execution by marking subsequent flows as cancelled.

        Args:
            errored_flow (str): The name of the flow that encountered an error.
            flows (List[FlowAPI]): A list of FlowAPI instances representing the flows in the execution sequence.
        Behavior:
            - Iterates through the provided flows.
            - When the flow with name matching `errored_flow` is found, sets a flag to indicate subsequent flows should be cancelled.
            - For each flow after the errored flow, calls `set_cancelled()` to mark it as cancelled.
        """
        flow_run = True
        for flow in flows:
            if flow.get_flow_name() == errored_flow:
                flow_run = False
            if flow.get_flow_name() != errored_flow and flow_run == False:
                flow.set_cancelled()

    async def run_flows(self, flows: list[FlowAPI]):
        """Executes a sequence of FlowAPI objects asynchronously and handles errors based on the configured mode.

        Args:
            flows (List[FlowAPI]): A list of FlowAPI instances to be executed sequentially.
        Behavior:
            - Logs the start of sequential execution.
            - For each flow in the list:
                - If the flow is a FlowAPI instance, sends its request asynchronously.
                - Otherwise, logs an error for the flow.
                - If the mode attribute is set to "break" and the flow has an error,
                  records the errored flow's name and breaks the loop.
            - If an error occurred and the loop was broken, handles the error for the errored flow.
        """
        self.log("Running sequentially", "Run Mode")
        errored_flow = ""

        for flow in flows:
            if isinstance(flow, FlowAPI):
                success = await flow.send_request()
            else:
                self.log(flow.get_flow_name(), "Error executing API request.")

            if self._attributes.get("mode") == "break" and flow.get_error():
                errored_flow = flow.get_flow_name()
                break

        if errored_flow:
            self._handle_error(errored_flow, flows)

    async def run_flows_in_parallel(self, flows: list) -> None:
        """Executes multiple FlowAPI instances in parallel using asyncio.

        Args:
            flows (List): A list of FlowAPI objects to be executed in parallel.

        Returns:
            None
        Side Effects:
            - Logs the start of parallel execution.
            - Logs a warning if a non-FlowAPI object is encountered in the flows list.
            - Logs the result of each flow execution, including errors and successes.

        Notes:
            - Each FlowAPI object's send_request() method is called asynchronously.
            - Exceptions raised during flow execution are caught and logged.
        """
        self.log("Running in Parallel", "Run Mode")
        tasks = []
        for flow in flows:
            if isinstance(flow, FlowAPI):
                tasks.append(flow.send_request())
            else:
                self.log("Non-FlowAPI object in flows list")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            flow = flows[i]
            if isinstance(result, Exception):
                self.log(f"Flow {i} failed with error: {result}")
            else:
                self.log(flow.get_flow_name(), f"Flow {i} completed: success = {result}")

    async def run_with_futures(self, flows: list, max_workers: int = 5):
        """Executes multiple flow objects concurrently using a thread pool, with support for error handling and break mode.

        Args:
            flows (List): A list of flow objects, each expected to have a `run_with_retries()` coroutine and `get_flow_name()`/`get_error()` methods.
            max_workers (int, optional): The maximum number of threads to use for concurrent execution. Defaults to 5.
        Behavior:
            - Each flow is executed in a separate thread, running its `run_with_retries()` coroutine.
            - If a flow raises an exception, the error is logged with the flow's name.
            - If "break" mode is enabled and any flow reports an error via `get_error()`, remaining flows are cancelled and error handling is triggered.
            - After execution, if any flow errored in break mode, `_handle_error` is called with the errored flow's name and the list of flows.
        """

        def run_flow_sync(flow):
            # Create a new event loop in this thread and run the async function
            return asyncio.run(flow.run_with_retries())

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_flow_sync, flow): flow for flow in flows}
            errored_flow = ""

            for future in as_completed(futures):
                flow = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    self.log(f"Flow failed with error: {e}", flow.get_flow_name())

                if self._attributes.get("mode") == "break" and flow.get_error():
                    self.log("Break mode activated. Cancelling remaining flows...", flow.get_flow_name())

                    errored_flow = flow.get_flow_name()

                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break  # Exit the loop early

        if errored_flow:
            self._handle_error(errored_flow, flows)

    async def collect_results_from_flows_as_dicts(self, flows: list) -> dict[str, str]:
        """Collects results from a list of FlowAPI instances and merges their responses into a single dictionary.

        Args:
            flows (List): A list of flow objects, expected to be instances of FlowAPI.

        Returns:
            Dict[str, str]: A dictionary containing merged responses from all completed flows.

        Notes:
            - Only flows that are instances of FlowAPI and have their response ready are processed.
            - Responses are merged using the merge_string_to_dict function.
            - Logs a message if a flow is not completed or not an instance of FlowAPI.
        """
        results: dict[str, str] = {}

        for flow in flows:
            if isinstance(flow, FlowAPI):
                if flow.is_response_ready:
                    print(flow.collect_response())

                    results = merge_string_to_dict(flow.collect_response(), results)
                else:
                    self.log(flow.get_flow_name(), "Is not completed")
            else:
                self.log("Is not an instance of a Flow")

        return results
        # Function to take in Flow inputs as a List and return only Flows that exist in LangFlow without duplicates

    def validate_flows(self, flows_input: list) -> list:
        """Filters and returns only valid, unique flow names from the user input.

        Args:
            flows_input (List): List of user-specified flow names.

        Returns:
            List: Validated list of unique flow names (non-empty, no duplicates).
        """
        validated_flows = []
        seen = set()

        for flow in flows_input:
            if flow and flow not in seen:
                validated_flows.append(flow)
                seen.add(flow)

        return validated_flows

    async def validate_flow(self, flow: str) -> str:
        valid_flows = await self.get_flow_names()
        return flow if flow in valid_flows else ""

    def validate_dicts(self, dicts_to_val: list[dict]) -> list[dict]:
        """Filters a list of dictionaries, removing any dictionary that is either empty or contains only a single key-value pair with an empty string as both the key and value.

        Args:
            dicts_to_val (list[dict]): A list of dictionaries to validate.

        Returns:
            list[dict]: A list of dictionaries that are not empty and do not consist solely of {'': ''}.
        """
        return [d for d in dicts_to_val if d and not (len(d) == 1 and "" in d and d[""] == "")]

    def flatten_dict(self, l: list) -> dict:
        """Flattens a list of dictionaries into a single dictionary. Non-recursive.

        Args:
            l (List): List of dictionaries to flatten.

        Returns:
            dict: Flattened dictionary.
        """
        result = {}
        if isinstance(l, list):
            for d in l:
                if isinstance(d, dict):
                    result.update(d)

        return result

    def unflatten_dict(self, d: dict) -> list[dict]:
        """Converts a flattened dictionary into a list of single-key dictionaries.

        Args:
            d (dict): Flattened dictionary.

        Returns:
            List[dict]: List of single-key dictionaries.
        """
        if not isinstance(d, dict):
            return []

        return [{k: v} for k, v in d.items()]

    def update_flow_data(self, flow_data: dict) -> bool:
        """Updates the 'flows_to_run' attribute by removing invalid or duplicate flows.

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        flows_to_run = self._attributes.get("flows_to_run", [])
        print(flows_to_run, "Flows to run before update")

        list_of_flows = flow_data.keys()
        print("List of flows:", list(list_of_flows))

        valid_flows = [flow for flow in flows_to_run if flow in list_of_flows]
        # Filter the original flow_data to only include valid_flows
        valid_flow_data = {key: flow_data[key] for key in valid_flows}

        print("Valid Flow Data", valid_flow_data)

        return valid_flows, valid_flow_data

    async def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None):
        """Updates the build configuration dictionary based on the provided field name and value.
        This method performs different update operations depending on the field being modified:
        - Synchronizes the visibility and value of 'num_parallel_workers' with 'seqParallel'.
        - If updating 'flows_to_run' or 'flow_dict', it flattens, updates, validates, and unflattens flow data.
        - If updating 'flows', it adds a new flow to the configuration and updates related fields.

        Args:
            build_config (dotdict): The build configuration dictionary to update.
            field_value (Any): The new value for the specified field.
            field_name (str | None, optional): The name of the field being updated.

        Returns:
            dotdict: The updated build configuration dictionary.
        """
        print("\n\n=== Updating Build Config === \n")
        # print(f"{field_name}: {field_value}")
        print(build_config["seqParallel"]["value"])
        build_config["num_parallel_workers"]["show"] = build_config["seqParallel"]["value"]
        build_config["num_parallel_workers"]["value"] = max(build_config["num_parallel_workers"]["value"], 2)

        # Unions flows to run and flows_to_run
        # Removes flows that don't exist in both
        if field_name == "flows_to_run" or field_name == "flow_dict":
            flow_data = self.flatten_dict(build_config["flow_dict"]["value"])
            print("Flow datas:", flow_data)
            updated_flows, updated_flow_data = self.update_flow_data(flow_data)

            build_config["flow_dict"]["value"] = self.validate_dicts(self.unflatten_dict(updated_flow_data))
            build_config["flows_to_run"]["value"] = self.validate_flows(updated_flows)

            return build_config

        if field_name == "flows":
            flow_to_add = await self.get_value_from_str_dict(build_config["flows"]["value"], "name")
            uuid_to_add = await self.get_value_from_str_dict(build_config["flows"]["value"], "id")
            print(uuid_to_add)

            if not flow_to_add or not uuid_to_add:
                return build_config
            new_flow = {flow_to_add: uuid_to_add}

            if flow_to_add:
                updated_flows = build_config["flows_to_run"]["value"]
                updated_flows.append(flow_to_add)

                flow_dict = self.flatten_dict(build_config["flow_dict"]["value"])
                flow_dict = flow_dict | new_flow
                flow_dict = self.unflatten_dict(flow_dict)

                # if isinstance(flow_dict, dict):
                #    flow_dict = [flow_dict]
                # flow_dict.append(new_flow)
                print(flow_dict)

                build_config["flows_to_run"]["value"] = self.validate_flows(updated_flows)
                build_config["flow_dict"]["value"] = self.validate_dicts(flow_dict)
                build_config["flows"]["value"] = ""

        return build_config

    def update_dict(self, dict1: dict, dict2: dict):
        return dict1 | dict2

    async def prepare_flows(self, flow_input: str) -> list[FlowAPI]:
        """Prepares a list of FlowAPI objects based on the provided flow input.

        Args:
            flow_input (str): The input string used to determine which flows to prepare.

        Returns:
            List[FlowAPI]: A list of FlowAPI instances that are ready to be run.
        Side Effects:
            Logs the selected flows for debugging or tracking purposes.
        """
        flows_selected = self._attributes.get("flow_dict")
        self.log(flows_selected, "flows_selected")
        flows_to_run = await self.build_list_of_flows(flows_selected, flow_input)

        return flows_to_run

    def prepare_output(self, flow_input: str) -> dict:
        """Prepares the output dictionary by merging a new dictionary containing the flow input with an empty output dictionary.

        Args:
            flow_input (str): The input string to be included in the output dictionary.

        Returns:
            Dict: A dictionary containing the key "input" mapped to the provided flow_input value.
        """
        output = {}
        dict_input = {"input": flow_input}

        return output | dict_input

    async def run_flowrunner(self, flows_to_run: list[FlowAPI]):
        """Executes a list of FlowAPI instances either in parallel or sequentially based on configuration.

        Args:
            flows_to_run (List[FlowAPI]): A list of FlowAPI objects to be executed.
        Behavior:
            - Logs the current mode and run mode (parallel or sequential).
            - If 'seqParallel' attribute is True, runs flows in parallel using a specified number of workers.
            - Otherwise, runs flows sequentially.
        Attributes Used:
            - mode: The current execution mode.
            - seqParallel: Boolean flag to determine parallel or sequential execution.
            - max_workers: Maximum number of parallel workers (used if parallel execution is enabled).
        """
        self.log(self._attributes.get("mode"), "Mode:")

        parallel = self._attributes.get("seqParallel")
        self.log(parallel, "Run Mode: parallel")

        max_workers = self._attributes.get("max_workers")

        if parallel:
            await self.run_with_futures(flows_to_run, max_workers=max_workers)
            # await self.run_flows_in_parallel(flows_to_run)
            # await self.run_flows(flows_to_run)
            # pass
        else:
            await self.run_flows(flows_to_run)
            # pass

    async def flows_collection(self, flows: list[FlowAPI]) -> dict:
        """Executes a collection of flow objects asynchronously and returns their results as a dictionary.

        Args:
            flows (List[FlowAPI]): A list of FlowAPI instances to be executed.

        Returns:
            Dict: A dictionary containing the results of each flow execution.
        Side Effects:
            Logs the results of the flows run.
        """
        results = await self.collect_results_from_flows_as_dicts(flows)
        self.log(results, "Results of flows run")
        return results

    def log_flows_status(self, flows: list[FlowAPI]) -> None:
        """Logs the status of each flow in the provided list.
        For each flow, checks if there is an error using `get_error()`. If an error exists,
        logs the error message along with the flow's name. Otherwise, logs a success message
        with the flow's name.

        Args:
            flows (List[FlowAPI]): A list of FlowAPI instances to check and log status for.

        Returns:
            None
        """
        for flow in flows:
            flow_error = flow.get_error()
            if flow_error:
                self.log(f"Error: {flow_error}", flow.get_flow_name())
            else:
                self.log("Run successfully", flow.get_flow_name())

    def create_final_output(self, partial_output: dict, results) -> dict:
        """Combines a partial output dictionary with additional results using the
        merge_string_to_dict function.

        Args:
            partial_output (Dict): The initial dictionary containing partial output data.
            results: Additional results to be merged into the partial output.

        Returns:
            Dict: The final output dictionary after merging the results.
        """
        return merge_string_to_dict(partial_output, results)

    async def build_output(self) -> Data:
        """Asynchronously builds the output data by preparing flows, running them, collecting results,
        and assembling the final output.
        Steps:
            1. Retrieves input data from attributes.
            2. Prepares flows to run based on the inputs.
            3. Prepares partial output from the inputs.
            4. Executes the prepared flows.
            5. Logs the status of the flows.
            6. Collects results from the executed flows.
            7. Combines partial output and results to create the final output.
            8. Logs the final output.

        Returns:
            Data: An instance containing the final output data.
        """
        inputs = self._attributes.get("flow_value")
        self.log(inputs, "inputs")

        flows_to_run = await self.prepare_flows(inputs)
        partial_output = self.prepare_output(inputs)

        await self.run_flowrunner(flows_to_run)
        self.log_flows_status(flows_to_run)

        results = await self.flows_collection(flows_to_run)
        final_output = self.create_final_output(partial_output, results)

        self.log(final_output, "Final output ")

        # Save JSON output to context for use in converting to JSON
        self.graph.context["jsonOutput"] = final_output

        return Data(data=final_output)
