import ast
from typing import Any

import httpx

from langflow.base.flow_processing.utils import build_data_from_run_outputs
from langflow.custom import Component
from langflow.helpers.flow import run_flow
from langflow.inputs.inputs import (
    DropdownInput,
    MessageTextInput,
)
from langflow.schema import Data, dotdict
from langflow.template import Output


def merge_string_to_dict(string: str, dict: dict) -> dict:
    """Parses a JSON-like string into a dictionary and merges it with another dictionary.

    Args:
    string (str): String to convert into a dictionary.
    dict (Dict): Dictionary to merge into.

    Returns:
    Dict: Merged dictionary.

    Raises:
    RuntimeError: If the string cannot be safely parsed.
    """
    # attempt to convert string into Dictionary structure
    try:
        string_dict = ast.literal_eval(string)
    except Exception:
        # msg = f"Error converting string input to dict: " + string
        # logger.exception(msg + "string:" + string)

        # Cast to string
        string_dict = {"Response": string}
        # raise RuntimeError(msg) from e

    # merge dicts
    dict = string_dict | dict
    return dict


class JobProgressManager:
    """Manages the progress of a job by sending updates to a remote service."""

    def __init__(self, base_url: str, job_id: str):
        """Initializes the JobProgressManager.

        Args:
            base_url: The base URL of the job progress service.
            job_id: The unique identifier for the job.
        """
        self.base_url = base_url
        self.job_id = job_id
        self.headers = {"Content-Type": "application/json"}
        self.job_steps = {}

    async def initialize_steps(self, step_names: list[str]):
        """Initializes the job with a list of step names and sends it to the service.
        This method now populates the self.job_steps property.
        """
        # Create the job steps dictionary
        self.job_steps = {
            str(i + 1): {
                "jobStepNo": i + 1,
                "jobStepName": name,
                "jobStepShortName": name.lower().replace(" ", "_"),
                "jobStepPercentComplete": 0,
                "jobStepStatusMessage": "Pending",
            }
            for i, name in enumerate(step_names)
        }

        # This is the main payload for the entire job's initial state
        payload = {
            "jobId": self.job_id,
            "jobStatus": 1,  # Assuming 1 means "InProgress"
            "jobStepTotalCount": len(step_names),
            "jobSteps": self.job_steps,
        }

        url = f"{self.base_url}/update-job-info/{self.job_id}"
        async with httpx.AsyncClient() as client:
            try:
                await client.post(url, headers=self.headers, json=payload)
            except httpx.RequestError as e:
                print(f"An error occurred while initializing job steps: {e}")

    async def update_step(self, step_no: int, percent: int, message: str):
        """Updates the status of a specific job step.
        It reuses the step name and short name from the self.job_steps property.
        """
        # Retrieve the step details from the stored property
        step_info = self.job_steps.get(str(step_no))

        if not step_info:
            print(f"Error: Step number {step_no} was not found. Make sure initialize_steps was called.")
            return

        # Build the payload using the stored names
        payload = {
            "jobStepNo": step_no,
            "jobStepName": step_info["jobStepName"],  # Reusing the name
            "jobStepShortName": step_info["jobStepShortName"],  # Reusing the short name
            "jobStepPercentComplete": percent,
            "jobStepStatusMessage": message,
        }

        url = f"{self.base_url}/update-job-step/{self.job_id}"
        async with httpx.AsyncClient() as client:
            try:
                await client.post(url, headers=self.headers, json=payload)
            except httpx.RequestError as e:
                print(f"An error occurred while updating step {step_no}: {e}")

    async def mark_job_index(self, total: int, current: int):
        """Updates the overall progress index of the job."""
        payload = {
            "jobId": self.job_id,
            "jobStatus": 1,  # Assuming 1 means "InProgress"
            "jobStepTotalCount": total,
            "jobStepCurrentIndex": current,
        }
        url = f"{self.base_url}/update-job-info/{self.job_id}"
        async with httpx.AsyncClient() as client:
            try:
                await client.post(url, headers=self.headers, json=payload)
            except httpx.RequestError as e:
                print(f"An error occurred while marking job index: {e}")


class Flow:
    """Flow class manages the execution and results of a single flow instance."""

    def __init__(self, flow_name: str, inputs: str, user_id: str, session_id: str):
        """Initialize a new Flow instance.

        Args:
            flow_name (str): The name of the flow to run.
            inputs (str): The input string to pass to the flow.
            user_id (str): Identifier for the user running the flow.
            session_id (int): Identifier for the session.
        """
        self._flow_name: str = flow_name
        self._inputs: str = inputs
        self._results: dict = {}
        self._retries: int = 0
        self._user_id: str = user_id
        self._session_id: str = session_id
        self._is_completed: bool = False

    def set_retries(self, no_of_retries: int):
        """Set the number of retries allowed if the flow fails.

        Args:
            no_of_retries (int): The number of retry attempts.
        """
        self._retries = max(0, no_of_retries)

    async def run_flow(self):
        """Executes the flow asynchronously and stores the result."""
        tweaks = {}  # No tweaks applied
        try:
            self._results = await run_flow(
                inputs={"input_value": self._inputs},
                output_type="all",
                flow_id=None,
                flow_name=self._flow_name,
                tweaks=tweaks,
                user_id=str(self._user_id),
                session_id=self._session_id,
            )
            self._is_completed = True
        except Exception as e:
            raise RuntimeError("Oof!" + str(e)) from e

    async def run_flow_with_retries(self):
        """Placeholder for running the flow with retry logic.
        To be implemented.
        """

    def collect_result_as_string(self) -> str:
        """Converts the flow output into a concatenated string.

        Returns:
            str: Concatenated string of all text results.
        """
        output = ""
        data = []
        if isinstance(self._results, list):
            for result in self._results:
                if result:
                    data.extend(build_data_from_run_outputs(result))

        for datum in data:
            output = datum.data.get("text", "")

        return output

    def get_flow_name(self) -> str:
        """Returns the name of the flow.

        Returns:
            str: Flow name.
        """
        return self._flow_name

    def clear_state_and_results(self):
        """Resets the flow's completion status and clears results."""
        self._is_completed = False
        self._results = {}

    def is_completed(self) -> bool:
        """Returns whether the flow execution has completed.

        Returns:
            bool: True if completed, False otherwise.
        """
        return self._is_completed


class FlowRunner(Component):
    display_name = "Flow Runner"
    description = "Component to run multiple flows sequentially.\n Removes invalid and duplicate flows to run."
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "Omniscien"
    name = "FlowRunnerOld"
    flow_name_selected = ""
    validated_flows: list[str] = []

    class DemoClass:
        def something(self):
            return True

    demo = DemoClass()

    inputs = [
        MessageTextInput(
            name="input_value",
            display_name="Flows to Run",
            info="This is input accepts only valid flows that exist in the LangFlow server. \n Removes inputs that are invalid Flows or duplicate Flows.",
            value=["Test Flow"],
            is_list=True,
            tool_mode=True,
            real_time_refresh=True,
        ),
        DropdownInput(
            name="mode",
            display_name="On Error:",
            options=["Cascade", "Break"],
            value="Cascade",
            dynamic=True,
            advanced=True,
            info="On Error, either continue to cascade the flows or break the flows from running.",
        ),
        MessageTextInput(
            name="flow_value", display_name="Flow Input", value="", dynamic=True, real_time_refresh=True, info="."
        ),
        MessageTextInput(
            name="retries",
            display_name="Retry Count",
            value=["0"],
            dynamic=True,
            is_list=True,
            info=".",
            advanced=True,
        ),
    ]

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

    # Function to take in Flow inputs as a List and return only Flows that exist in LangFlow without duplicates
    async def validate_flows(self, flows_input: list):
        """Filters and returns only valid, unique flow names from the user input.

        Args:
        flows_input (List): List of user-specified flow names.

        Returns:
        List: Validated list of unique flow names.
        """
        validated_flows = []
        valid_flows = await self.get_flow_names()

        # If the flow input is valid or still empty
        for i, flow in enumerate(flows_input):
            is_last = i == len(flows_input) - 1

            if (flow in valid_flows) and (flow not in validated_flows):
                # if the flow input is not already in the validated_flows
                validated_flows.append(flow)
            # if flow == "" and is_last:
            #    validated_flows.append("")

        return validated_flows

    async def validate_flow(self, flow: str) -> str:
        valid_flows = await self.get_flow_names()
        return flow if flow in valid_flows else ""

    async def get_value_from_str_dict(self, str_dict: str, key: str) -> str:
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

    async def build_list_of_flows(self, flows_to_run: list, input_value) -> list:
        list_of_flows = []

        for flow_name_selected in flows_to_run:
            if flow_name_selected == "" or flow_name_selected is None:
                continue

            new_flow = Flow(flow_name_selected, input_value, self.user_id, "1")
            list_of_flows.append(new_flow)

            self.log(flow_name_selected, "Added to flows to run")

        return list_of_flows

    async def run_flows(self, flows: list):
        for flow in flows:
            if isinstance(flow, Flow):
                await flow.run_flow()
                self.log(flow.get_flow_name(), "Successful flow run")
            else:
                self.log(flow.get_flow_name(), "Error running flow")

    async def collect_results_from_flows_as_dicts(self, flows: list) -> dict[str, str]:
        results: dict[str, str] = {}

        for flow in flows:
            if isinstance(flow, Flow):
                if flow.is_completed():
                    self.log(flow.collect_result_as_string(), "String result")
                    results = merge_string_to_dict(flow.collect_result_as_string(), results)
                else:
                    self.log(flow.get_flow_name(), "Is not completed")
            else:
                self.log("Is not an instance of a Flow")

        return results

    async def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None):
        """Updates the component's config when a field value is changed (especially 'input_value').

        Args:
            build_config (dotdict): Current configuration.
            field_value (Any): New field value.
            field_name (str | None): Name of the field being updated.

        Returns:
            dotdict: Updated configuration.
        """
        # When Flows are updated, validate them
        # if field_name == "input_value":
        #     build_config["input_value"]["value"] = await self.validate_flows(build_config["input_value"]["value"])
        #     return build_config

        if field_name == "flows":
            flow_to_add = await self.get_value_from_str_dict(build_config["flows"]["value"], "name")

            if flow_to_add:
                updated_flows = build_config["input_value"]["value"]
                updated_flows.append(flow_to_add)
                build_config["input_value"]["value"] = await self.validate_flows(updated_flows)
                build_config["flows"]["value"] = ""

        return build_config

    async def build_output(self) -> Data:
        """Orchestrates the complete process of:
        - Running validated flows
        - Merging their results with the input
        - Returning a combined Data output

        Returns:
            Data: Final structured output with results of all processed flows.
        """
        flows_selected = self._attributes.get("input_value")
        self.log(flows_selected, "Flows selected")
        await self.get_flow(flows_selected[0])

        inputs = self._attributes.get("flow_value")
        self.log(inputs, "inputs")

        flows_to_run = await self.build_list_of_flows(flows_selected, inputs)
        self.log(type(flows_to_run).__name__, "Flows to run type")

        # Monitor Process
        job_id = self.graph.context["jobpayload"]["jobid"]
        self.log(job_id, "Job ID")
        job_server_url = "http://devdemo.languagestudio.com:4051/api/v1/jobinfos"
        progress = JobProgressManager(job_server_url, job_id)
        await progress.initialize_steps(flows_selected)

        # Run each flow and update progress
        for i, flow in enumerate(flows_to_run):
            step_no = i + 1
            try:
                # Set the current step index and mark it as starting
                await progress.mark_job_index(len(flows_to_run), step_no)
                await progress.update_step(step_no, 0, f"Starting {flow.get_flow_name()}...")

                # Run the actual flow
                await flow.run_flow()

                # # Incrementally update the completion percentage for the current step (test)
                # for percent in range(10, 101, 20):  # e.g., 10, 30, 50...
                #     await asyncio.sleep(1)  # Pause to allow the update to be visible
                #     status_message = f"Completed {flow.get_flow_name()}" if percent == 100 else f"Processing {flow.get_flow_name()}..."
                #     await progress.update_step(step_no, percent, status_message)
                await progress.update_step(step_no, 100, f"Completed {flow.get_flow_name()}")

            except Exception as e:
                raise e

        results = await self.collect_results_from_flows_as_dicts(flows_to_run)
        self.graph.context.jsonOutput = results

        return Data(data=results)

        # await self.run_flows(flows_to_run)
        # results = await self.collect_results_from_flows_as_dicts(flows_to_run)
        #
        # # Save JSON into Context Graph for JSON Conversion
        # self.graph.context.jsonOutput = results
        #
        # output = results
        #
        # self.log(output, "output")
        # self.log(f"Self Context Graph: {self.graph.context.get("jsonOutput")}")
        #
        # return Data(data=output)
