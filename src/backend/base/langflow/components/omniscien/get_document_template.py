import json
import re

import requests

from langflow.custom import Component
from langflow.io import MessageTextInput, Output
from langflow.schema import Data


class GetDocumentTemplate(Component):
    display_name = "Get Document Template"
    description = "Use this to retrieve the template document (base64), based on the Flow Profile"
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "Omniscien"
    name = "GetDocumentTemplate"

    inputs = [
        MessageTextInput(
            name="doc_id_key",
            display_name="Doc ID Key",
            info="This is a component to retrieve a document template based on the flow ID.",
            tool_mode=True,
            advanced=True,
        ),
        MessageTextInput(
            name="flow_id_key",
            display_name="Flow ID Key",
            info="This is a component to retrieve a document template based on the flow ID.",
            tool_mode=True,
            advanced=True,
        ),
        MessageTextInput(
            name="job_payload",
            display_name="Job Payload",
            info="This is a component to retrieve a document template based on the flow ID.",
            tool_mode=True,
        ),
        MessageTextInput(
            name="flow_profile",
            display_name="Flow Profile",
            info="This is a component to retrieve a document template based on the flow ID.",
            tool_mode=True,
        ),
    ]

    outputs = [
        Output(display_name="Output", name="output", method="build_output"),
    ]

    def build_output(self) -> Data:
        # To obtain the flow ID and document ID from the input
        flow_profile = self.obtain_key_value(self.flow_profile, self.doc_id_key)
        document_id = self.obtain_document_id(flow_profile)
        flow_id = self.obtain_key_value(self.job_payload, self.flow_id_key)

        api_url = "http://devdemo.languagestudio.com:4000/lsrestapi/v6/genai/getflowdoctemplate"

        params = {
            "id": flow_id,  # self.flow_id will be the value from the input
            "doctemplateid": document_id,
        }

        response = requests.get(api_url, params=params)

        self.log(f"Response from API: {response.text}")

        if response.status_code != 200:
            self.log(f"Error: {response.status_code} - {response.text}")
            raise Exception(f"Failed to retrieve document template: {response.text}")

        data = Data(value=response.text)
        # data = Data(value=response)
        self.status = data
        return data

    def obtain_key_value(self, payload: str, key: str) -> str:
        # Extract the string from Message or use as-is
        input_str = payload.text if hasattr(payload, "text") else payload

        result_str = "Hello World"

        # First load
        data = json.loads(input_str)

        # If it's still a string, load again (double-encoded case)
        if isinstance(data, str):
            data = json.loads(data)

        # Navigate the nested keys
        keys = key.split(".")
        for k in keys:
            data = data[k]

        # Convert to string and remove unnecessary characters
        result_str = str(data).replace("[", "").replace("]", "").replace("'", "").replace('"', "")

        return result_str

    def obtain_document_id(self, payload: str) -> int:
        # Extract the string from Message or use as-is
        match = re.search(r"id: (\d+)", payload)
        if match:
            value = match.group(1)

        return value
