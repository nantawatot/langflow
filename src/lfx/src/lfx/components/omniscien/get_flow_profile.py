import requests
from langflow.custom import Component
from langflow.io import MessageTextInput, Output
from langflow.schema import Message


class GetFlowProfile(Component):
    display_name = "Get Flow Profile"
    description = "Use API to retrieve the Flow Profile"
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "Omniscien"
    name = "GetFlowProfile"

    inputs = [
        MessageTextInput(
            name="flow_id",
            display_name="Flow ID",
            info="This is a component to retrieve a document template based on the flow ID.",
            tool_mode=True,
        ),
    ]

    outputs = [
        Output(display_name="Output", name="output", method="build_output"),
    ]

    def build_output(self) -> Message:
        api_url = "http://devdemo.languagestudio.com:4000/lsrestapi/v6/genai/getflowprofile"

        params = {
            "id": self.flow_id,  # self.flow_id will be the value from the input
            "detail": 1,
        }

        response = requests.get(api_url, params=params)

        self.log(f"Response from API: {response.text}")

        if response.status_code != 200:
            self.log(f"Error: {response.status_code} - {response.text}")
            raise Exception(f"Failed to retrieve document template: {response.text}")

        # self.log(f"Response from API: {response.text}")

        data = Message(value=response.text)
        self.status = data
        return Message(text=response.text)
