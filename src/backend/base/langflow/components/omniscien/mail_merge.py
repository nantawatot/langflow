import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import requests

from langflow.custom import Component
from langflow.inputs import BoolInput, DataInput, MessageInput
from langflow.io import Output
from langflow.schema import Message


class MailMerge(Component):
    display_name = "Mail Merge Component"
    description = "Performs Mail Merge of Contract and Template"
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "Omniscien"
    name = "MailMerge"

    inputs = [
        DataInput(
            name="json_to_merge",
            display_name="JSON To Merge",
            info="This is a custom component Input",
            value="Hello, World!",
            tool_mode=True,
        ),
        MessageInput(
            name="template_path_to_merge",
            display_name="Template To Merge",
            info="This is a custom component Input",
            value="Hello, World!",
            tool_mode=True,
        ),
        MessageInput(
            name="output_file_name",
            display_name="Output File Name",
            info="This is a custom component Input",
            value="Hello, World!",
            tool_mode=True,
            advanced=True,
        ),
        BoolInput(
            name="debug_mode",
            display_name="Debug Mode",
            info="This is a custom component Input",
            value=False,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Output", name="output", method="build_output"),
    ]

    def build_output(self) -> Message:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Obtain output file path
        base_output_file_path = self.graph.context["jobpayload"]["jobprofile"]["datatoprocess"]["inputfilepath"]
        self.log(f"Base Output File Path: {base_output_file_path}")

        # Convert to Path object
        path_obj = Path(base_output_file_path)

        # Remove the last 3 parts (e.g., /input/contracts/filename)
        base_output_file_path = path_obj.parents[2]  # parents[0] = filename, [1] = 'contracts', [2] = 'input'
        self.log(f"Base Output File Path After Cut: {base_output_file_path}")

        # Define Variables
        template_path_to_merge = self.template_path_to_merge.text
        # contract_title = self.contract_title.text
        output_file_name = self.output_file_name.text
        # Prepend timestamp
        output_file_name = f"{timestamp}_{output_file_name}.docx"
        # output_path = "/opt/omniscien/tmp/tools/documentmerge/"  # self defined for now because it is the path on the server which may not be ideal for user to prompt this
        base_path = base_output_file_path.text if hasattr(base_output_file_path, "text") else base_output_file_path
        output_file_path = os.path.join(base_path, "output", "contracts")
        log_file_path = "/opt/omniscien/tmp/tools/documentmerge/contract_summary.log.json"  # self defined for now because it is the path on the server which may not be ideal for user to prompt this

        ############### DEBUG ################
        if self.debug_mode:
            print("Output File Name: ", output_file_name, flush=True)

        # Define where to save the JSON
        json_output_path = os.path.join(
            "/opt/omniscien/lsev6/enterprise/tmp/tools/documentmerge/", "zone_issue_extra_space.json"
        )

        # Ensure the directory exists
        os.makedirs("/opt/omniscien/lsev6/enterprise/tmp/tools/documentmerge/", exist_ok=True)

        # Write json_to_merge to file
        with open(json_output_path, "w", encoding="utf-8") as json_file:
            json.dump(self.json_to_merge, json_file, ensure_ascii=False, indent=4)

        self.log(f"JSON written to: {json_output_path}")

        # Specify the URL you want to send the request to [API to call jsontodoc]
        api_url = "http://devdemo.languagestudio.com:4000/lsrestapi/v6/mailmerge/jsontodoc"

        # Serialize JSON to a string
        self.log(f"json_to_merge: {self.json_to_merge}")

        json_string = json.dumps(self.json_to_merge)
        self.log(f"json_string: {json_string}")

        # Encode the string to bytes
        bytes_data = json_string.encode("utf-8")
        # bytes_data = json_string.encode()
        # self.log(f"json_to_merge: {self.json_to_merge}")
        self.log(f"bytes_data: {bytes_data}")

        # Define the form data as a dictionary
        form_data = {
            "outputfilename": output_file_name,
            "outputpath": output_file_path,
            "log": str(1),
            "logfilepath": log_file_path,
        }

        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp_json_file:
            tmp_json_file.write(json_string)
            # print("INSIDE TEMP FILE: ",json_string)
            tmp_json_file_path = tmp_json_file.name  # Get path to temp file

        # Open files and send request
        with open(template_path_to_merge, "rb") as template_file, open(tmp_json_file_path, "rb") as json_file_temp:
            files = {
                "jsonfile": ("contract-legal-data.json", bytes_data, "application/json"),
                "templatefile": (
                    "contract-legal-template-original.docx",
                    template_file,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ),
            }

            response = requests.post(api_url, data=form_data, files=files)
            self.log(f"Response Status Code: {response.status_code}")
            self.log(f"Response Body: {response.text}")

        #     ####################### DEBUG #######################
        #     if self.debug_mode:
        #         print("Status Code:", response.status_code, flush=True)
        #         print("Response Body:", response.text, flush=True)
        #         print("Response JSON:", response.json(), flush=True)

        #     if response.status_code == 200:
        #         print("Request Successful! Document has been Successfully Merged", flush=True)

        #         result_json = response.json()

        # data = Data(data=result_json)
        # data = Data(data=self.template_path_to_merge)

        full_output_path = os.path.join(output_file_path, output_file_name)
        return Message(text=full_output_path)
