import base64
import json
import os

from langflow.custom import Component
from langflow.inputs import DataInput
from langflow.io import Output
from langflow.schema import Data, Message


class ConvertIntoDocx(Component):
    display_name = "Convert into Word Doc (.docx)"
    description = "Use this to decode a Base64 word document and transform into a .docx file "
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "Omniscien"
    name = "CobvertIntoDocx"

    inputs = [
        DataInput(
            name="input_value",
            display_name="Input Value",
            info="This is a custom component Input",
            value="Hello, World!",
            tool_mode=True,
        ),
    ]

    outputs = [
        Output(display_name="Output", name="output", method="build_output"),
    ]

    def build_output(self) -> Message:
        if isinstance(self.input_value, Data):
            json_string_from_input = self.input_value.value
        else:
            json_string_from_input = self.input_value

        try:
            # Step 1: Parse the outer JSON
            outer_data = json.loads(json_string_from_input)

            self.log(f"Successfully parsed outer JSON: {outer_data}")

            # The 'value' field is itself a JSON string that needs to be parsed
            inner_json_string = outer_data["result"][0]["doctemplate"]
            # inner_data = json.loads(inner_json_string)
            self.log(f"Successfully parsed inner JSON: {inner_json_string}")

            # Step 2: Extract the doctemplate payload
            # The 'result' is a list, and the doctemplate is inside the first item
            doctemplate_base64 = inner_json_string
            print("Successfully extracted Base64 string.")
            self.log("Successfully extracted Base64 string.")

            # Step 3: Base64 Decode the doctemplate string
            # It's important to convert the string to bytes before decoding
            decoded_bytes = base64.b64decode(doctemplate_base64)
            print("Successfully Base64 decoded the string.")
            self.log("Successfully Base64 decoded the string.")

            # Step 4: Save the decoded bytes as a .docx file
            output_filename = "output_document.docx"
            with open(output_filename, "wb") as f:
                f.write(decoded_bytes)

            print(f"Word document saved successfully as '{output_filename}'")
            self.log(f"Word document saved successfully as '{output_filename}'")

            current_directory = os.getcwd()

            # Log Path being saved to
            path = os.path.join(current_directory, output_filename)  # Return the full path

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            self.log(f"Error decoding JSON: {e}")
        except KeyError as e:
            print(f"Key not found in JSON payload: {e}")
            self.log(f"Key not found in JSON payload: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            self.log(f"An unexpected error occurred: {e}")

        data = Data(value=self.input_value)
        self.status = data
        return Message(text=path)
