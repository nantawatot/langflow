import json

from langflow.custom.custom_component.component import Component
from langflow.io import MessageTextInput, Output
from langflow.schema.data import Data  # Data is used for status/error handling

# The Message return type hint is kept as per your original code structure,
# though the function returns a string or Data object.


class AppendToArray(Component):
    """Appends a new object (constructed from a key-value pair) to a specified
    top-level array within a JSON object.
    """

    display_name = "Append To Array"
    description = "Appends a key-value pair to a specified array within the JSON."
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "Omniscien"
    name = "AppendToArray"

    inputs = [
        MessageTextInput(
            name="json_input",
            display_name="JSON Input",
            info="The JSON string containing the array to be modified.",
            # Example input provided for default visualization
            value='{"contract_obligations": [{"party_name": "Client", "obligations": ["..."]}]}',
            tool_mode=True,
        ),
        MessageTextInput(
            name="array_key",
            display_name="Array Key Name",
            info="The top-level key whose value is an array (e.g., 'contract_obligations').",
            value="contract_obligations",
            tool_mode=True,
        ),
        MessageTextInput(
            name="key_to_add",
            display_name="New Object Key",
            info="The key for the new object to be appended (e.g., 'list_style').",
            value="list_style",
            tool_mode=True,
        ),
        MessageTextInput(
            name="value_to_add",
            display_name="New Object Value",
            info="The value for the new object's key (e.g., 'double_side_bullet').",
            value="double_side_bullet",
            tool_mode=True,
        ),
    ]

    outputs = [
        Output(display_name="Modified JSON Output", name="output", method="build_output"),
    ]

    # Note: Returning `str` or `Data` is more precise for LangFlow output
    def build_output(self) -> Data:
        """Parses the JSON, appends a new object to the specified array, and returns the modified JSON string."""
        try:
            # 1. Get and parse inputs
            json_string = self.json_input
            array_key = self.array_key
            new_key = self.key_to_add
            new_value = self.value_to_add

            data_obj = json.loads(json_string)

            # 2. Validate and access the target array
            if array_key not in data_obj:
                raise ValueError(f"Key '{array_key}' not found in the JSON object.")

            target_array = data_obj[array_key]

            if not isinstance(target_array, list):
                raise TypeError(f"Value for key '{array_key}' is not a list/array.")

            # 3. Create the new object to append
            new_object = {new_key: new_value}

            # 4. Append the new object
            target_array.append(new_object)

            self.log(f"Successfully appended object '{new_object}' to array '{array_key}'.")

            # 5. Serialize the modified object back into a JSON string
            # Use indent=2 for clean, readable output
            modified_json_string = json.dumps(data_obj, indent=2)

            # 6. Set status and return the string (a valid LangFlow output type)
            self.status = Data(text=modified_json_string)
            return modified_json_string

        except json.JSONDecodeError:
            error_message = "Error: The 'JSON Input' is not a valid JSON string."
            self.log(error_message)
            error_data = Data(text=error_message, value={"original_input": self.json_input})
            self.status = error_data
            return error_data

        except (ValueError, TypeError) as e:
            error_message = f"Array modification error: {e}"
            self.log(error_message)
            # For flow control/error display, return Data containing the error
            error_data = Data(text=error_message)
            self.status = error_data
            return error_data

        except Exception as e:
            self.log(f"An unexpected error occurred: {e}")
            raise
