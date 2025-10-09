# from lfx.field_typing import Data
from lfx.custom.custom_component.component import Component
from lfx.io import MessageTextInput, Output, MultilineInput
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame
import json
from loguru import logger
import re


class ConverseDataFrame(Component):
    display_name = "Converse Json to DataFrame"
    description = "Converse Json string to DataFrame."
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "code"
    name = "JsonToDataFrame"

    inputs = [
        MultilineInput(
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

    def build_output(self) -> DataFrame:
        """Build the output data."""
        try:
            clean_json = re.sub(r"^```json\s*|\s*```$", "", self.input_value.strip(), flags=re.MULTILINE)
            data_frame = json.loads(clean_json)
            data = DataFrame(data_frame)
        except Exception as e:
            logger.error(f"Error parsing JSON: {e}")
            return DataFrame()
        return data
