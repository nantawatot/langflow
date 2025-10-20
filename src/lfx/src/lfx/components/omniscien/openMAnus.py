from langflow.custom import Component
from langflow.inputs.inputs import BoolInput, DropdownInput, IntInput, MessageTextInput, MultilineInput
from langflow.omniscien_backend.open_manus.utils.executor import openmanus
from langflow.schema import Data, Message
from langflow.template import Output


class OpenManus(Component):
    display_name = "Deep Research"
    description = "Deep Research"
    documentation: str = "https://docs.langflow.org/components-custom-components"
    name = "deep_research"
    icon = "Omniscien"

    inputs = [
        MultilineInput(
            name="input",
            display_name="Query:",
            info="",
            value="",
            real_time_refresh=True,
            advanced=False,
        ),
        IntInput(
            name="max_steps",
            display_name="Max Research Steps",
            info="Affects output if research depth is not sufficient for scope.",
            value=5,
        ),
        MessageTextInput(name="filename", display_name="Output File Name", info="", value="sample"),
        DropdownInput(
            name="output_type", display_name="Output Type", value="txt", options=["Text", "Markdown", "Webpage"]
        ),
        BoolInput(
            name="cleanup",
            info="Set this to false if you'd like to iteratively improve the output (reuse current output)",
            value=False,
        ),
    ]

    _ext_map = {
        "Text": ".txt",
        "Markdown": ".md",
        "Webpage": ".html",
    }

    outputs = [Output(display_name="Output", name="output", method="build_output")]

    def map_output_type(self) -> str:
        return self._ext_map.get(self.output_type, ".txt")

    async def build_output(self) -> Data:
        manus = openmanus(prompt=self._attributes.get("input"))
        manus.set_max_steps(self._attributes.get("max_steps"))
        self.log(self._attributes.get("filename"))
        filename = self._attributes.get("filename") + self.map_output_type()
        self.log(filename)
        manus.set_output_file_as(filename)
        manus.run()

        output_content = manus.get_output_content()
        if cleanup := self._attributes.get("cleanup"):
            manus.clean_up()

        return Message(text=output_content)
