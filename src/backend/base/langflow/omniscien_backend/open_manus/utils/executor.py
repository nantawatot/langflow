import os
import subprocess

from langflow.omniscien_backend.open_manus.utils.path_resolvers import open_path_configs


class openmanus:
    def __init__(self, prompt: str):
        self.prompt = prompt
        self._output_file = "output.txt"
        self.paths = open_path_configs("/app/src/backend/base/langflow/omniscien_backend/open_manus/utils/paths.toml")
        self.call_to_execute = None
        self.max_steps = 5

    def _get_python_path(self):
        return self.paths.get("python")

    def _get_main_path(self):
        return self.paths.get("main")

    def _get_result_path(self):
        return self.paths.get("result")

    def _define_base_prompt(self, base_prompt: str = None):
        if base_prompt is None:
            pass

    def set_output_file_as(self, output_file: str):
        self._output_file = output_file

    def set_max_steps(self, max_steps: int):
        self.max_steps = max_steps

    def run(self):
        python_path = self._get_python_path()
        main_path = self._get_main_path()

        subprocess.run(
            [
                python_path,
                main_path,
                "--prompt",
                f"Output the file as {self._output_file}. The question is {self.prompt}",
                "--max_steps",
                str(self.max_steps),
            ],
            check=False,
        )

    def get_output_file_path(self):
        return self._get_result_path() + self._output_file

    def get_output_content(self):
        try:
            with open(self.get_output_file_path()) as f:
                return f.read()
        except FileNotFoundError:
            return "Failed to produce output. Please increase the max research steps and try again."

    def clean_up(self):
        try:
            os.remove(self.get_output_file_path())
        except FileNotFoundError:
            pass


# myanus= openmanus("What is the size of a banana?")
# myanus.set_max_steps(10)
# myanus.set_output_file_as("Banana.md")
# myanus.run()

# # Read the output file from the get_output_file_path
# output = myanus.get_output_content()
# print("Output:", output)
# myanus.clean_up()
