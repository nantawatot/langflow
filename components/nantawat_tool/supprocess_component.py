"""Subprocess Base Component."""

import subprocess
from pathlib import Path

from langflow.custom.custom_component.component import Component
from langflow.inputs.inputs import MessageTextInput
from langflow.io import (
    Output,
)
from langflow.schema.data import Data


def find_venv_path(start_path: Path, venv_name=".venv"):
    """Searches for a .venv folder upward from current directory."""
    current = start_path.resolve()
    for parent in [current, *list(current.parents)]:
        venv_dir = parent / venv_name
        if (venv_dir / "bin" / "python").exists():
            return venv_dir
    return None


def find_venv_path_from_file(file_path: str | Path, venv_name=".venv"):
    """Searches for a .venv folder upward from the directory of the given file."""
    file_path = Path(file_path).resolve()
    return find_venv_path(file_path.parent, venv_name)


class SubprocessComponent(Component):
    """Base class for components that run a subprocess command."""

    display_name = "SubprocessCalling"
    description = "Run a subprocess command and return the output (stdout)."
    icon = "Globe"
    name = "Sub Process Calling"

    inputs = [
        MessageTextInput(
            name="module_directory",
            display_name="Module Directory",
            required=True,
            info="The directory where the module is located."
            " This is used to set the working directory for the subprocess.",
        ),
        MessageTextInput(
            name="arguments",
            display_name="Arguments",
            is_list=True,
            info="Additional arguments to pass to the command. This can be used to customize the command execution.",
        ),
    ]
    outputs = [
        Output(display_name="Output", name="output", method="fetch_output"),
    ]
    command_history: list[str] = []

    def fetch_output(self) -> Data:
        """Run the command and return the output."""
        venv_path = find_venv_path_from_file(self.module_directory)
        if not venv_path:
            message = f"No virtual environment found for {self.module_directory}"
            raise FileNotFoundError(message)

        venv_python = venv_path / "bin" / "python"
        if not venv_python.exists():
            message = f"Python interpreter not found in {venv_python}"
            raise FileNotFoundError(message)

        arguments: list[str] = self.arguments if isinstance(self.arguments, list) else self.arguments.split()

        # Set the working directory
        module_directory = self.module_directory.strip()
        if not module_directory:
            message = "Module directory cannot be empty."
            raise ValueError(message)
        # Run the command in a subprocess
        command_list = [str(venv_python), str(self.module_directory), *arguments]
        if not isinstance(command_list, list):
            message = "Command list must be a list of strings."
            raise TypeError(message)

        try:
            result = subprocess.run(command_list, capture_output=True, text=True, check=True)  # noqa: S603
            output = result.stdout.strip()

        except subprocess.CalledProcessError as e:
            error_message = f"Command failed with error: {e.stderr.strip()}"
            raise RuntimeError(error_message) from e

        return Data(data={"output": output})
