"""Route Name Analyzer Component."""

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


class MapRoute(Component):
    """Component for analyzing route names based on city and country/state information."""

    display_name = "Route Name Analyzer"
    description = (
        "Get Route Name Along the City Name. City Name must be in the "
        "format of 'City, Country' or 'City, State, Country'. "
        "Example usage: "
        "'Bangkok, Thailand' 'Chiang Mai, Thailand' will return 'Bangkok' to 'Chiang Mai' Route Name. "
        "This is useful for analyzing route names in a given context."
        "'Deinze, Flanders, Belgium' 'Gavere, Flanders, Belgium' 'Velzeke, Belgium' "
        "will return 'Deinze' to 'Gavere' to 'Velzeke' Route Name. "
        "This is useful for analyzing route names in a given context."
    )
    icon = "Globe"
    name = "RouteNameAnalyzer"

    inputs = [
        MessageTextInput(
            name="module_directory",
            display_name="Module Directory",
            required=True,
            info="The directory where the module is located. "
            "This is used to set the working directory for the subprocess.",
        ),
        MessageTextInput(
            name="name_search",
            display_name="Search Name",
            info="The name of something that want to search for. "
            "example: 'Bangkok Thailand' 'Chiang Mai, Thailand'."
            " This will be used to find the route name and analyze it.",
            tool_mode=True,
            is_list=True,
        ),
    ]
    outputs = [
        Output(display_name="Output", name="output", method="get_route"),
    ]

    def get_route(self) -> Data:
        """Run the command and return the output.

        This method finds the virtual environment, constructs the command to run, and captures the output.
        """
        venv_path = find_venv_path_from_file(self.module_directory)
        if not venv_path:
            message = f"No virtual environment found for {self.module_directory}"
            raise FileNotFoundError(message)

        venv_python = venv_path / "bin" / "python"
        if not venv_python.exists():
            message = f"Python interpreter not found in {venv_python}"
            raise FileNotFoundError(message)

        # Prepare the command arguments
        if isinstance(self.name_search, list):
            arguments: list[str] = ["--query", *self.name_search]
        elif isinstance(self.name_search, str):
            arguments: list[str] = ["--query", self.name_search]
        else:
            message = "name_search must be a string or a list of strings."
            raise TypeError(message)

        # arguments: list[str] = ["--query"] + self.name_search

        # Set the working directory
        module_directory = self.module_directory.strip()
        if not module_directory:
            message = "Module directory cannot be empty."
            raise ValueError(message)
        command_list = [str(venv_python), str(self.module_directory), *arguments]
        if not isinstance(command_list, list):
            message = "Command list must be a list of strings."
            raise TypeError(message)

        try:
            # Run the command in a subprocess
            command_list = " ".join(command_list).split()
            result = subprocess.run(command_list, capture_output=True, text=True, check=True)  # noqa: S603
            output = result.stdout.strip()

        except subprocess.CalledProcessError as e:
            error_message = f"Command failed with error: {e.stderr.strip()}"
            raise RuntimeError(error_message) from e

        return Data(data={"output": output})
