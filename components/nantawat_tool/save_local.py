import importlib
import json
import re
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import orjson
import pandas as pd
from fastapi import UploadFile
from fastapi.encoders import jsonable_encoder
from langchain.tools import StructuredTool
from lfx.base.langchain_utilities.model import LCToolComponent
from lfx.field_typing import Tool
from lfx.io import (
    DropdownInput,
    MultilineInput,
    Output,
)
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame
from lfx.schema.message import Message
from lfx.services.deps import get_settings_service, get_storage_service, session_scope
from lfx.utils.request_utils import get_user_agent
from loguru import logger
from pydantic import BaseModel, Field

# Constants
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_DEPTH = 1
DEFAULT_FORMAT = "HTML"


URL_REGEX = re.compile(
    r"^(https?:\/\/)?" r"(www\.)?" r"([a-zA-Z0-9.-]+)" r"(\.[a-zA-Z]{2,})?" r"(:\d+)?" r"(\/[^\s]*)?$",
    re.IGNORECASE,
)

USER_AGENT = None
# Check if langflow is installed using importlib.util.find_spec(name))
if importlib.util.find_spec("langflow"):
    langflow_installed = True
    USER_AGENT = get_user_agent()
else:
    langflow_installed = False
    USER_AGENT = "lfx"


class ListURLGather(BaseModel):
    url: list = Field(..., description="The URL to crawl")
    file_name: str = Field(..., description="The file name to save the content (extension will be added automatically)")


class SaveContent(LCToolComponent):
    display_name = "Save to Local"
    description = "Save Content to Local"
    documentation = ""
    icon = "Globe"

    LOCAL_DATA_FORMAT_CHOICES = ["csv", "excel", "json", "markdown"]
    LOCAL_MESSAGE_FORMAT_CHOICES = ["txt", "json", "markdown"]
    AWS_FORMAT_CHOICES = [
        "txt",
        "json",
        "csv",
        "xml",
        "html",
        "md",
        "yaml",
        "log",
        "tsv",
        "jsonl",
        "parquet",
        "xlsx",
        "zip",
    ]
    GDRIVE_FORMAT_CHOICES = ["txt", "json", "csv", "xlsx", "slides", "docs", "jpg", "mp3"]

    inputs = [
        MultilineInput(
            name="input_data",
            display_name="Input Data",
            info="The input data to be saved. Can be of type Data, DataFrame",
        ),
        MultilineInput(
            name="directory_name",
            display_name="Directory Name",
            info="Base Directory.",
            required=True,
            # show=False,
        ),
        MultilineInput(
            name="file_name",
            display_name="File Name",
            info="Name file will be saved as (without extension).",
            # required=True,
            # show=False,
            tool_mode=True,
        ),
        # Format inputs (dynamic based on storage location)
        DropdownInput(
            name="local_format",
            display_name="File Format",
            options=list(dict.fromkeys(LOCAL_DATA_FORMAT_CHOICES + LOCAL_MESSAGE_FORMAT_CHOICES)),
            info="Select the file format for local storage.",
            value="json",
            # show=False,
        ),
    ]

    outputs = [
        # Output(display_name="Extracted Pages", name="page_results", method="fetch_content", tool_mode=False),
        # Output(display_name="Raw Content", name="raw_results", method="fetch_content_as_message", tool_mode=False),
        Output(display_name="File Path", name="message", method="save_content"),
    ]

    def build_tool(self) -> Tool:
        return StructuredTool.from_function(
            name="save_content_to_file",
            description="Saves the crawled content to a file in the specified storage location.",
            func=self.save_content,
        )

    async def run_model(self) -> Message:
        return await self.save_content()

    ###############################################################################################################

    async def save_content(self) -> Message:
        """Save the input to a file and upload it, returning a confirmation message."""
        # Validate inputs

        logger.info("Starting to save content to file...")
        input_data = Message(text=self.input_data)
        if not self.file_name:
            self.file_name = uuid.uuid4().hex
        if not self.file_name and not self.directory_name:
            msg = "File name must be provided."
            raise ValueError(msg)
        if not self._get_input_type(input_data):
            msg = "Input type is not set."
            raise ValueError(msg)

        # Get selected storage location
        storage_location = "Local"  # or self._get_selected_storage_location()
        if not storage_location:
            msg = "Storage location must be selected."
            raise ValueError(msg)

        # Route to appropriate save method based on storage location
        if storage_location == "Local":
            return await self._save_to_local(input_data)
        msg = f"Unsupported storage location: {storage_location}"
        raise ValueError(msg)

    def _get_input_type(self, input_data) -> str:
        """Determine the input type based on the provided input."""
        # Use exact type checking (type() is) instead of isinstance() to avoid inheritance issues.
        # Since Message inherits from Data, isinstance(message, Data) would return True for Message objects,
        # causing Message inputs to be incorrectly identified as Data type.
        if type(input_data) is DataFrame:
            return "DataFrame"
        if type(input_data) is Message:
            return "Message"
        if type(input_data) is Data:
            return "Data"
        msg = f"Unsupported input type: {type(input_data)}"
        raise ValueError(msg)

    def _get_default_format(self, input_data) -> str:
        """Return the default file format based on input type."""
        if self._get_input_type(input_data) == "DataFrame":
            return "csv"
        if self._get_input_type(input_data) == "Data":
            return "json"
        if self._get_input_type(input_data) == "Message":
            return "json"
        return "json"  # Fallback

    def _adjust_file_path_with_format(self, path: Path, fmt: str) -> Path:
        """Adjust the file path to include the correct extension."""
        file_extension = path.suffix.lower().lstrip(".")
        if fmt == "excel":
            return Path(f"{path}.xlsx").expanduser() if file_extension not in ["xlsx", "xls"] else path
        return Path(f"{path}.{fmt}").expanduser() if file_extension != fmt else path

    async def _upload_file(self, file_path: Path) -> None:
        """Upload the saved file using the upload_user_file service."""
        from langflow.api.v2.files import upload_user_file
        from langflow.services.database.models.user.crud import get_user_by_id

        # Ensure the file exists
        if not file_path.exists():
            msg = f"File not found: {file_path}"
            raise FileNotFoundError(msg)

        # Upload the file
        with file_path.open("rb") as f:
            async with session_scope() as db:
                if not self.user_id:
                    msg = "User ID is required for file saving."
                    raise ValueError(msg)
                current_user = await get_user_by_id(db, self.user_id)

                await upload_user_file(
                    file=UploadFile(filename=file_path.name, file=f, size=file_path.stat().st_size),
                    session=db,
                    current_user=current_user,
                    storage_service=get_storage_service(),
                    settings_service=get_settings_service(),
                )

    def _save_dataframe(self, dataframe: DataFrame, path: Path, fmt: str) -> str:
        """Save a DataFrame to the specified file format."""
        if fmt == "csv":
            dataframe.to_csv(path, index=False)
        elif fmt == "excel":
            dataframe.to_excel(path, index=False, engine="openpyxl")
        elif fmt == "json":
            dataframe.to_json(path, orient="records", indent=2)
        elif fmt == "markdown":
            path.write_text(dataframe.to_markdown(index=False), encoding="utf-8")
        else:
            msg = f"Unsupported DataFrame format: {fmt}"
            raise ValueError(msg)
        return f"DataFrame saved successfully as '{path}'"

    def _save_data(self, data: Data, path: Path, fmt: str) -> str:
        """Save a Data object to the specified file format."""
        if fmt == "csv":
            pd.DataFrame(data.data).to_csv(path, index=False)
        elif fmt == "excel":
            pd.DataFrame(data.data).to_excel(path, index=False, engine="openpyxl")
        elif fmt == "json":
            path.write_text(
                orjson.dumps(jsonable_encoder(data.data), option=orjson.OPT_INDENT_2).decode("utf-8"), encoding="utf-8"
            )
        elif fmt == "markdown":
            path.write_text(pd.DataFrame(data.data).to_markdown(index=False), encoding="utf-8")
        else:
            msg = f"Unsupported Data format: {fmt}"
            raise ValueError(msg)
        return f"Data saved successfully as '{path}'"

    async def _save_message(self, message: Message, path: Path, fmt: str) -> str:
        """Save a Message to the specified file format, handling async iterators."""
        content = ""
        if message.text is None:
            content = ""
        elif isinstance(message.text, AsyncIterator):
            async for item in message.text:
                content += str(item) + " "
            content = content.strip()
        elif isinstance(message.text, Iterator):
            content = " ".join(str(item) for item in message.text)
        else:
            content = str(message.text)

        if fmt == "txt":
            path.write_text(content, encoding="utf-8")
        elif fmt == "json":
            path.write_text(json.dumps({"message": content}, indent=2), encoding="utf-8")
        elif fmt == "markdown":
            path.write_text(f"**Message:**\n\n{content}", encoding="utf-8")
        else:
            msg = f"Unsupported Message format: {fmt}"
            raise ValueError(msg)
        return f"Message saved successfully as '{path}'"

    # def _get_selected_storage_location(self) -> str:
    #     """Get the selected storage location from the SortableListInput."""
    #     if hasattr(self, "storage_location") and self.storage_location:
    #         if isinstance(self.storage_location, list) and len(self.storage_location) > 0:
    #             return self.storage_location[0].get("name", "")
    #         if isinstance(self.storage_location, dict):
    #             return self.storage_location.get("name", "")
    #     return ""

    def _get_file_format_for_location(self, location: str, input_data) -> str:
        """Get the appropriate file format based on storage location."""
        if location == "Local":
            return getattr(self, "local_format", None) or self._get_default_format(input_data=input_data)
        if location == "AWS":
            return getattr(self, "aws_format", "txt")
        if location == "Google Drive":
            return getattr(self, "gdrive_format", "txt")
        return self._get_default_format(input_data=input_data)

    async def _save_to_local(self, input_data) -> Message:
        """Save File content."""
        file_format = self._get_file_format_for_location("Local", input_data)

        # Validate file format based on input type
        allowed_formats = (
            self.LOCAL_MESSAGE_FORMAT_CHOICES
            if self._get_input_type(input_data) == "Message"
            else self.LOCAL_DATA_FORMAT_CHOICES
        )
        if file_format not in allowed_formats:
            msg = f"Invalid file format '{file_format}' for {self._get_input_type(input_data)}. Allowed: {allowed_formats}"
            raise ValueError(msg)

        if not self.directory_name:
            raise ValueError("Directory name must be provided for local storage.")

        # Prepare file path
        self.directory_name = (
            self.directory_name.rstrip("/") if self.directory_name.endswith("/") else self.directory_name
        ) or "."
        file_path = Path(self.directory_name + "/" + self.file_name).expanduser()
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path = self._adjust_file_path_with_format(file_path, file_format)

        # Save the input to file based on type
        if self._get_input_type(input_data) == "DataFrame":
            confirmation = self._save_dataframe(input_data, file_path, file_format)
        elif self._get_input_type(input_data) == "Data":
            confirmation = self._save_data(input_data, file_path, file_format)
        elif self._get_input_type(input_data) == "Message":
            confirmation = await self._save_message(input_data, file_path, file_format)
        else:
            msg = f"Unsupported input type: {self._get_input_type(input_data)}"
            raise ValueError(msg)

        # Upload the saved file
        await self._upload_file(file_path)

        # Return the final file path and confirmation message
        final_path = Path.cwd() / file_path if not file_path.is_absolute() else file_path
        logger.info(f"File saved successfully as '{final_path}'")
        return Message(text=f"{confirmation} at {final_path}")
