import importlib
import json
import re
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import orjson
import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import UploadFile
from fastapi.encoders import jsonable_encoder
from langchain.tools import StructuredTool
from langchain_community.document_loaders import RecursiveUrlLoader
from lfx.base.langchain_utilities.model import LCToolComponent
from lfx.field_typing import Tool
from lfx.field_typing.range_spec import RangeSpec
from lfx.helpers.data import safe_convert
from lfx.inputs import SortableListInput
from lfx.io import (
    BoolInput,
    DropdownInput,
    IntInput,
    MessageTextInput,
    Output,
    SecretStrInput,
    SliderInput,
    StrInput,
    TableInput,
)
from lfx.log.logger import logger
from lfx.schema.data import Data  # , DataFrame, Message
from lfx.schema.dataframe import DataFrame
from lfx.schema.message import Message
from lfx.services.deps import get_settings_service, get_storage_service, session_scope
from lfx.utils.request_utils import get_user_agent
from pydantic import BaseModel, Field

# Constants
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_DEPTH = 1
DEFAULT_FORMAT = "Text"


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


class GatheringComponent(LCToolComponent):
    display_name = "Gathering Component"
    description = "Crawling web to gather information"
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
        MessageTextInput(
            name="urls",
            display_name="URLs",
            info="Enter one or more URLs to crawl recursively, by clicking the '+' button.",
            is_list=True,
            tool_mode=True,
            placeholder="Enter a URL...",
            list_add_label="Add URL",
            input_types=["list"],
        ),
        SliderInput(
            name="max_depth",
            display_name="Depth",
            info=(
                "Controls how many 'clicks' away from the initial page the crawler will go:\n"
                "- depth 1: only the initial page\n"
                "- depth 2: initial page + all pages linked directly from it\n"
                "- depth 3: initial page + direct links + links found on those direct link pages\n"
                "Note: This is about link traversal, not URL path depth."
            ),
            value=DEFAULT_MAX_DEPTH,
            range_spec=RangeSpec(min=1, max=5, step=1),
            required=False,
            min_label=" ",
            max_label=" ",
            min_label_icon="None",
            max_label_icon="None",
            # slider_input=True
        ),
        BoolInput(
            name="prevent_outside",
            display_name="Prevent Outside",
            info=(
                "If enabled, only crawls URLs within the same domain as the root URL. "
                "This helps prevent the crawler from going to external websites."
            ),
            value=True,
            required=False,
            advanced=True,
        ),
        BoolInput(
            name="use_async",
            display_name="Use Async",
            info=(
                "If enabled, uses asynchronous loading which can be significantly faster "
                "but might use more system resources."
            ),
            value=True,
            required=False,
            advanced=True,
        ),
        DropdownInput(
            name="format",
            display_name="Output Format",
            info="Output Format. Use 'Text' to extract the text from the HTML or 'HTML' for the raw HTML content.",
            options=["Text", "HTML"],
            value=DEFAULT_FORMAT,
            advanced=True,
        ),
        IntInput(
            name="timeout",
            display_name="Timeout",
            info="Timeout for the request in seconds.",
            value=DEFAULT_TIMEOUT,
            required=False,
            advanced=True,
        ),
        TableInput(
            name="headers",
            display_name="Headers",
            info="The headers to send with the request",
            table_schema=[
                {
                    "name": "key",
                    "display_name": "Header",
                    "type": "str",
                    "description": "Header name",
                },
                {
                    "name": "value",
                    "display_name": "Value",
                    "type": "str",
                    "description": "Header value",
                },
            ],
            value=[{"key": "User-Agent", "value": USER_AGENT}],
            advanced=True,
            input_types=["DataFrame"],
        ),
        BoolInput(
            name="filter_text_html",
            display_name="Filter Text/HTML",
            info="If enabled, filters out text/css content type from the results.",
            value=True,
            required=False,
            advanced=True,
        ),
        BoolInput(
            name="continue_on_failure",
            display_name="Continue on Failure",
            info="If enabled, continues crawling even if some requests fail.",
            value=True,
            required=False,
            advanced=True,
        ),
        BoolInput(
            name="check_response_status",
            display_name="Check Response Status",
            info="If enabled, checks the response status of the request.",
            value=False,
            required=False,
            advanced=True,
        ),
        BoolInput(
            name="autoset_encoding",
            display_name="Autoset Encoding",
            info="If enabled, automatically sets the encoding of the request.",
            value=True,
            required=False,
            advanced=True,
        ),
        SortableListInput(
            name="storage_location",
            display_name="Storage Location",
            placeholder="Select Location",
            info="Choose where to save the file.",
            options=[
                {"name": "Local", "icon": "hard-drive"},
                {"name": "AWS", "icon": "Amazon"},
                {"name": "Google Drive", "icon": "google"},
            ],
            real_time_refresh=True,
            limit=1,
        ),
        # Common inputs
        # HandleInput(
        #     name="input",
        #     display_name="File Content",
        #     info="The input to save.",
        #     dynamic=True,
        #     input_types=["Data", "DataFrame", "Message"],
        #     required=True,
        # ),
        StrInput(
            name="directory_name",
            display_name="Directory Name",
            info="Base Directory.",
            required=True,
            show=False,
            # tool_mode=True,
        ),
        StrInput(
            name="file_name",
            display_name="File Name",
            info="Name file will be saved as (without extension).",
            required=True,
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
            show=False,
        ),
        DropdownInput(
            name="aws_format",
            display_name="File Format",
            options=AWS_FORMAT_CHOICES,
            info="Select the file format for AWS S3 storage.",
            value="txt",
            show=False,
        ),
        DropdownInput(
            name="gdrive_format",
            display_name="File Format",
            options=GDRIVE_FORMAT_CHOICES,
            info="Select the file format for Google Drive storage.",
            value="txt",
            show=False,
        ),
        # AWS S3 specific inputs
        SecretStrInput(
            name="aws_access_key_id",
            display_name="AWS Access Key ID",
            info="AWS Access key ID.",
            show=False,
            advanced=True,
        ),
        SecretStrInput(
            name="aws_secret_access_key",
            display_name="AWS Secret Key",
            info="AWS Secret Key.",
            show=False,
            advanced=True,
        ),
        StrInput(
            name="bucket_name",
            display_name="S3 Bucket Name",
            info="Enter the name of the S3 bucket.",
            show=False,
            advanced=True,
        ),
        StrInput(
            name="aws_region",
            display_name="AWS Region",
            info="AWS region (e.g., us-east-1, eu-west-1).",
            show=False,
            advanced=True,
        ),
        StrInput(
            name="s3_prefix",
            display_name="S3 Prefix",
            info="Prefix for all files in S3.",
            show=False,
            advanced=True,
        ),
        # Google Drive specific inputs
        SecretStrInput(
            name="service_account_key",
            display_name="GCP Credentials Secret Key",
            info="Your Google Cloud Platform service account JSON key as a secret string (complete JSON content).",
            show=False,
            advanced=True,
        ),
        StrInput(
            name="folder_id",
            display_name="Google Drive Folder ID",
            info=(
                "The Google Drive folder ID where the file will be uploaded. "
                "The folder must be shared with the service account email."
            ),
            show=False,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Extracted Pages", name="page_results", method="fetch_content", tool_mode=False),
        Output(display_name="Raw Content", name="raw_results", method="fetch_content_as_message", tool_mode=False),
        Output(display_name="File Path", name="message", method="save_to_file"),
    ]

    def build_tool(self) -> Tool:
        return StructuredTool.from_function(
            name="save_content_to_file",
            description="Saves the crawled content to a file in the specified storage location.",
            func=self.save_to_file,
            args_schema=ListURLGather,
        )

    async def run_model(self) -> Message:
        return await self.save_to_file()

    @staticmethod
    def validate_url(url: str) -> bool:
        """Validates if the given string matches URL pattern.

        Args:
            url: The URL string to validate

        Returns:
            bool: True if the URL is valid, False otherwise
        """
        return bool(URL_REGEX.match(url))

    def ensure_url(self, url: str) -> str:
        """Ensures the given string is a valid URL.

        Args:
            url: The URL string to validate and normalize

        Returns:
            str: The normalized URL

        Raises:
            ValueError: If the URL is invalid
        """
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        if not self.validate_url(url):
            msg = f"Invalid URL: {url}"
            raise ValueError(msg)

        return url

    def _create_loader(self, url: str) -> RecursiveUrlLoader:
        """Creates a RecursiveUrlLoader instance with the configured settings.

        Args:
            url: The URL to load

        Returns:
            RecursiveUrlLoader: Configured loader instance
        """
        headers_dict = {header["key"]: header["value"] for header in self.headers if header["value"] is not None}
        extractor = (lambda x: x) if self.format == "HTML" else (lambda x: BeautifulSoup(x, "lxml").get_text())

        return RecursiveUrlLoader(
            url=url,
            max_depth=self.max_depth,
            prevent_outside=self.prevent_outside,
            use_async=self.use_async,
            extractor=extractor,
            timeout=self.timeout,
            headers=headers_dict,
            check_response_status=self.check_response_status,
            continue_on_failure=self.continue_on_failure,
            base_url=url,  # Add base_url to ensure consistent domain crawling
            autoset_encoding=self.autoset_encoding,  # Enable automatic encoding detection
            exclude_dirs=[],  # Allow customization of excluded directories
            link_regex=None,  # Allow customization of link filtering
        )

    async def fetch_url_contents(self) -> list[dict]:
        """Load documents from the configured URLs.

        Returns:
            List[Data]: List of Data objects containing the fetched content

        Raises:
            ValueError: If no valid URLs are provided or if there's an error loading documents
        """
        logger.info("Starting to fetch URL contents...")
        try:
            urls = list({self.ensure_url(url) for url in self.urls if url.strip()})
            logger.debug(f"URLs: {urls}")
            if not urls:
                msg = "No valid URLs provided."
                raise ValueError(msg)

            all_docs = []
            for url in urls:
                logger.debug(f"Loading documents from {url}")

                try:
                    loader = self._create_loader(url)
                    if self.use_async:
                        docs = await loader.aload()
                    else:
                        docs = loader.load()
                    # docs = loader.load()

                    if not docs:
                        logger.warning(f"No documents found for {url}")
                        continue

                    logger.debug(f"Found {len(docs)} documents from {url}")
                    all_docs.extend(docs)

                except requests.exceptions.RequestException as e:
                    logger.exception(f"Error loading documents from {url}: {e}")
                    continue

            if not all_docs:
                msg = "No documents were successfully loaded from any URL"
                raise ValueError(msg)

            # data = [Data(text=doc.page_content, **doc.metadata) for doc in all_docs]
            data = [
                {
                    "text": safe_convert(doc.page_content, clean_data=True),
                    "url": doc.metadata.get("source", ""),
                    "title": doc.metadata.get("title", ""),
                    "description": doc.metadata.get("description", ""),
                    "content_type": doc.metadata.get("content_type", ""),
                    "language": doc.metadata.get("language", ""),
                }
                for doc in all_docs
            ]
        except Exception as e:
            error_msg = e.message if hasattr(e, "message") else e
            msg = f"Error loading documents: {error_msg!s}"
            logger.info(msg, exc_info=True)
            logger.exception(msg)
            raise ValueError(msg) from e
        return data

    async def fetch_content(self) -> DataFrame:
        """Convert the documents to a DataFrame."""
        return DataFrame(data=await self.fetch_url_contents())

    async def fetch_content_as_message(self) -> Message:
        """Convert the documents to a Message."""
        url_contents = await self.fetch_url_contents()
        return Message(text="\n\n".join([x["text"] for x in url_contents]), data={"data": url_contents})

    ###############################################################################################################
    def update_build_config(self, build_config, field_value, field_name=None):
        """Update build configuration to show/hide fields based on storage location selection."""
        if field_name != "storage_location":
            return build_config

        # Extract selected storage location
        selected = [location["name"] for location in field_value] if isinstance(field_value, list) else []

        # Hide all dynamic fields first
        dynamic_fields = [
            "directory_name",
            "file_name",  # Common fields (input is always visible)
            "local_format",
            "aws_format",
            "gdrive_format",
            "aws_access_key_id",
            "aws_secret_access_key",
            "bucket_name",
            "aws_region",
            "s3_prefix",
            "service_account_key",
            "folder_id",
        ]

        for f_name in dynamic_fields:
            if f_name in build_config:
                build_config[f_name]["show"] = False

        # Show fields based on selected storage location
        if len(selected) == 1:
            location = selected[0]

            # Show file_name when any storage location is selected (input is always visible)
            if "file_name" in build_config:
                build_config["file_name"]["show"] = True

            if location == "Local":
                build_config["directory_name"]["show"] = True
                if "local_format" in build_config:
                    build_config["local_format"]["show"] = True

            elif location == "AWS":
                aws_fields = [
                    "aws_format",
                    "aws_access_key_id",
                    "aws_secret_access_key",
                    "bucket_name",
                    "aws_region",
                    "s3_prefix",
                ]
                for f_name in aws_fields:
                    if f_name in build_config:
                        build_config[f_name]["show"] = True

            elif location == "Google Drive":
                gdrive_fields = ["gdrive_format", "service_account_key", "folder_id"]
                for f_name in gdrive_fields:
                    if f_name in build_config:
                        build_config[f_name]["show"] = True

        return build_config

    async def save_to_file(self) -> Message:
        """Save the input to a file and upload it, returning a confirmation message."""
        # Validate inputs
        input_data = DataFrame(data=await self.fetch_url_contents())

        if not self.file_name and not self.directory_name:
            msg = "File name must be provided."
            raise ValueError(msg)
        if not self._get_input_type(input_data):
            msg = "Input type is not set."
            raise ValueError(msg)

        # Get selected storage location
        storage_location = self._get_selected_storage_location()
        if not storage_location:
            msg = "Storage location must be selected."
            raise ValueError(msg)

        # Route to appropriate save method based on storage location
        if storage_location == "Local":
            return await self._save_to_local(input_data)
        if storage_location == "AWS":
            return await self._save_to_aws(input_data)
        if storage_location == "Google Drive":
            return await self._save_to_google_drive(input_data)
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

    def _get_selected_storage_location(self) -> str:
        """Get the selected storage location from the SortableListInput."""
        if hasattr(self, "storage_location") and self.storage_location:
            if isinstance(self.storage_location, list) and len(self.storage_location) > 0:
                return self.storage_location[0].get("name", "")
            if isinstance(self.storage_location, dict):
                return self.storage_location.get("name", "")
        return ""

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
        """Save file to local storage (original functionality)."""
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
        return Message(text=f"{confirmation} at {final_path}")

    async def _save_to_aws(self, input_data) -> Message:
        """Save file to AWS S3 using S3 functionality."""
        # Validate AWS credentials
        if not getattr(self, "aws_access_key_id", None):
            msg = "AWS Access Key ID is required for S3 storage"
            raise ValueError(msg)
        if not getattr(self, "aws_secret_access_key", None):
            msg = "AWS Secret Key is required for S3 storage"
            raise ValueError(msg)
        if not getattr(self, "bucket_name", None):
            msg = "S3 Bucket Name is required for S3 storage"
            raise ValueError(msg)

        # Use S3 upload functionality
        try:
            import boto3
        except ImportError as e:
            msg = "boto3 is not installed. Please install it using `uv pip install boto3`."
            raise ImportError(msg) from e

        # Create S3 client
        client_config = {
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
        }

        if hasattr(self, "aws_region") and self.aws_region:
            client_config["region_name"] = self.aws_region

        s3_client = boto3.client("s3", **client_config)

        # Extract content
        content = self._extract_content_for_upload()
        file_format = self._get_file_format_for_location("AWS", input_data)

        # Generate file path
        file_path = f"{self.file_name}.{file_format}"
        if hasattr(self, "s3_prefix") and self.s3_prefix:
            file_path = f"{self.s3_prefix.rstrip('/')}/{file_path}"

        # Create temporary file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=f".{file_format}", delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Upload to S3
            s3_client.upload_file(temp_file_path, self.bucket_name, file_path)
            s3_url = f"s3://{self.bucket_name}/{file_path}"
            return Message(text=f"File successfully uploaded to {s3_url}")
        finally:
            # Clean up temp file
            if Path(temp_file_path).exists():
                Path(temp_file_path).unlink()

    async def _save_to_google_drive(self, input_data) -> Message:
        """Save file to Google Drive using Google Drive functionality."""
        # Validate Google Drive credentials
        if not getattr(self, "service_account_key", None):
            msg = "GCP Credentials Secret Key is required for Google Drive storage"
            raise ValueError(msg)
        if not getattr(self, "folder_id", None):
            msg = "Google Drive Folder ID is required for Google Drive storage"
            raise ValueError(msg)

        # Use Google Drive upload functionality
        try:
            import json
            import tempfile

            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
        except ImportError as e:
            msg = "Google API client libraries are not installed. Please install them."
            raise ImportError(msg) from e

        # Parse credentials
        try:
            credentials_dict = json.loads(self.service_account_key)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON in service account key: {e!s}"
            raise ValueError(msg) from e

        # Create Google Drive service
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict, scopes=["https://www.googleapis.com/auth/drive.file"]
        )
        drive_service = build("drive", "v3", credentials=credentials)

        # Extract content and format
        content = self._extract_content_for_upload(input_data)
        file_format = self._get_file_format_for_location("Google Drive", input_data)

        # Handle special Google Drive formats
        if file_format in ["slides", "docs"]:
            return await self._save_to_google_apps(drive_service, content, file_format)

        # Create temporary file
        file_path = f"{self.file_name}.{file_format}"
        with tempfile.NamedTemporaryFile(mode="w", suffix=f".{file_format}", delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Upload to Google Drive
            file_metadata = {"name": file_path, "parents": [self.folder_id]}
            media = MediaFileUpload(temp_file_path, resumable=True)

            uploaded_file = drive_service.files().create(body=file_metadata, media_body=media, fields="id").execute()

            file_id = uploaded_file.get("id")
            file_url = f"https://drive.google.com/file/d/{file_id}/view"
            return Message(text=f"File successfully uploaded to Google Drive: {file_url}")
        finally:
            # Clean up temp file
            if Path(temp_file_path).exists():
                Path(temp_file_path).unlink()

    async def _save_to_google_apps(self, drive_service, content: str, app_type: str) -> Message:
        """Save content to Google Apps (Slides or Docs)."""
        import time

        if app_type == "slides":
            from googleapiclient.discovery import build

            slides_service = build("slides", "v1", credentials=drive_service._http.credentials)

            file_metadata = {
                "name": self.file_name,
                "mimeType": "application/vnd.google-apps.presentation",
                "parents": [self.folder_id],
            }

            created_file = drive_service.files().create(body=file_metadata, fields="id").execute()
            presentation_id = created_file["id"]

            time.sleep(2)  # Wait for file to be available  # noqa: ASYNC251

            presentation = slides_service.presentations().get(presentationId=presentation_id).execute()
            slide_id = presentation["slides"][0]["objectId"]

            # Add content to slide
            requests = [
                {
                    "createShape": {
                        "objectId": "TextBox_01",
                        "shapeType": "TEXT_BOX",
                        "elementProperties": {
                            "pageObjectId": slide_id,
                            "size": {
                                "height": {"magnitude": 3000000, "unit": "EMU"},
                                "width": {"magnitude": 6000000, "unit": "EMU"},
                            },
                            "transform": {
                                "scaleX": 1,
                                "scaleY": 1,
                                "translateX": 1000000,
                                "translateY": 1000000,
                                "unit": "EMU",
                            },
                        },
                    }
                },
                {"insertText": {"objectId": "TextBox_01", "insertionIndex": 0, "text": content}},
            ]

            slides_service.presentations().batchUpdate(
                presentationId=presentation_id, body={"requests": requests}
            ).execute()
            file_url = f"https://docs.google.com/presentation/d/{presentation_id}/edit"

        elif app_type == "docs":
            from googleapiclient.discovery import build

            docs_service = build("docs", "v1", credentials=drive_service._http.credentials)

            file_metadata = {
                "name": self.file_name,
                "mimeType": "application/vnd.google-apps.document",
                "parents": [self.folder_id],
            }

            created_file = drive_service.files().create(body=file_metadata, fields="id").execute()
            document_id = created_file["id"]

            time.sleep(2)  # Wait for file to be available  # noqa: ASYNC251

            # Add content to document
            requests = [{"insertText": {"location": {"index": 1}, "text": content}}]
            docs_service.documents().batchUpdate(documentId=document_id, body={"requests": requests}).execute()
            file_url = f"https://docs.google.com/document/d/{document_id}/edit"

        return Message(text=f"File successfully created in Google {app_type.title()}: {file_url}")

    def _extract_content_for_upload(self, input_data) -> str:
        """Extract content from input for upload to cloud services."""
        if self._get_input_type(input_data) == "DataFrame":
            return input_data.to_csv(index=False)
        if self._get_input_type(input_data) == "Data":
            if hasattr(input_data, "data") and input_data.data:
                if isinstance(input_data.data, dict):
                    import json

                    return json.dumps(input_data.data, indent=2, ensure_ascii=False)
                return str(input_data.data)
            return str(input_data)
        if self._get_input_type(input_data) == "Message":
            return str(input_data.text) if input_data.text else str(input_data)
        return str(input_data)
