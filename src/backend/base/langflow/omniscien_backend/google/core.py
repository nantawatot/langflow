import json
import re
from enum import Enum
from pathlib import Path
from typing import Any

import requests
from anytree import Node, RenderTree
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from langflow.omniscien_backend.google.model.docs import (
    BulletPreset,
    CreateParagraphBulletsRequest,
    DeleteContentRangeRequest,
    DeleteParagraphBulletsRequest,
    DeleteTableColumnRequest,
    DeleteTableRowRequest,
    Document,
    EndOfSegmentLocation,
    GlyphType,
    InsertTableColumnRequest,
    InsertTableRequest,
    InsertTableRowRequest,
    InsertTextRequest,
    Location,
    Range,
    ReplaceAllTextRequest,
    Request,
    SubstringMatchCriteria,
    Table,
    TableCellLocation,
)
from langflow.omniscien_backend.google.model.drive import File, FileList
from langflow.omniscien_backend.google.model.mime_type import DocsExportMimeType

ƒƒ


class ServiceType(Enum):
    DRIVE = "drive"
    DOCS = "docs"


class GoogleAPIServiceClient:
    """Internal service client - not meant for direct user interaction."""

    def __init__(self, integration_name: str, client_id: str, client_secret: str, scopes: list[str]):
        self.credentials = Credentials(
            token=None,
            refresh_token=self._get_refresh_token(integration_name),
            scopes=scopes,
            client_id=client_id,
            client_secret=client_secret,
            token_uri="https://accounts.google.com/o/oauth2/token",
        )

    @staticmethod
    def _get_refresh_token(integration_name: str) -> str:
        result = requests.get(
            f"http://devdemo.languagestudio.com:4000/lsrestapi/v6/getintegrationinfo?accountid=1001&integrationtypecode=8&integrationname={integration_name}"
        )
        if result.status_code != 200:
            raise Exception("Failed to get token")
        if result.json()["result"]["parameters"]["refreshtoken"] is None:
            raise Exception("Failed to get token")
        return result.json()["result"]["parameters"]["refreshtoken"]

    def get_service(self, service_name: ServiceType, version: str):
        """Get a Google API service instance."""
        return build(serviceName=service_name.value, version=version, credentials=self.credentials)


class GoogleDriveClient:
    """Public API for Google Drive operations."""

    def __init__(self, service_client: GoogleAPIServiceClient):
        self._service = service_client.get_service(ServiceType.DRIVE, version="v3")

    # ========== PUBLIC METHODS ==========

    def get_folder_tree(self, folder_id: str) -> Node | None:
        """Get the complete file tree for a folder.

        Args:
            folder_id: The unique ID of the folder

        Returns:
            A tree node representing the folder structure, or None if not found
        """
        print(f"Fetching all descendants for folder ID: {folder_id}...")
        files = self._fetch_all_descendants(folder_id)

        if not files:
            print(f"Warning: Folder with ID '{folder_id}' not found or is empty.")
            return None

        print("Building subtree...")
        return self._build_file_tree(files, target_root_id=folder_id)

    def find_files(self, name: str) -> list[File]:
        """Find files or folders by exact name.

        Args:
            name: The exact name to search for

        Returns:
            List of matching files/folders
        """
        query = f"name = '{name}' and trashed = false"

        result = self._service.files().list(q=query, fields="files(id, name, parents, mimeType)").execute()

        return FileList(**result).files or []

    def list_folder_contents(self, folder_id: str = "root") -> list[File]:
        """Get all files and folders directly within a folder.

        Args:
            folder_id: The folder ID (defaults to 'root')

        Returns:
            List of files in the folder
        """
        fields = "*"
        query = f"'{folder_id}' in parents and trashed = false"

        result = self._service.files().list(q=query, fields=fields).execute()
        return FileList(**result).files or []

    def node_to_dict(self, node: Node) -> dict:
        """Recursively convert a Node tree into a dictionary for JSON serialization."""
        data = {
            "id": node.file.id if hasattr(node.file, "id") else None,
            "name": node.file.name if hasattr(node.file, "name") else node.name,
            "mimeType": getattr(node.file, "mimeType", None),
            "children": [self.node_to_dict(child) for child in node.children],
        }
        return data

    def export_docs(self, file_id: str, export_format: DocsExportMimeType, save_to: Path):
        request = self._service.files().export_media(fileId=file_id, mimeType=export_format.value)

        with open(save_to, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"Download {int(status.progress() * 100)}%.")

    # ========== PRIVATE METHODS ==========

    @staticmethod
    def _build_file_tree(files: list[File], target_root_id: str | None = None) -> Node | None:
        """Build tree structure from flat file list."""
        if not files:
            return None

        file_tree: dict[str, Node] = {}
        virtual_root = Node("virtual_root")
        all_file_ids = {file.id for file in files}

        for file in files:
            file_tree[file.id] = Node(file.name, file=file)

        for file in files:
            file_node = file_tree[file.id]
            if file.parents and file.parents[0] in all_file_ids:
                parent_node = file_tree.get(file.parents[0])
                if parent_node:
                    file_node.parent = parent_node
            else:
                file_node.parent = virtual_root

        if target_root_id and target_root_id in file_tree:
            return file_tree[target_root_id]

        if len(virtual_root.children) == 1:
            return virtual_root.children[0]

        return virtual_root

    def _fetch_all_descendants(self, folder_id: str) -> list[File]:
        """Recursively fetch all items under a folder."""
        all_files: list[File] = []
        fields = "*"

        root_folder_meta = self._service.files().get(fileId=folder_id, fields=fields).execute()
        all_files.append(File(**root_folder_meta))

        folders_to_process = [folder_id]
        while folders_to_process:
            current_folder_id = folders_to_process.pop(0)
            query = f"'{current_folder_id}' in parents and trashed = false"
            page_token = None
            while True:
                result = (
                    self._service.files()
                    .list(q=query, fields=f"nextPageToken, files({fields})", pageToken=page_token)
                    .execute()
                )

                parsed_result = FileList(**result)
                items = parsed_result.files or []

                for item in items:
                    all_files.append(item)
                    if item.mimeType == "application/vnd.google-apps.folder":
                        folders_to_process.append(item.id)

                page_token = parsed_result.nextPageToken
                if not page_token:
                    break
        return all_files


class GoogleDocsClient:
    """Public API for Google Docs operations."""

    def __init__(self, document_id: str, service_client: GoogleAPIServiceClient):
        self.document_id = document_id
        self._service = service_client.get_service(ServiceType.DOCS, version="v1")

    # ========== DOCUMENT OPERATIONS ==========

    def get_document(self) -> Document:
        """Fetch the complete document.

        Returns:
            Document object with all content and metadata
        """
        document_dict = self._service.documents().get(documentId=self.document_id).execute()
        Path("../document.json").write_text(json.dumps(document_dict, indent=4))
        return Document(**document_dict)

    def update_document(self, requests_list: list) -> dict:
        """Apply batch updates to the document.

        Args:
            requests_list: List of request dictionaries

        Returns:
            The API response
        """
        return (
            self._service.documents()
            .batchUpdate(documentId=self.document_id, body={"requests": requests_list})
            .execute()
        )

    # ========== TEXT EXTRACTION ==========

    def get_text(self, document: Document) -> str:
        """Extract plain text from a document.

        Args:
            document: The document to extract text from

        Returns:
            The document text as a string
        """
        return self._extract_text(document)

    def save_as_markdown(self, document: Document, output_path: Path = Path("../document.md")):
        """Export document to markdown file.

        Args:
            document: The document to export
            output_path: Path for the output file
        """
        text = self._extract_text(document)
        output_path.write_text(text)

    # ========== MARKDOWN OPERATIONS ==========

    @staticmethod
    def parse_markdown_to_requests(markdown_text: str, start_index: int = 1) -> list[dict]:
        requests_list = []
        current_index = start_index
        lines = markdown_text.strip().splitlines()

        # Group lines into blocks: text, bullets, numbered
        blocks = []
        current_block = []
        block_type = None

        def flush_block():
            nonlocal current_block, block_type
            if current_block:
                blocks.append((block_type, current_block))
                current_block = []

        for line in lines:
            # Check for markdown table
            if GoogleDocsClient.is_markdown_table_line(line):
                if block_type != "table":
                    flush_block()
                    block_type = "table"
                current_block.append(line)
            elif re.match(r"^(\s*)([-*])\s+(.+)$", line):
                if block_type != "bullet":
                    flush_block()
                    block_type = "bullet"
                current_block.append(line)
            elif re.match(r"^(\s*)(\d+)\.\s+(.+)$", line):
                if block_type != "numbered":
                    flush_block()
                    block_type = "numbered"
                current_block.append(line)
            else:
                if block_type != "text":
                    flush_block()
                    block_type = "text"
                current_block.append(line)

        flush_block()

        # Process blocks
        for btype, blines in blocks:
            if btype == "table":
                table_data = GoogleDocsClient.parse_markdown_table(blines)
                if table_data:
                    table_start = current_index
                    num_rows = table_data["num_rows"]
                    num_cols = table_data["num_cols"]

                    # Create empty table request
                    requests_list.append(GoogleDocsClient.build_insert_table_request(num_rows, num_cols, current_index))

                    # Generate requests to populate table cells
                    cell_population_requests = GoogleDocsClient.populate_table_cells(table_start, table_data)
                    requests_list.extend(cell_population_requests)

                    # CORRECTED INDEX CALCULATION:
                    # The total size increase is the sum of the structural elements,
                    # the content inserted, and the newline Google Docs auto-inserts before the table.

                    # 1. Calculate the structural size of the table itself.
                    #    Structure: 1 (table start) + 1 (table end) + num_rows * (1 (row start) + num_cols * 2 (cell overhead))
                    structural_size = 2 + num_rows * (1 + num_cols * 2)

                    # 2. Calculate the total length of all text inserted into the cells.
                    total_text_length = 0
                    for req in cell_population_requests:
                        # Ensure we're looking at an insertText request with actual text
                        if "insertText" in req and "text" in req["insertText"]:
                            text_to_insert = req["insertText"]["text"]
                            # Text length is measured in UTF-16 code units
                            total_text_length += len(text_to_insert.encode("utf-16-le")) // 2

                    # 3. The auto-inserted newline before the table takes 1 index position.
                    auto_inserted_newline_size = 1

                    # 4. Update current_index with the total size.
                    current_index += auto_inserted_newline_size + structural_size + total_text_length
            elif btype == "bullet":
                start = current_index
                for line in blines:
                    match = re.match(r"^(\s*)([-*])\s+(.+)$", line)
                    indent = match.group(1)
                    text = match.group(3)
                    level = len(indent) // 4
                    bullet_text = "\t" * level + text + "\n"
                    requests_list.append(GoogleDocsClient.build_insert_text_request(bullet_text, current_index))
                    current_index += len(bullet_text.encode("utf-16-le")) // 2
                current_index -= 1
                requests_list.append(
                    GoogleDocsClient.build_add_bullets_request(
                        start, current_index, bullet_preset=BulletPreset.BULLET_DISC_CIRCLE_SQUARE
                    )
                )
            elif btype == "numbered":
                start = current_index
                for line in blines:
                    match = re.match(r"^(\s*)(\d+)\.\s+(.+)$", line)
                    indent = match.group(1)
                    text = match.group(3)
                    level = len(indent) // 4
                    bullet_text = "\t" * level + text + "\n"
                    requests_list.append(GoogleDocsClient.build_insert_text_request(bullet_text, current_index))
                    current_index += len(bullet_text.encode("utf-16-le")) // 2
                current_index -= 1
                requests_list.append(
                    GoogleDocsClient.build_add_bullets_request(
                        start, current_index, bullet_preset=BulletPreset.NUMBERED_DECIMAL_ALPHA_ROMAN
                    )
                )
            else:  # normal text
                for line in blines:
                    if not line.strip():
                        text = "\n"
                    else:
                        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
                        text = header_match.group(2) + "\n" if header_match else line + "\n"
                    requests_list.append(GoogleDocsClient.build_insert_text_request(text, current_index))
                    current_index += len(text.encode("utf-16-le")) // 2

        return requests_list

    # ========== REQUEST BUILDERS ==========

    @staticmethod
    def build_insert_text_request(text: str, index: int | None = None, insert_at_end: bool = False) -> dict:
        """Create a request to insert text at a specific position."""
        if insert_at_end:
            request = Request(
                insertText=InsertTextRequest(endOfSegmentLocation=EndOfSegmentLocation(segmentId=""), text=text)
            )
        else:
            if index is None:
                raise ValueError("index must be provided when insert_at_end=False")
            request = Request(insertText=InsertTextRequest(location=Location(index=index), text=text))
        return request.model_dump(exclude_none=True)

    @staticmethod
    def build_delete_text_request(start_index: int, end_index: int) -> dict:
        """Create a request to delete text in a range."""
        request = Request(
            deleteContentRange=DeleteContentRangeRequest(range=Range(startIndex=start_index, endIndex=end_index))
        )
        return request.model_dump(exclude_none=True)

    @staticmethod
    def build_replace_text_request(old_text: str, new_text: str, match_case: bool = False) -> dict:
        """Create a request to replace all occurrences of text."""
        request = Request(
            replaceAllText=ReplaceAllTextRequest(
                replaceText=new_text,
                containsText=SubstringMatchCriteria(text=old_text, matchCase=match_case),
            )
        )
        return request.model_dump(exclude_none=True)

    @staticmethod
    def build_add_bullets_request(start_index: int, end_index: int, bullet_preset: BulletPreset) -> dict:
        """Create a request to add bullets to paragraphs."""
        request = Request(
            createParagraphBullets=CreateParagraphBulletsRequest(
                range=Range(startIndex=start_index, endIndex=end_index),
                bulletPreset=bullet_preset,
            )
        )
        return request.model_dump(exclude_none=True)

    @staticmethod
    def build_remove_bullets_request(start_index: int, end_index: int) -> dict:
        """Create a request to remove bullets from paragraphs."""
        request = Request(
            deleteParagraphBullets=DeleteParagraphBulletsRequest(
                range=Range(startIndex=start_index, endIndex=end_index),
            )
        )
        return request.model_dump(exclude_none=True)

    @staticmethod
    def build_insert_table_request(
        rows: int, columns: int, index: int | None = None, insert_at_end: bool = False
    ) -> dict:
        """Create a request to insert a table.

        Args:
            rows: Number of rows
            columns: Number of columns
            index: Position to insert (required if insert_at_end=False)
            insert_at_end: If True, insert at end of document body

        Returns:
            Request dictionary
        """
        insert_table_params = {"rows": rows, "columns": columns}

        if insert_at_end:
            insert_table_params["endOfSegmentLocation"] = EndOfSegmentLocation(segmentId="").model_dump(
                exclude_none=True
            )
        else:
            if index is None:
                raise ValueError("index must be provided when insert_at_end=False")
            insert_table_params["location"] = Location(index=index).model_dump(exclude_none=True)

        request = Request(insertTable=InsertTableRequest(**insert_table_params))
        return request.model_dump(exclude_none=True)

    @staticmethod
    def build_insert_table_row_request(
        table_start_index: int, row_index: int, column_index: int, insert_below: bool = True
    ) -> dict:
        """Create a request to insert a table row."""
        request = Request(
            insertTableRow=InsertTableRowRequest(
                tableCellLocation=TableCellLocation(
                    tableStartLocation=Location(index=table_start_index), rowIndex=row_index, columnIndex=column_index
                ),
                insertBelow=insert_below,
            )
        )
        return request.model_dump(exclude_none=True)

    @staticmethod
    def build_insert_table_column_request(
        table_start_index: int, row_index: int, column_index: int, insert_right: bool = True
    ) -> dict:
        """Create a request to insert a table column."""
        request = Request(
            insertTableColumn=InsertTableColumnRequest(
                tableCellLocation=TableCellLocation(
                    tableStartLocation=Location(index=table_start_index), rowIndex=row_index, columnIndex=column_index
                ),
                insertRight=insert_right,
            )
        )
        return request.model_dump(exclude_none=True)

    @staticmethod
    def build_delete_table_row_request(table_start_index: int, row_index: int, column_index: int) -> dict:
        """Create a request to delete a table row."""
        request = Request(
            deleteTableRow=DeleteTableRowRequest(
                tableCellLocation=TableCellLocation(
                    tableStartLocation=Location(index=table_start_index), rowIndex=row_index, columnIndex=column_index
                )
            )
        )
        return request.model_dump(exclude_none=True)

    @staticmethod
    def build_delete_table_column_request(table_start_index: int, row_index: int, column_index: int) -> dict:
        """Create a request to delete a table column."""
        request = Request(
            deleteTableColumn=DeleteTableColumnRequest(
                tableCellLocation=TableCellLocation(
                    tableStartLocation=Location(index=table_start_index), rowIndex=row_index, columnIndex=column_index
                )
            )
        )
        return request.model_dump(exclude_none=True)

    # ========== PRIVATE HELPERS: TEXT EXTRACTION ==========

    def _extract_text(self, document: Document) -> str:
        """Extract text content from document."""
        list_styles = self._extract_list_styles(document)
        list_counters = {}
        text_parts = []

        for content in document.body.content:
            if content.paragraph:
                self._process_paragraph(content.paragraph, list_styles, list_counters, text_parts)
            elif content.table:
                text_parts.append(self._build_html_table(content.table))

        return "".join(text_parts)

    def _process_paragraph(self, paragraph, list_styles: dict, list_counters: dict, text_parts: list[str]):
        """Process a single paragraph and append to text_parts."""
        for element in paragraph.elements:
            if paragraph.bullet:
                formatted_text = self._format_bullet_item(paragraph.bullet, element, list_styles, list_counters)
                text_parts.append(formatted_text)
            elif element.textRun:
                text_parts.append(element.textRun.content)

    def _format_bullet_item(self, bullet, element, list_styles: dict, list_counters: dict) -> str:
        """Format a bullet list item with appropriate styling."""
        list_id = bullet.listId
        nesting = bullet.nestingLevel or 0

        if list_id not in list_counters:
            list_counters[list_id] = {}

        # Reset counters for deeper nesting levels
        list_counters[list_id] = {lvl: cnt for lvl, cnt in list_counters[list_id].items() if lvl <= nesting}

        counter = list_counters[list_id].get(nesting, list_styles[list_id][nesting]["startNumber"] or 1)
        list_counters[list_id][nesting] = counter + 1

        style = list_styles[list_id][nesting]
        return self._format_list_item(nesting, counter, style, element.textRun.content)

    @staticmethod
    def _extract_list_styles(document: Document) -> dict:
        """Extract list style information from document."""
        return {
            list_id: [
                {"glyphType": lvl.glyphType, "glyphSymbol": lvl.glyphSymbol, "startNumber": lvl.startNumber}
                for lvl in list_obj.listProperties.nestingLevels
            ]
            for list_id, list_obj in document.lists.items()
        }

    @classmethod
    def _format_list_item(cls, nesting_level: int, counter: int, style: dict, content: str) -> str:
        """Format a list item based on its style."""
        prefix = nesting_level * "\t"
        glyph_type = style["glyphType"]
        glyph_symbol = style["glyphSymbol"]

        if glyph_symbol:
            return f"{prefix} {glyph_symbol} {content}"

        match glyph_type:
            case GlyphType.DECIMAL:
                return f"{prefix} {counter}. {content}"
            case GlyphType.UPPER_ALPHA:
                return f"{prefix} {cls._int_to_upper_alpha(counter)}. {content}"
            case GlyphType.ALPHA:
                return f"{prefix} {cls._int_to_alpha(counter)}. {content}"
            case GlyphType.UPPER_ROMAN:
                return f"{prefix} {cls._int_to_roman(counter)}. {content}"
            case GlyphType.ROMAN:
                return f"{prefix} {cls._int_to_roman(counter).lower()}. {content}"
            case _:
                return content

    @staticmethod
    def _build_html_table(table_data: Table) -> str:
        """Build HTML representation of a table."""
        rows = table_data.tableRows
        if not rows:
            return ""

        num_cols = len(rows[0].tableCells)
        spanned = [[False] * num_cols for _ in range(len(rows))]
        html = ["<table>"]

        for r_idx, row in enumerate(rows):
            html.append("  <tr>")
            c_idx = 0

            for cell in row.tableCells:
                # Skip already-spanned cells
                while c_idx < num_cols and spanned[r_idx][c_idx]:
                    c_idx += 1
                if c_idx >= num_cols:
                    continue

                colspan = getattr(cell.tableCellStyle, "columnSpan", 1) or 1
                rowspan = getattr(cell.tableCellStyle, "rowSpan", 1) or 1

                # Build cell attributes
                td_attrs = []
                if colspan > 1:
                    td_attrs.append(f'colspan="{colspan}"')
                if rowspan > 1:
                    td_attrs.append(f'rowspan="{rowspan}"')

                # Extract cell text
                cell_text = "".join(
                    el.textRun.content.strip()
                    for content in cell.content
                    if content.paragraph
                    for el in content.paragraph.elements
                    if el.textRun
                )

                html.append(f"    <td {' '.join(td_attrs)}>{cell_text}</td>")

                # Mark spanned cells
                for r_off in range(rowspan):
                    for c_off in range(colspan):
                        if r_off or c_off:
                            if r_idx + r_off < len(rows) and c_idx + c_off < num_cols:
                                spanned[r_idx + r_off][c_idx + c_off] = True

                c_idx += colspan

            html.append("  </tr>")

        html.append("</table>")
        return "\n".join(html)

    # ========== PRIVATE HELPERS: NUMBER FORMATTING ==========

    @staticmethod
    def _int_to_roman(num: int) -> str:
        """Convert integer to Roman numeral."""
        m = ["", "M", "MM", "MMM"]
        c = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]
        x = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
        i = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
        return m[num // 1000] + c[(num % 1000) // 100] + x[(num % 100) // 10] + i[num % 10]

    @staticmethod
    def _int_to_alpha(num: int) -> str:
        """Convert integer to lowercase alphabetic (a, b, c, ... z, aa, ab, ...)."""
        result = ""
        while num > 0:
            num, remainder = divmod(num - 1, 26)
            result = chr(97 + remainder) + result
        return result

    @classmethod
    def _int_to_upper_alpha(cls, num: int) -> str:
        """Convert integer to uppercase alphabetic."""
        return cls._int_to_alpha(num).upper()

    @staticmethod
    def is_markdown_table_line(line: str) -> bool:
        """Check if a line is part of a markdown table."""
        stripped = line.strip()
        if "|" not in stripped:
            return False

        # Check if it's a separator line (e.g., |---|---|)
        if re.match(r"^\s*\|?[\s\-:|]+\|\s*$", stripped):
            return True

        # Check if it's a content line with pipes
        if stripped.startswith("|") or stripped.endswith("|"):
            return True

        # Check if it has at least one pipe (inline table format)
        if "|" in stripped:
            return True

        return False

    @staticmethod
    def parse_markdown_table(table_lines: list[str]) -> dict[str, list[Any] | list[str] | int] | None:
        """Parse markdown table lines into structured data.

        Returns:
            dict with 'headers', 'rows', 'num_cols', 'num_rows'
        """
        if not table_lines:
            return None

        # Remove empty lines
        table_lines = [line.strip() for line in table_lines if line.strip()]

        # Find the separator line (usually second line with dashes)
        separator_idx = None
        for i, line in enumerate(table_lines):
            if re.match(r"^\s*\|?[\s\-:|]+\|\s*$", line):
                separator_idx = i
                break

        if separator_idx is None:
            return None

        # Parse headers (line before separator)
        headers = []
        if separator_idx > 0:
            header_line = table_lines[separator_idx - 1]
            headers = [cell.strip() for cell in header_line.split("|") if cell.strip()]

        # Parse data rows (lines after separator)
        rows = []
        for line in table_lines[separator_idx + 1 :]:
            cells = [cell.strip() for cell in line.split("|") if cell.strip()]
            if cells:
                rows.append(cells)

        num_cols = max(len(headers), max((len(row) for row in rows), default=0)) if headers or rows else 0

        return {"headers": headers, "rows": rows, "num_cols": num_cols, "num_rows": len(rows) + (1 if headers else 0)}

    @staticmethod
    def populate_table_cells(table_start_index: int, table_data: dict) -> list[dict]:
        """Generate requests to populate table cells with data, correctly accounting for index shifts from previous insertions within the same batch request.

        Args:
            table_start_index: Starting index of the table.
            table_data: Dict with 'headers', 'rows', 'num_cols', 'num_rows'.

        Returns:
            List of request dictionaries to insert content into table cells.
        """
        requests = []
        num_cols = table_data.get("num_cols", 0)
        if num_cols == 0:
            return []

        # This variable tracks the total length of all text inserted so far,
        # which is used to offset the indices for subsequent insertions.
        cumulative_text_length = 0

        # Combine headers and rows into a single list for easier iteration.
        all_rows_data = []
        if table_data.get("headers"):
            all_rows_data.append(table_data["headers"])
        all_rows_data.extend(table_data.get("rows", []))

        for row_idx, row_cells in enumerate(all_rows_data):
            for col_idx, cell_text in enumerate(row_cells):
                # Process only if the cell has content and is within the table's column bounds.
                if col_idx < num_cols and cell_text:
                    # Calculate the index of the current cell's paragraph in the initial empty table.
                    base_cell_index = GoogleDocsClient.calculate_table_cell_index(
                        table_start_index, row_idx, col_idx, num_cols
                    )

                    # The actual insertion index is the base index plus the length of all
                    # text we've inserted into previous cells in this batch operation.
                    insertion_index = base_cell_index + cumulative_text_length

                    # Create the request to insert the current cell's text.
                    requests.append(GoogleDocsClient.build_insert_text_request(cell_text, insertion_index))

                    # Update the cumulative length with the length of the text just added.
                    # Google Docs API indices are based on UTF-16 code units, where each
                    # unit is 2 bytes.
                    cumulative_text_length += len(cell_text.encode("utf-16-le")) // 2

        return requests

    @staticmethod
    def calculate_table_cell_index(table_start_index: int, row_idx: int, col_idx: int, num_cols: int) -> int:
        """Calculate the paragraph start index for a table cell without fetching document.

        IMPORTANT: Google Docs automatically inserts a newline BEFORE the table!

        If you insert table at index 214, the structure becomes:
          213: (original content)
          214: \\n (auto-inserted by Google Docs)
          215: Table start
          216: Row 0 start
          217: Cell 0,0 para with \\n
          219: Cell 0,1 para with \\n
          221: Row 1 start
          ...

        So actual table_start is at (requested_index + 1)

        Args:
            table_start_index: The index you REQUESTED for table insertion
            row_idx: Zero-based row index
            col_idx: Zero-based column index
            num_cols: Total number of columns in the table

        Returns:
            Index where paragraph content starts (where to insert/delete text)
        """
        # Account for the auto-inserted newline before table
        actual_table_start = table_start_index + 1

        # Now calculate from actual table position
        index = actual_table_start + 1  # Skip table start

        # Skip complete rows before target
        indices_per_row = 1 + (num_cols * 2)  # 1 row_start + cells
        index += row_idx * indices_per_row

        # Add row start
        index += 1

        # Skip cells before target (2 indices per cell)
        index += col_idx * 2

        # Now at cell start, paragraph content is next index
        index += 1

        return index


# ========== EXAMPLE USAGE ==========
if __name__ == "__main__":
    try:
        # Initialize the drive client
        drive_client = GoogleDriveClient(
            service_client=GoogleAPIServiceClient(
                integration_name="GD",
                client_id="19512985706-3vgcgv3cl7ak78q93vanih7dqjn2a01a.apps.googleusercontent.com",
                client_secret="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",
                scopes=["https://www.googleapis.com/auth/drive"],
            )
        )

        docs_client = GoogleDocsClient(
            document_id="1cH0zf0EznIdMSwMcb_0c3NRuyF0OT1fkVdI-COOlBUE",
            service_client=GoogleAPIServiceClient(
                integration_name="GD",
                client_id="19512985706-3vgcgv3cl7ak78q93vanih7dqjn2a01a.apps.googleusercontent.com",
                client_secret="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",
                scopes=["https://www.googleapis.com/auth/documents"],
            ),
        )

        # Download document

        markdown_content = """
# Weekly Report

## Day 1
- Task 1: Finish report
    * Subtask 1: Finish report section 1
- Task 2: Finish report section 2
Test
## Day 2
1. Task 1: Finish report
    1. Subtask 1: Finish report section 1
2. Task 2: Finish report section 2

| Syntax      | Description |
| ----------- | ----------- |
| Header      | Title       |
| Paragraph   | Text        |

## Day 1
- Task 1: Finish report
    * Subtask 1: Finish report section 1
- Task 2: Finish report section 2

## Day 2
1. Task 1: Finish report
    1. Subtask 1: Finish report section 1
2. Task 2: Finish report section 2
\n
"""

        # Get folder tree - PUBLIC API
        target_folder_id = "root"
        print(f"\nFetching folder tree for: '{target_folder_id}'")
        subtree_node = drive_client.get_folder_tree(target_folder_id)

        if subtree_node:
            print("\n--- Folder Structure ---")
            for pre, _, node in RenderTree(subtree_node):
                print(f"{pre}{node.file.name} [{node.file.mimeType}]")
                if node.file.mimeType == "application/vnd.google-apps.document":
                    docs_client = GoogleDocsClient(
                        document_id=node.file.id,
                        service_client=GoogleAPIServiceClient(
                            integration_name="GD",
                            client_id="19512985706-3vgcgv3cl7ak78q93vanih7dqjn2a01a.apps.googleusercontent.com",
                            client_secret="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",
                            scopes=["https://www.googleapis.com/auth/documents"],
                        ),
                    )
                    request_list = docs_client.parse_markdown_to_requests(markdown_content, start_index=1)
                    docs_client.update_document(request_list)

        else:
            print("\nFolder not found or empty.")

    except HttpError as err:
        print(f"API error: {err}")
