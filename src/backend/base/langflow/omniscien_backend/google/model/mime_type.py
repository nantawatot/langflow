from enum import Enum


class DocsExportMimeType(Enum):
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    OPEN_DOCUMENT = "application/vnd.oasis.opendocument.text"
    RICH_TEXT = "application/rtf"
    PDF = "application/pdf"
    PLAIN_TEXT = "text/plain"
    HTML = "application/zip"
    EPUB = "application/epub+zip"
    MARKDOWN = "text/markdown"
