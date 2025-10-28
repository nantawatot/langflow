from langflow.custom.custom_component.component import Component
from langflow.inputs import MultilineInput
from langflow.io import MessageTextInput, Output
from langflow.omniscien_backend.google.core import GoogleAPIServiceClient, GoogleDocsClient
from langflow.schema.data import Data


class EditGoogleDocs(Component):
    display_name = "Edit Google Docs"
    description = "Component to Edit Google Docs."
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "Omniscien"
    name = "EditGoogleDocs"

    inputs = [
        MessageTextInput(
            name="integration_name",
            display_name="Integration Name",
            info="Name of the integration to use.",
            value="GD",
        ),
        MessageTextInput(
            name="document_id",
            display_name="Document ID",
            info="ID of the Google Docs to edit.",
            value="1cH0zf0EznIdMSwMcb_0c3NRuyF0OT1fkVdI-COOlBUE",
        ),
        MultilineInput(
            name="markdown_content",
            display_name="Markdown Content",
            info="Markdown content to be added to the Google Docs.",
            value="""
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
""",
        ),
    ]

    outputs = [
        Output(display_name="Output", name="output", method="build_output"),
    ]

    def build_output(self) -> Data:
        docs_client = GoogleDocsClient(
            document_id=self.document_id,
            service_client=GoogleAPIServiceClient(
                integration_name=self.integration_name,
                client_id="19512985706-3vgcgv3cl7ak78q93vanih7dqjn2a01a.apps.googleusercontent.com",
                client_secret="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",
                scopes=["https://www.googleapis.com/auth/documents"],
            ),
        )

        document = docs_client.get_document()

        # Get the last index of the document
        last_index = document.body.content[-1].endIndex
        print(last_index)

        request_list = docs_client.parse_markdown_to_requests(self.markdown_content, start_index=last_index - 1)
        docs_client.update_document(request_list)

        return Data(data={"Done": self.document_id})
