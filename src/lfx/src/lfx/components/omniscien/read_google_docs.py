from langflow.custom.custom_component.component import Component
from langflow.io import MessageTextInput, Output
from langflow.omniscien_backend.google.core import GoogleAPIServiceClient, GoogleDocsClient
from langflow.schema import Message


class ReadGoogleDocs(Component):
    display_name = "Read Google Docs"
    description = "Component to Edit Google Docs."
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "Omniscien"
    name = "ReadGoogleDocs"

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
    ]

    outputs = [
        Output(display_name="Output", name="output", method="build_output"),
    ]

    def build_output(self) -> Message:
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
        text = docs_client.get_text(document)

        return Message(text=text)
