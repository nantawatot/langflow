from langflow.custom.custom_component.component import Component
from langflow.io import MessageTextInput, Output
from langflow.omniscien_backend.google.core import GoogleAPIServiceClient, GoogleDriveClient
from langflow.schema.data import Data


class ListGoogleDrive(Component):
    display_name = "List Google Drive"
    description = "Component to list the Google Drive files."
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "Omniscien"
    name = "ListGoogleDrive"

    inputs = [
        MessageTextInput(
            name="integration_name",
            display_name="Integration Name",
            info="Name of the integration to use.",
            value="GD",
        ),
        MessageTextInput(
            name="folder_id",
            display_name="Folder ID",
            info="ID of the folder to list. Defaults to root folder.",
            value="root",
        ),
    ]

    outputs = [
        Output(display_name="Output", name="output", method="build_output"),
    ]

    def build_output(self) -> Data:
        drive_client = GoogleDriveClient(
            service_client=GoogleAPIServiceClient(
                integration_name=self.integration_name,
                client_id="19512985706-3vgcgv3cl7ak78q93vanih7dqjn2a01a.apps.googleusercontent.com",
                client_secret="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",
                scopes=["https://www.googleapis.com/auth/drive"],
            )
        )

        subtree_node = drive_client.get_folder_tree(self.folder_id)
        tree_json = {}
        if subtree_node:
            tree_json = drive_client.node_to_dict(subtree_node)

        data = Data(data=tree_json)
        self.status = data
        return data
