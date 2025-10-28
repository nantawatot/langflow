from langflow.custom.custom_component.component import Component
from langflow.io import MessageTextInput, Output
from langflow.omniscien_backend.google.core import GoogleAPIServiceClient, GoogleDriveClient
from langflow.schema import DataFrame
from langflow.schema.data import Data


class ListGoogleDrive(Component):
    display_name = "List Google Drive"
    description = "Component to list the Google Drive files, optionally filtered by MIME type."
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
        MessageTextInput(
            name="mime_type_filter",
            display_name="MIME Type Filter",
            info="Optional: filter files by MIME type (e.g., 'application/pdf', 'application/vnd.google-apps.folder'). Leave blank for all types.",
            value="",
        ),
    ]

    outputs = [
        Output(display_name="File Tree", name="file_tree", method="build_output_file_tree"),
        Output(display_name="Flattened File List", name="file_list", method="build_output_file_list"),
    ]

    def build_output_file_tree(self) -> Data:
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

            # Apply mimeType filter if specified
            if self.mime_type_filter:

                def filter_tree(node):
                    if node.get("mimeType") == "application/vnd.google-apps.folder":
                        node["children"] = [child for child in (node.get("children") or []) if filter_tree(child)]
                        return bool(node["children"])
                    return node.get("mimeType") == self.mime_type_filter

                # Filter the top-level tree
                if not filter_tree(tree_json):
                    tree_json = {}

        data = Data(data=tree_json)
        self.status = data
        return data

    def build_output_file_list(self) -> DataFrame:
        drive_client = GoogleDriveClient(
            service_client=GoogleAPIServiceClient(
                integration_name=self.integration_name,
                client_id="19512985706-3vgcgv3cl7ak78q93vanih7dqjn2a01a.apps.googleusercontent.com",
                client_secret="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",
                scopes=["https://www.googleapis.com/auth/drive"],
            )
        )

        flat_list = drive_client.get_flattened_file_list(self.folder_id)

        # Apply mimeType filter if specified
        if self.mime_type_filter:
            flat_list = [f for f in flat_list if f.get("mimeType") == self.mime_type_filter]

        return DataFrame(data=flat_list)
