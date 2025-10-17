import json
from typing import Any

from langflow.custom import Component
from langflow.io import DataInput, Output
from langflow.schema import Data


class ZoneAssignment(Component):
    display_name = "Zone Assignment"
    description = "Convert JSON into valid Zone format for Mail Merge API to use"
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "Omniscien"
    name = "ZoneAssignment"

    inputs = [
        DataInput(
            name="json_input",
            display_name="JSON Input",
            required=True,
            info="Input json for the component that will be used to create the zones.",
        )
    ]

    outputs = [
        Output(display_name="Zone Assigned Output", name="zone_assigned_output", method="make_zones"),
    ]

    def make_zones(self) -> Data:
        input_json = self.json_input

        if isinstance(input_json, str):
            input_data = json.loads(input_json)
        else:
            input_data = input_json.data

        all_sections = []

        placeholder_styles_list = self.graph.context.get("placeholder_styles")

        for section_name, section_data in input_data.items():
            zone_section = {"zonename": section_name, "zonekeys": {}, "zonearray": []}

            required = True
            if isinstance(section_data, dict):
                if section_data.get("required_section") == "False":
                    required = False

            zone_section["zonedelete"] = not required

            if isinstance(section_data, list):
                # Pass the placeholder styles list to the helper function
                zone_section["zonearray"] = self.process_zone_array(section_data, placeholder_styles_list, section_name)
                zone_section["zonekeys"]["item_count"] = len(zone_section["zonearray"])

            elif isinstance(section_data, dict):
                zone_section["zonekeys"] = self.process_zone_keys(section_data, placeholder_styles_list)

            else:
                zone_section["zonekeys"][section_name] = section_data

            all_sections.append(zone_section)

        zone_structure = {"zones": all_sections}
        return zone_structure

    def process_zone_keys(self, data: dict[str, Any], placeholder_styles_list: list[dict[str, Any]]) -> dict[str, Any]:
        processed_keys = {}
        list_type = data.get("list_type")

        for key, value in data.items():
            if key in ["list_type", "required_section"]:
                continue

            if isinstance(value, list) and list_type:
                # Find the style for the list key and pass it to the formatter
                item_style = next((item for item in placeholder_styles_list if item.get("placeholder") == key), {})
                font_name = item_style.get("font_name")
                font_size = item_style.get("font_size")
                processed_keys[key] = self.format_list_as_markdown(value, list_type, font_name, font_size)
            else:
                processed_keys[key] = value

        return processed_keys

    def format_list_as_markdown(
        self, items: list[Any], list_type: str, font_name: str = None, font_size: float = None
    ) -> str:
        font_style = f"font-family: {font_name};" if font_name else ""
        font_size_style = f"font-size: {font_size}pt;" if font_size else ""
        div_style = f'style="{font_style} {font_size_style}"' if font_name or font_size else ""
        self.log(f"[Function format_list_as_markdown]: font-style: {font_style}; font-size: {font_size}px;")

        if list_type == "multi_column_bullet":
            self.log("Formatting multi_column_bullet list")
            formatted_items = [f"- {item}" for item in items]
            return "MD: " + "\n".join(formatted_items)
        if list_type == "number":
            formatted_items = [f"{i + 1}. {item}" for i, item in enumerate(items)]
            return "MD: " + "\n".join(formatted_items)
        if list_type == "bullet":
            # Generate <li> tags for each item and wrap in a <ul> tag
            list_items = [f"<li>{item}</li>" for item in items]
            formatted_items = f"<div {div_style}><ul>{''.join(list_items)}</ul></div>"
            return "MD: " + formatted_items
        return str(items)

    def process_zone_array(
        self, data: list[Any], placeholder_styles_list: list[dict[str, Any]], section_name: str
    ) -> list[Any]:
        processed_array = []
        list_style = None
        items_to_process = []

        for item in data:
            if isinstance(item, dict) and "list_style" in item:
                list_style = item["list_style"]
            else:
                items_to_process.append(item)

        if list_style == "double_side_bullet":
            # Find the style for the table_output placeholder from the pre-processed list
            table_style = next(
                (item for item in placeholder_styles_list if item.get("placeholder") == "table_output"), {}
            )
            font_name = table_style.get("font_name")
            font_size = table_style.get("font_size")

            rows = []
            for i in range(0, len(items_to_process), 2):
                row_items = items_to_process[i : i + 2]
                row = []
                for entry in row_items:
                    if not isinstance(entry, dict):
                        continue

                    entry_keys = list(entry.keys())
                    if len(entry_keys) >= 2:
                        title_key = entry_keys[0]
                        items_key = entry_keys[1]

                        element = {
                            "title": entry.get(title_key, ""),
                            "items": self.format_list_as_markdown(
                                entry.get(items_key, []), "multi_column_bullet", font_name, font_size
                            ),
                        }
                        row.append(element)
                rows.append(row)

            # Pass the font info to the table formatter
            md_table = self.format_multi_column_table(rows, font_name, font_size)
            return [{"table_output": "MD: " + md_table}]
        # For each dictionary in the items_to_process list, find the list keys and apply styles
        for entry in items_to_process:
            if not isinstance(entry, dict):
                processed_array.append(entry)
                continue

            element = {}
            for key, value in entry.items():
                self.log(f"Processing key: {key} with value type: {type(value)}")
                if isinstance(value, list):
                    # Find the style for the specific list key (e.g., 'out_of_industry_point')
                    item_style = next((item for item in placeholder_styles_list if item.get("placeholder") == key), {})
                    font_name = item_style.get("font_name")
                    font_size = item_style.get("font_size")
                    element[key] = self.format_list_as_markdown(value, "bullet", font_name, font_size)
                else:
                    element[key] = value
            processed_array.append(element)

        return processed_array

    def format_multi_column_table(self, row_data: list[list[dict[str, Any]]], font_name: str, font_size: float) -> str:
        font_style = f"font-family: {font_name}" if font_name else ""
        font_size_style = f"font-size: {font_size}px;" if font_size else ""
        self.log(f"Applying font style: {font_style}, font size style: {font_size_style}")

        md_output = f'<div style="{font_style}; font-size: 10px;">\n'

        for row in row_data:
            md_output += '<table style="width: 100%; border-collapse: collapse;"><tr>\n'
            for idx, col in enumerate(row):
                title = col.get("title", "")
                items_md = col.get("items", "")

                if isinstance(items_md, str) and items_md.startswith("MD:"):
                    items_md = items_md[3:].strip()

                border_style = (
                    "border-right: 1px solid #000; padding-right: 8px;"
                    if idx == 0 and len(row) == 2
                    else "padding: 8px;"
                )

                md_output += f'<td style="vertical-align: top; width: 50%; {border_style}">'
                md_output += f'<div style="text-align: center; font-weight: normal;">{title}</div>\n'
                md_output += f'<ul style="{font_style}; font-size: 10px; margin: 0; padding-left: 20px;">\n'

                for line in items_md.split("\n"):
                    line_text = line.removeprefix("- ")
                    md_output += f'<li style="{font_style}; font-size: 13px;">{line_text}</li>\n'

                md_output += "</ul></td>\n"

            if len(row) == 1:
                md_output += '<td style="width:50%"></td>\n'

            md_output += "</tr></table>\n"

        md_output += "</div>"
        return md_output


######################### Old version of the table formatter for reference #########################

# def format_multi_column_table(self, row_data: List[List[Dict[str, Any]]]) -> str:
#     """
#     Generic formatter for any 2-column table:
#     - title: top-centered (was party_name)
#     - items: bullet points (was obligations)
#     """
#     md_output = '<div style="font-family: Arial; font-size: 10px;">\n'
#
#     for row in row_data:
#         md_output += '<table style="width: 100%; border-collapse: collapse;"><tr>\n'
#         for idx, col in enumerate(row):
#             title = col.get("title", "")
#             items_md = col.get("items", "")
#
#             # Remove MD prefix if present
#             if isinstance(items_md, str) and items_md.startswith("MD:"):
#                 items_md = items_md[3:].strip()
#
#             border_style = 'border-right: 1px solid #000; padding-right: 8px;' if idx == 0 and len(
#                 row) == 2 else 'padding: 8px;'
#
#             md_output += f'<td style="vertical-align: top; width: 50%; {border_style}">'
#             md_output += f'<div style="text-align: center; font-weight: normal;">{title}</div>\n'
#             md_output += '<ul style="font-family: Arial; font-size: 10px; margin: 0; padding-left: 20px;">\n'
#
#             for line in items_md.split("\n"):
#                 line_text = line[2:] if line.startswith("- ") else line
#                 md_output += f'<li style="font-family: Arial; font-size: 13px;">{line_text}</li>\n'
#
#             md_output += '</ul></td>\n'
#
#         # Add empty cell if only one column in the row
#         if len(row) == 1:
#             md_output += '<td style="width:50%"></td>\n'
#
#         md_output += '</tr></table>\n'
#
#     md_output += '</div>'
#     return md_output


######################### Old version of the process_zone_array for reference #########################
# def process_zone_array(self, data: List[Any], placeholder_styles_list: List[Dict[str, Any]]) -> List[Any]:
#     processed_array = []
#     list_style = None
#     items_to_process = []
#
#     for item in data:
#         if isinstance(item, dict) and "list_style" in item:
#             list_style = item["list_style"]
#         else:
#             items_to_process.append(item)
#
#     if list_style == "double_side_bullet":
#         # Find the style for the table_output placeholder from the pre-processed list
#         table_style = next((item for item in placeholder_styles_list if item.get("placeholder") == "table_output"),
#                            {})
#         font_name = table_style.get("font_name")
#         font_size = table_style.get("font_size")
#
#         rows = []
#         for i in range(0, len(items_to_process), 2):
#             row_items = items_to_process[i:i + 2]
#             row = []
#             for entry in row_items:
#                 if not isinstance(entry, dict):
#                     continue
#                 element = {
#                     "title": entry.get("party_name", ""),
#                     "items": self.format_list_as_markdown(entry.get("obligations", []), "bullet")
#                 }
#                 row.append(element)
#             rows.append(row)
#
#         # Pass the font info to the table formatter
#         md_table = self.format_multi_column_table(rows, font_name, font_size)
#         return [{"table_output": "MD: " + md_table}]
#
#     for entry in items_to_process:
#         if not isinstance(entry, dict):
#             processed_array.append(entry)
#             continue
#
#         element = {}
#         for key, value in entry.items():
#             if isinstance(value, list):
#                 element[key] = self.format_list_as_markdown(value, "bullet")
#             else:
#                 element[key] = value
#         processed_array.append(element)
#
#     return processed_array


######################### Old version of the list formatter for reference #########################

# def format_list_as_markdown(self, items: List[Any], list_type: str) -> str:
#     if list_type == "bullet":
#         self.log("Formatting bullet list")
#         formatted_items = [f"- {item}" for item in items]
#     elif list_type == "number":
#         formatted_items = [f"{i + 1}. {item}" for i, item in enumerate(items)]
#     else:
#         return str(items)
#
#     return "MD: " + "\n".join(formatted_items)


######################### Old version of the process_zone_keys for reference #########################
# def process_zone_keys(self, data: Dict[str, Any]) -> Dict[str, Any]:
#     processed_keys = {}
#     list_type = data.get("list_type")
#
#     for key, value in data.items():
#         if key == "list_type" or key == "required_section":
#             continue
#
#         if isinstance(value, list) and list_type:
#             processed_keys[key] = self.format_list_as_markdown(value, list_type)
#         else:
#             processed_keys[key] = value
#
#     return processed_keys
