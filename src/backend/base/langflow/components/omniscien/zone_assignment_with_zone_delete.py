import json
from typing import Any

from langflow.custom import Component
from langflow.io import DataInput, Output
from langflow.schema import Data


class ZoneAssignmentWithZoneDelete(Component):
    display_name = "Zone Assignment with Zone Delete"
    description = "Convert JSON into valid Zone format for Mail Merge API to use"
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "Omniscien"
    name = "ZoneAssignmentWithZoneDelete"

    inputs = [
        DataInput(
            name="input",
            display_name="Input",
            required=True,
            info="Input json for the component that will be used to create the zones.",
        )
    ]

    outputs = [
        Output(display_name="Output", name="output", method="make_zones"),
    ]

    def make_zones(self) -> Data:
        input_json = self.input

        if isinstance(input_json, str):
            input_data = json.loads(input_json)
        else:
            input_data = input_json.data

        all_sections = []

        for section_name, section_data in input_data.items():
            zone_section = {"zonename": section_name, "zonekeys": {}, "zonearray": []}

            # Safe check: Only get 'required_section' if section_data is a dict
            if isinstance(section_data, dict):
                required = section_data.get("required_section", True)
                self.log(f"Processing section '{section_name}' with required status: {required}")
                if section_data.get("required_section", True) == "False":
                    self.log(f"THIS IS FALSE REQUIRED SECTION {section_name}")
                    required = False
            else:
                required = True  # default to True if section_data is not a dict
                self.log(f"Processing section '{section_name}' with default required status: {required}")

            zone_section["zonedelete"] = not required
            # zone_section["zonedelete"] = required

            if isinstance(section_data, list):
                zone_section["zonearray"] = self.process_zone_array(section_data, section_name)
                zone_section["zonekeys"]["item_count"] = len(zone_section["zonearray"])

            elif isinstance(section_data, dict):
                zone_section["zonekeys"] = self.process_zone_keys(section_data)
            else:
                zone_section["zonekeys"][section_name] = section_data

            all_sections.append(zone_section)

        zone_structure = {"zones": all_sections}
        return zone_structure

    def process_zone_keys(self, data: dict[str, Any]) -> dict[str, Any]:
        processed_keys = {}
        list_type = data.get("list_type")

        for key, value in data.items():
            if key == "list_type" or key == "required_section":
                continue

            if isinstance(value, list) and list_type:
                processed_keys[key] = self.format_list_as_markdown(value, list_type)
            else:
                processed_keys[key] = value

        return processed_keys

    def format_list_as_markdown(self, items: list[Any], list_type: str) -> str:
        if list_type == "bullet":
            formatted_items = [f"- {item}" for item in items]
        elif list_type == "number":
            formatted_items = [f"{i + 1}. {item}" for i, item in enumerate(items)]
        else:
            return str(items)

        return "MD: " + "\n".join(formatted_items)

    # def process_zone_array(self, data: List[Any]) -> List[Dict[str, Any]]:
    #     processed_array = []
    #     for item in data:
    #         if not isinstance(item, dict):
    #             processed_array.append(item)
    #             continue

    #         element = {}
    #         child_arrays = []

    #         for key, value in item.items():
    #             if isinstance(value, list):
    #                 element[key] = self.format_list_as_markdown(value, "bullet")
    #             else:
    #                 element[key] = value

    #         if child_arrays:
    #             element["zonechildarray"] = child_arrays

    #         processed_array.append(element)

    #     return processed_array

    # def process_zone_array(self, data: List[Any], section_name: str = "") -> List[Any]:
    #     processed_array = []

    #     # Special handling for contract_obligations
    #     if section_name == "contract_obligations":
    #         for i in range(0, len(data), 2):
    #             row_parties = data[i:i+2]
    #             row = []
    #             for party in row_parties:
    #                 if not isinstance(party, dict):
    #                     continue
    #                 element = {}
    #                 element["party_name"] = party.get("party_name", "")
    #                 obligations = party.get("obligations", [])
    #                 if isinstance(obligations, list):
    #                     # Convert to MD bullet points
    #                     element["obligations"] = self.format_list_as_markdown(obligations, "bullet")
    #                 else:
    #                     element["obligations"] = obligations
    #                 row.append(element)
    #             processed_array.append(row)  # row may have 1 or 2 parties
    #         return processed_array

    #     # Default processing for other sections
    #     for item in data:
    #         if not isinstance(item, dict):
    #             processed_array.append(item)
    #             continue

    #         element = {}
    #         for key, value in item.items():
    #             if isinstance(value, list):
    #                 element[key] = self.format_list_as_markdown(value, "bullet")
    #             else:
    #                 element[key] = value

    #         processed_array.append(element)

    #     return processed_array

    def process_zone_array(self, data: list[Any], section_name: str = "") -> list[Any]:
        processed_array = []

        # Special handling for contract_obligations
        if section_name == "contract_obligations":
            # First, group parties 2 per row
            rows = []
            for i in range(0, len(data), 2):
                row_parties = data[i : i + 2]
                row = []
                for party in row_parties:
                    if not isinstance(party, dict):
                        continue
                    element = {
                        "party_name": party.get("party_name", ""),
                        "obligations": self.format_list_as_markdown(party.get("obligations", []), "bullet"),
                    }
                    row.append(element)
                rows.append(row)

            # Convert the grouped rows into a single MD table
            md_table = self.format_contract_obligations_table(rows)
            return [{"party_table": "MD: " + md_table}]  # Add MD: prefix

        # Default processing for other sections
        for item in data:
            if not isinstance(item, dict):
                processed_array.append(item)
                continue

            element = {}
            for key, value in item.items():
                if isinstance(value, list):
                    element[key] = self.format_list_as_markdown(value, "bullet")
                else:
                    element[key] = value

            processed_array.append(element)

        return processed_array

    def format_contract_obligations_table(self, zonearray: list[list[dict[str, Any]]]) -> str:
        """Convert the zonearray of contract obligations into an MD table
        - 2 parties per row
        - Party name centered on top
        - Obligations as bullet points
        - Vertical line between left and right party
        """
        md_output = '<div style="font-family: Arial; font-size: 10px;">\n'

        for row in zonearray:
            md_output += '<table style="width: 100%; border-collapse: collapse;"><tr>\n'
            for idx, party in enumerate(row):
                party_name = party.get("party_name", "")
                obligations_md = party.get("obligations", "")

                # Remove MD prefix if exists
                if obligations_md.startswith("MD:"):
                    obligations_md = obligations_md[3:].strip()

                # Add border-right for left column if there are 2 columns
                border_style = (
                    "border-right: 1px solid #000; padding-right: 8px;"
                    if idx == 0 and len(row) == 2
                    else "padding: 8px;"
                )

                md_output += f'<td style="vertical-align: top; width: 50%; {border_style}">'
                # Center party name without bold
                md_output += f'<div style="text-align: center; font-weight: normal;">{party_name}</div>\n'
                md_output += '<ul style="font-family: Arial; font-size: 10px; margin: 0; padding-left: 20px;">\n'

                for line in obligations_md.split("\n"):
                    line_text = line.removeprefix("- ")
                    md_output += f'<li style="font-family: Arial; font-size: 13px;">{line_text}</li>\n'

                md_output += "</ul></td>\n"

            # Add empty cell if row has only 1 party
            if len(row) == 1:
                md_output += '<td style="width:50%"></td>\n'

            md_output += "</tr></table>\n"

        md_output += "</div>"
        return md_output
