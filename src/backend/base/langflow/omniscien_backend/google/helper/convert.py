import ast
import json
import sys
from pathlib import Path

import requests
from datamodel_code_generator import InputFileType, generate


class DiscoveryToOpenAPI:
    """Converter for Google Discovery Documents to OpenAPI 3.0 and Pydantic models"""

    @staticmethod
    def convert_refs(obj, prefix="#/components/schemas/"):
        """Convert Google Discovery $ref format to OpenAPI format"""
        if isinstance(obj, dict):
            new_obj = {}
            for key, value in obj.items():
                if key == "$ref" and isinstance(value, str):
                    # Convert from bare reference to full OpenAPI reference
                    new_obj[key] = f"{prefix}{value}"
                else:
                    new_obj[key] = DiscoveryToOpenAPI.convert_refs(value, prefix)
            return new_obj
        if isinstance(obj, list):
            return [DiscoveryToOpenAPI.convert_refs(item, prefix) for item in obj]
        return obj

    @staticmethod
    def normalize_types(obj):
        """Normalize Google Discovery types to OpenAPI types"""
        if isinstance(obj, dict):
            obj = {k: DiscoveryToOpenAPI.normalize_types(v) for k, v in obj.items()}
            # Convert "string" + "int64" into "integer" + "int64"
            if obj.get("type") == "string" and obj.get("format") == "int64":
                obj["type"] = "integer"
            return obj
        if isinstance(obj, list):
            return [DiscoveryToOpenAPI.normalize_types(i) for i in obj]
        return obj

    @staticmethod
    def discovery_to_openapi(discovery_doc):
        """Convert Google Discovery Document to OpenAPI 3.0"""
        # Extract schemas
        schemas = discovery_doc.get("schemas", {})

        # Convert $ref format
        schemas = DiscoveryToOpenAPI.convert_refs(schemas)

        # Normalize types
        schemas = DiscoveryToOpenAPI.normalize_types(schemas)

        # Create OpenAPI document
        openapi_doc = {
            "openapi": "3.0.0",
            "info": {
                "title": discovery_doc.get("title", "API"),
                "version": discovery_doc.get("version", "v1"),
                "description": discovery_doc.get("description", ""),
            },
            "paths": {},  # Empty paths since we only want schemas
            "components": {"schemas": schemas},
        }

        return openapi_doc

    @staticmethod
    def add_enum_config(py_path: Path):
        """Adds 'class Config: use_enum_values = True' to Pydantic models
        that use Enum types.
        """
        try:
            with open(py_path) as source:
                tree = ast.parse(source.read())

            enum_classes = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id == "Enum":
                            enum_classes.add(node.name)

            models_to_modify = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    is_pydantic_model = any(
                        isinstance(base, ast.Name) and base.id == "BaseModel" for base in node.bases
                    )
                    if is_pydantic_model:
                        for body_item in node.body:
                            if isinstance(body_item, ast.AnnAssign):
                                # Direct annotation
                                if (
                                    isinstance(body_item.annotation, ast.Name)
                                    and body_item.annotation.id in enum_classes
                                ):
                                    models_to_modify.append(node)
                                    break
                                # Annotation within Optional, List, etc.
                                for sub_node in ast.walk(body_item.annotation):
                                    if isinstance(sub_node, ast.Name) and sub_node.id in enum_classes:
                                        models_to_modify.append(node)
                                        break
                                else:
                                    continue
                                break

            for model in models_to_modify:
                # Check if Config class already exists
                has_config = any(isinstance(item, ast.ClassDef) and item.name == "Config" for item in model.body)
                if not has_config:
                    config_class = ast.ClassDef(
                        name="Config",
                        bases=[],
                        keywords=[],
                        body=[
                            ast.Assign(
                                targets=[ast.Name(id="use_enum_values", ctx=ast.Store())],
                                value=ast.Constant(value=True),
                                lineno=0,
                                col_offset=0,
                            )
                        ],
                        decorator_list=[],
                    )
                    model.body.insert(0, config_class)

            with open(py_path, "w") as source:
                source.write(ast.unparse(tree))
            print(f"Added Enum config to Pydantic models in {py_path}")
        except Exception as e:
            print(f"Error during post-processing of {py_path}: {e}")

    @staticmethod
    def convert(api_url: str, output_filename: str = "output", output_dir: Path | None = None, save_json: bool = False):
        """Convert Google Discovery Document to OpenAPI and generate Pydantic models

        Args:
            api_url: URL of the Google Discovery Document
            output_filename: Base name for output files (without extension)
            output_dir: Directory for output files (defaults to current directory)
            save_json: Whether to save the OpenAPI JSON file (defaults to False)

        Returns:
            tuple: (json_path, py_path) - json_path is None if save_json=False
        """
        if output_dir is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        output_json = output_dir / f"{output_filename}.json"
        output_py = output_dir / f"{output_filename}.py"

        try:
            # Download discovery document
            print(f"Downloading discovery document from {api_url}...")
            response = requests.get(api_url)
            response.raise_for_status()
            discovery_doc = response.json()

            # Convert to OpenAPI
            openapi_doc = DiscoveryToOpenAPI.discovery_to_openapi(discovery_doc)

            print(f"Found {len(openapi_doc['components']['schemas'])} schemas")

            # Save OpenAPI JSON only if requested
            json_path = None
            if save_json:
                output_json.write_text(json.dumps(openapi_doc, indent=2))
                print(f"Saved OpenAPI JSON to {output_json}")
                json_path = output_json

            # Convert OpenAPI doc to JSON string for datamodel-codegen
            openapi_content = json.dumps(openapi_doc, indent=2)

            # Generate Pydantic models with circular reference support
            print("Generating Pydantic models...")
            generate(
                input_=openapi_content,
                input_file_type=InputFileType.OpenAPI,
                output=output_py,
            )
            print(f"Generated Pydantic models in {output_py}")

            # Post-process the generated file to add Enum config
            DiscoveryToOpenAPI.add_enum_config(output_py)

            return json_path, output_py

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)


def main():
    api_url = "https://www.googleapis.com/discovery/v1/apis/drive/v3/rest"
    output_filename = "drive"

    DiscoveryToOpenAPI.convert(api_url, output_filename)


if __name__ == "__main__":
    main()
