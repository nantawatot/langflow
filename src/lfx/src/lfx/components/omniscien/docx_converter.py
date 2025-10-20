import os
import queue
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

from langflow.custom import Component
from langflow.field_typing import Data
from langflow.inputs import BoolInput, IntInput, MessageInput
from langflow.io import Output
from langflow.omniscien_backend.docx_converter.docx_converter import (
    convert_doc_to_json,
    convert_doc_to_md,
    convert_docx_to_other_types,
    insert_file_to_main_job,
)


class DocxConverter(Component):
    display_name = "DOCX File Converter"
    description = "Allows you to convert word doc into different document types"
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "Omniscien"
    name = "DocxConverter"

    inputs = [
        BoolInput(
            display_name="Enable Overwrite",
            name="enable_overwrite",
        ),
        # BoolInput(
        #     display_name="DOCX (Microsoft Word)",
        #     name="docx",
        # ),
        BoolInput(
            display_name=" JSON (JavaScript Object Notation)",
            name="json",
        ),
        BoolInput(
            display_name="PDF (Portable Document Format)",
            name="pdf",
        ),
        BoolInput(
            display_name="TXT (Plain Text)",
            name="txt",
        ),
        BoolInput(
            display_name="XML (Extensible Markup Language)",
            name="xml",
        ),
        BoolInput(
            display_name="HTML (HyperText Markup Language)",
            name="html",
        ),
        BoolInput(
            display_name="MD (Markdown)",
            name="md",
        ),
        MessageInput(
            name="path_to_doc",
            display_name="Path To DOCX",
            info="This is a custom component Input",
            value="hello_world",
            tool_mode=True,
        ),
        MessageInput(
            name="output_file_name",
            display_name="Output File Name",
            info="This is a custom component Input",
            value="Hello, World!",
            tool_mode=True,
            advanced=True,
        ),
        IntInput(
            name="max_parallel_tasks",
            display_name="Max Parallel Tasks",
            info="Maximum number of tasks to run in parallel.",
            value=2,  # Default value
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Output", name="output", method="build_output"),
    ]

    # This is a helper function that will be executed by the thread pool workers.
    def _conversion_worker(self, log_queue, conversion_type, input_file, output_file_path):
        """A worker method that calls the appropriate conversion function
        based on the conversion type.
        """
        job_id = self.graph.context["jobpayload"]["jobid"]
        json_data = self.graph.context.get("jsonOutput")

        if conversion_type == "json":
            convert_doc_to_json(job_id, json_data, self.path_to_doc.text, log_queue, output_file_path)

        elif conversion_type == "md":
            convert_doc_to_md(job_id, self.path_to_doc.text, log_queue, input_file, output_file_path)

        elif conversion_type == "docx":
            insert_file_to_main_job(job_id, log_queue, input_file)

        else:  # All other types handled by the same function
            convert_docx_to_other_types(job_id, log_queue, input_file, output_file_path, conversion_type)

    async def build_output(self) -> Data:
        ########################## PATH AND FILE PROCESSING #########################
        # Base output file path to be used to retrieve the path for output (by removing the last 3 parts and adding output/contracts)
        base_output_file_path = self.graph.context["jobpayload"]["jobprofile"]["datatoprocess"]["inputfilepath"]
        self.log(f"Base Output File Path: {base_output_file_path}")

        # Convert to Path object
        path_obj = Path(base_output_file_path)

        # Remove the last 3 parts (e.g., /input/contracts/filename)
        base_output_file_path = path_obj.parents[2]  # parents[0] = filename, [1] = 'contracts', [2] = 'input'
        self.log(f"Base Output File Path After Cut: {base_output_file_path}")

        # Construct the output file path by adding "output/contracts"
        output_file_path = os.path.join(base_output_file_path, "output", "contracts")
        self.log(f"Output File Path: {output_file_path}")
        self.log(f"output_file_path type: {type(output_file_path)}")

        # Get the input file path from the component input
        input_file = self.path_to_doc.text

        ######################## FORM RETRIEVAL AND PROCESSING #########################
        # Check payload to see which document type is being checked in the form
        document_type_form = self.graph.context["jobpayload"]["jobprofile"]["processingoptions"]["variables"]

        ######################## OVERWRITE FORM OPTIONS IF ENABLED #########################
        # If overwrite is enabled, override the document type form with the component inputs
        if self.enable_overwrite:
            # Override with component inputs
            document_type_form["chkOutputPDF"]["value"] = 1 if self.pdf else 0
            document_type_form["chkOutputXML"]["value"] = 1 if self.xml else 0
            document_type_form["chkOutputHTML"]["value"] = 1 if self.html else 0
            document_type_form["chkOutputJSON"]["value"] = 1 if self.json else 0
            document_type_form["chkOutputPlainText"]["value"] = 1 if self.txt else 0
            document_type_form["chkOutputMicrosoftWord"]["value"] = 1 if self.docx else 0
            document_type_form["chkOutputMD"]["value"] = 1 if self.md else 0

            self.log(f"[Overwrite Enabled] Document type form: {document_type_form}")

        else:
            self.log(f"[Overwrite Not Enabled, Using Original] Document type form: {document_type_form}")

        # Create a thread-safe queue for logging
        log_queue = queue.Queue()

        # Create a list of tasks to be processed by the worker pool
        tasks_to_process = []

        # Iterate through the document types and add tasks to the list
        for key, info in document_type_form.items():
            if info.get("value") != 1:
                continue  # Skip non-selected types

            # Based on the document type, add the appropriate conversion task
            match key:
                case "chkOutputHTML":
                    tasks_to_process.append(("html", input_file, output_file_path))
                    self.log("Output format selected: HTML")

                case "chkOutputPDF":
                    tasks_to_process.append(("pdf", input_file, output_file_path))
                    self.log("Output format selected: PDF")

                case "chkOutputXML":
                    tasks_to_process.append(("xml", input_file, output_file_path))
                    self.log("Output format selected: XML")

                case "chkOutputJSON":
                    tasks_to_process.append(("json", input_file, output_file_path))
                    self.log("Output format selected: JSON")

                case "chkOutputPlainText":
                    tasks_to_process.append(("txt", input_file, output_file_path))
                    self.log("Output format selected: Plain Text")

                case "chkOutputMicrosoftWord":
                    tasks_to_process.append(("docx", input_file, output_file_path))  # A new conversion type
                    self.log("Output format selected: Microsoft Word")

                case "chkOutputMD":
                    tasks_to_process.append(("md", input_file, output_file_path))
                    self.log("Output format selected: Markdown")

                case _:
                    self.log(f"Unknown output format: {key}")

        # Use a ThreadPoolExecutor to limit the number of parallel tasks
        # The `max_workers` is set by the component input
        with ThreadPoolExecutor(max_workers=self.max_parallel_tasks) as executor:
            # Create a list of futures (the results of the tasks)
            futures = [
                executor.submit(self._conversion_worker, log_queue, task[0], task[1], task[2])
                for task in tasks_to_process
            ]

            # Wait for all futures to complete
            wait(futures)

        # After all threads are done, process the logs from the queue
        while not log_queue.empty():
            log_message = log_queue.get()
            self.log(log_message)

        return Data(value="File conversions finished successfully")
