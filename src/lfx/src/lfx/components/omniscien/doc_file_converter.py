import json
import os
from datetime import datetime
from pathlib import Path

import requests
from langflow.custom import Component
from langflow.inputs import MessageInput
from langflow.io import Output
from langflow.schema import Message


class DocFileConverter(Component):
    display_name = "Doc File Converter"
    description = "Allows you to convert word doc into different document types"
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "Omniscien"
    name = "DocFileConverter"

    inputs = [
        MessageInput(
            name="path_to_doc",
            display_name="Path To Doc",
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
    ]

    outputs = [
        Output(display_name="Output", name="output", method="build_output"),
    ]

    def build_output(self) -> Message:
        # Get user ID from the context
        user_id = self.graph.context["jobpayload"]["jobprofile"]["processingoptions"]["userid"]

        # Full file path to be uploaded
        file_path = self.path_to_doc.text

        # Check if file exists
        if not os.path.exists(file_path):
            error_message = f"File not found: {file_path}"
            self.log(error_message)

        # Check payload to see which document type is being checked in the form
        document_type_form = self.graph.context["jobpayload"]["jobprofile"]["processingoptions"]["variables"]

        self.log(f"Document type form: {document_type_form}")

        # Iterate through the document types and perform actions based on the selected types
        for key, info in document_type_form.items():
            if info.get("value") != 1:
                continue  # Skip non-selected types

            # Based on the document type, call the appropriate conversion method
            match key:
                case "chkOutputHTML":
                    self.convert_doc_to_html(user_id, file_path)
                    self.log("Output format selected: HTML")

                case "chkOutputPDF":
                    self.convert_doc_to_pdf(user_id, file_path)
                    self.log("Output format selected: PDF")

                case "chkOutputXML":
                    self.convert_doc_to_xml(user_id, file_path)
                    self.log("Output format selected: XML")

                case "chkOutputJSON":
                    self.convert_doc_to_json()
                    self.log("Output format selected: JSON")

                case "chkOutputPlainText":
                    self.convert_doc_to_text(user_id, file_path)
                    self.log("Output format selected: Plain Text")

                case "chkOutputMicrosoftWord":
                    self.convert_doc_to_word()
                    self.log("Output format selected: Microsoft Word")

                case _:
                    print(f"Unknown output format: {key}")

        return Message(text="Document conversion completed successfully")

    def convert_doc_to_pdf(self, user_id: int, file_path: str) -> Message:
        url = "http://devdemo.languagestudio.com:4000/lsrestapi/v6/jobsubmit"

        # JSON payload as dict
        json_payload = {
            "header": {
                "requesttype": "job.submit",
                "requesttask": "convertfileformat",
                "authentication": {"userid": user_id},
            },
            "body": {
                "jobprofile": {
                    "processingoptions": {
                        "sourceformat": "doc",
                        "targetformat": "pdf",
                        "pagerangemode": 0,
                        "pagerange": "",
                        "pdfstandardcompliance": 1,
                        "fontembeddingmode": 0,
                        "fontembeddingfull": False,
                        "imagecompressionjpegquality": 50,
                        "debug": {"logjobsteps": 1},
                    }
                }
            },
        }

        # Prepare the files for multipart/form-data
        with open(file_path, "rb") as file:
            files = {
                "json": (None, json.dumps(json_payload)),  # This simulates the curl --form 'json=...'
                "files": (
                    os.path.basename(file_path),
                    file,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ),
            }

            # Perform the request
            response = requests.post(url, files=files)

            self.obtain_job_id_from_converter_output(response.text, "pdf")

        return Message(text="PDF conversion not implemented yet")

    def convert_doc_to_text(self, user_id: int, file_path: str) -> Message:
        url = "http://devdemo.languagestudio.com:4000/lsrestapi/v6/jobsubmit"

        # JSON payload as dict
        json_payload = {
            "header": {
                "requesttype": "job.submit",
                "requesttask": "file.convert.format",
                "authentication": {"userid": user_id},
            },
            "body": {
                "jobprofile": {
                    "processingoptions": {
                        "sourceformat": "docx",
                        "targetformat": "txt",
                        "pagerangemode": 0,
                        "pagerange": "",
                        "bidimark": False,
                        "encodingstandard": "UTF-8",
                        "headerfootermode": 1,
                        "forcepagebreak": False,
                        "preservetablelayout": False,
                        "simplifylisttable": False,
                        "endofline": "rn",
                        "commitchangetracking": 0,
                        "debug": {"logjobsteps": 1},
                    }
                }
            },
        }

        # Prepare the files for multipart/form-data
        with open(file_path, "rb") as file:
            files = {
                "json": (None, json.dumps(json_payload)),  # This simulates the curl --form 'json=...'
                "files": (
                    os.path.basename(file_path),
                    file,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ),
            }

            # Perform the request
            response = requests.post(url, files=files)

        return Message(text="Text conversion not implemented yet")

    def convert_doc_to_json(self) -> Message:
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Obtain output file path
        base_output_file_path = self.graph.context["jobpayload"]["jobprofile"]["datatoprocess"]["inputfilepath"]
        self.log(f"Base Output File Path: {base_output_file_path}")

        # Convert to Path object
        path_obj = Path(base_output_file_path)

        # Remove the last 3 parts (e.g., /input/contracts/filename)
        base_output_file_path = path_obj.parents[2]  # parents[0] = filename, [1] = 'contracts', [2] = 'input'
        self.log(f"Base Output File Path After Cut: {base_output_file_path}")

        output_file_path = os.path.join(base_output_file_path, "output", "contracts")
        self.log(f"Output File Path: {output_file_path}")

        output_file_name = f"{timestamp}_{self.output_file_name.text}.json"

        # Ensure directory exists
        os.makedirs(output_file_path, exist_ok=True)

        # Full path to the file
        file_path = os.path.join(output_file_path, output_file_name)

        # Write dictionary to JSON file
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(self.graph.context.get("jsonOutput"), json_file, indent=2)
        self.log(f"JSON file created at: {file_path}")

        # Check if file was created successfully
        if not os.path.exists(file_path):
            error_message = f"Failed to create JSON file at: {file_path}"
            self.log(error_message)
            raise Exception(error_message)

        # Call API to insert JSON into Job
        # Obtain Job ID from Context Graph (Global Variable)
        job_id = self.graph.context["jobpayload"]["jobid"]
        self.log(f"JOB ID: {job_id}")

        api_url = "http://devdemo.languagestudio.com:4000/lsrestapi/v6/jobfile/insert"

        params = {
            "jobid": job_id,
            "filepath": file_path,
            "jobfiletypecode": 2,
        }

        response = requests.get(api_url, params=params)
        self.log(f"Response from API: {response.text}")

        if response.status_code != 200:
            self.log(f"Error: {response.status_code} - {response.text}")
            raise Exception(f"Failed to retrieve document template: {response.text}")

        return Message(text=f"JSON file created at: {file_path}")

    def convert_doc_to_word(self) -> Message:
        # Obtain Job ID from Context Graph (Global Variable)
        job_id = self.graph.context["jobpayload"]["jobid"]
        self.log(f"JOB ID: {job_id}")

        api_url = "http://devdemo.languagestudio.com:4000/lsrestapi/v6/jobfile/insert"

        params = {
            "jobid": job_id,
            "filepath": self.path_to_doc.text,
            "jobfiletypecode": 2,
        }

        response = requests.get(api_url, params=params)
        self.log(f"Response from API: {response.text}")

        if response.status_code != 200:
            self.log(f"Error: {response.status_code} - {response.text}")
            raise Exception(f"Failed to retrieve document template: {response.text}")

        return Message(text="Word Conversion")

    def convert_doc_to_xml(self, user_id: int, file_path: str) -> Message:
        url = "http://devdemo.languagestudio.com:4000/lsrestapi/v6/jobsubmit"

        # JSON payload as dict
        json_payload = {
            "header": {
                "requesttype": "job.submit",
                "requesttask": "convertfileformat",
                "authentication": {"userid": user_id},
            },
            "body": {
                "jobprofile": {
                    "processingoptions": {
                        "sourceformat": "docx",
                        "targetformat": "xliff",
                        "pagerangemode": 0,
                        "pagerange": "",
                        "sourcelanguage": "ZH",
                        "debug": {"logjobsteps": 1},
                    }
                }
            },
        }

        # Prepare the files for multipart/form-data
        with open(file_path, "rb") as file:
            files = {
                "json": (None, json.dumps(json_payload)),  # This simulates the curl --form 'json=...'
                "files": (
                    os.path.basename(file_path),
                    file,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ),
            }

            # Perform the request
            response = requests.post(url, files=files)

            self.log(f"Response from XML conversion API: {response.text}")

        return Message(text="Text conversion not implemented yet")

    def convert_doc_to_html(self, user_id: int, file_path: str) -> Message:
        url = "http://devdemo.languagestudio.com:4000/lsrestapi/v6/jobsubmit"

        # JSON payload as dict
        json_payload = {
            "header": {
                "requesttype": "job.submit",
                "requesttask": "convertfileformat",
                "authentication": {"userid": user_id},
            },
            "body": {
                "jobprofile": {
                    "processingoptions": {
                        "sourceformat": "docx",
                        "targetformat": "html",
                        "pagerangemode": 0,
                        "pagerange": "",
                        "exportroundtripinformation": True,
                        "cssstylesheettype": 0,
                        "exportfontresources": False,
                        "exportfontsasbase64": False,
                        "allownegativeindent": False,
                        "debug": {"logjobsteps": 1},
                    }
                }
            },
        }

        # Prepare the files for multipart/form-data
        with open(file_path, "rb") as file:
            files = {
                "json": (None, json.dumps(json_payload)),  # This simulates the curl --form 'json=...'
                "files": (
                    os.path.basename(file_path),
                    file,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ),
            }

            # Perform the request
            response = requests.post(url, files=files)

        return Message(text="HTML conversion not implemented yet")

    def obtain_job_id_from_converter_output(self, response: str, file_extension: str) -> Message:
        try:
            # Parse JSON
            data = json.loads(response)

            # Extract the first jobid
            job_id = data["result"]["jobs"][0]["jobid"]

            time.sleep(10)

            self.log(f"[FOR DOC TYPE] Extracted jobid: {job_id} from response: {response}")

            job_id = int(job_id)  # Ensure job_id is an integer

            self.download_job_file(file_extension, job_id)

            return Message(text=str(job_id))

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            self.log(f"Failed to extract jobid: {e} | Response: {response}")
            raise ValueError(f"Unable to extract jobid from response: {response}")

    def download_job_file(self, file_extension: str, convert_job_id: int) -> Message:
        url = "http://devdemo.languagestudio.com:4000/lsrestapi/v6/jobsubmit"

        self.log(f"Converting job ID: {convert_job_id} to file extension: {file_extension}")

        # JSON payload as string
        json_payload_str = json.dumps(
            {
                "header": {"requesttype": "job.download", "authentication": {"userid": 96}},
                "body": {"jobprofile": {"jobidlist": convert_job_id, "filetypes": "TGT"}},
            }
        )

        # Send as multipart/form-data with field 'json'
        files = {
            "json": (None, json_payload_str)  # (filename, content)
        }

        response = requests.post(url, files=files)

        # 1️⃣ Check status code first
        if response.status_code != 200:
            self.log(f"Download failed: {response.status_code} {response.text}")
            # raise ValueError(f"Failed to download file, error message: {response.text} {type(convert_job_id)}")
            return Message(text="Download failed")

        # 2️⃣ Check content type or length
        content_type = response.headers.get("Content-Type", "")
        content_length = len(response.content)

        self.log(f"Content-Type: {content_type}, Content-Length: {content_length} bytes")

        if content_length == 0:
            raise ValueError("Downloaded file is empty — aborting write.")

        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Obtain output file path
        base_output_file_path = self.graph.context["jobpayload"]["jobprofile"]["datatoprocess"]["inputfilepath"]
        self.log(f"Base Output File Path: {base_output_file_path}")

        # Convert to Path object
        path_obj = Path(base_output_file_path)

        # Remove the last 3 parts (e.g., /input/contracts/filename)
        base_output_file_path = path_obj.parents[2]  # parents[0] = filename, [1] = 'contracts', [2] = 'input'
        self.log(f"Base Output File Path After Cut: {base_output_file_path}")

        output_file_path = os.path.join(base_output_file_path, "output", "contracts")
        self.log(f"Output File Path: {output_file_path}")

        output_file_name = f"{timestamp}_{self.output_file_name.text}.{file_extension}"

        # Ensure directory exists
        os.makedirs(output_file_path, exist_ok=True)

        # Full path to the file
        file_path = os.path.join(output_file_path, output_file_name)

        # Write binary content to file
        with open(file_path, "wb") as f:
            f.write(response.content)

        # Check if file was created successfully
        if not os.path.exists(file_path):
            error_message = f"Failed to create PDF file at: {file_path}"
            self.log(error_message)
            raise Exception(error_message)

        self.log("READY TO INSERT FILE TO MAIN JOB")
        self.insert_file_to_main_job(file_path)

        return Message(text="Download job file not implemented yet")

    def insert_file_to_main_job(self, file_path: str) -> Message:
        # Call API to insert JSON into Job
        # Obtain Job ID from Context Graph (Global Variable)
        job_id = self.graph.context["jobpayload"]["jobid"]
        self.log(f"[INSERT FILE TO MAIN FUNCTION] JOB ID: {job_id}")

        api_url = "http://devdemo.languagestudio.com:4000/lsrestapi/v6/jobfile/insert"

        params = {
            "jobid": job_id,
            "filepath": file_path,
            "jobfiletypecode": 2,
        }

        response = requests.get(api_url, params=params)
        self.log(f"Response from API: {response.text}")

        if response.status_code != 200:
            self.log(f"Error: {response.status_code} - {response.text}")
            raise Exception(f"Failed to retrieve document template: {response.text}")

        return Message(text="File inserted to main job successfully")
