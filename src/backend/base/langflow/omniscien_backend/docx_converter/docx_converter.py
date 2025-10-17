import json
import os
import shutil
import subprocess
import tempfile

import requests


def convert_docx_to_other_types(job_id: str, log_queue, input_file: str, output_file_path: str, file_type: str) -> str:
    # Use a temporary directory for LibreOffice user profile to prevent race conditions
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_profile_dir = os.path.join(temp_dir, "profile")

        # Run LibreOffice headless conversion
        try:
            # Updated subprocess.run to capture stdout and stderr for better debugging and to use a unique profile directory
            result = subprocess.run(
                [
                    "libreoffice",
                    "--headless",
                    "--nologo",
                    "--nofirststartwizard",
                    "--convert-to",
                    file_type,
                    input_file,
                    "--outdir",
                    output_file_path,
                    "-env:UserInstallation=file://" + temp_profile_dir,
                ],
                check=True,
                capture_output=True,
                text=True,  # This ensures output is a string instead of bytes
            )

            # Get base filename without extension
            base_name = os.path.splitext(os.path.basename(input_file))[0]  # → "20250822_041417_Contract-Summary"

            # Construct final PDF path
            # The file type is defined above, so it will be saved as the respective file type
            path_to_download_merged_file = os.path.join(output_file_path, base_name + f".{file_type}")

            # Insert converted file into main job
            insert_file_to_main_job(job_id, log_queue, path_to_download_merged_file)

            log_queue.put(f"File Type Converted Successfully: {file_type}")

        except subprocess.CalledProcessError as e:
            error_message = f"Conversion failed: {e}"
            log_queue.put(error_message)
            log_queue.put(f"STDOUT: {e.stdout}")  # Log stdout from the failed process
            log_queue.put(f"STDERR: {e.stderr}")  # Log stderr from the failed process
            return error_message

        finally:
            log_queue.put(f"Attempting to remove temporary directory: {temp_dir}")
            try:
                shutil.rmtree(temp_dir)
                log_queue.put(f"Successfully removed temporary directory: {temp_dir}")
            except OSError as e:
                log_queue.put(f"Error removing temporary directory {temp_dir}: {e.strerror}")

    return f"File converted successfully to {output_file_path}"


def convert_doc_to_json(job_id: str, json_data, path_to_doc: str, log_queue, output_file_path: str) -> str:
    # Ensure directory exists
    os.makedirs(output_file_path, exist_ok=True)

    input_file = path_to_doc  # e.g. /contracts/20250822_041417_Contract-Summary.docx
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(input_file))[0]  # → "20250822_041417_Contract-Summary"
    path_to_json_file = os.path.join(output_file_path, base_name + ".json")

    # Write dictionary to JSON file
    with open(path_to_json_file, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, indent=2)

    # Check if file was created successfully
    if not os.path.exists(path_to_json_file):
        error_message = f"Failed to create JSON file at: {path_to_json_file}"
        log_queue.put(error_message)
        raise Exception(error_message)

    # insert the JSON file into the main job
    insert_file_to_main_job(job_id, log_queue, path_to_json_file)

    return f"JSON file created at: {path_to_json_file}"


def convert_doc_to_md(job_id: str, path_to_doc: str, log_queue, input_file: str, output_file_path: str) -> str:
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(input_file))[0]  # → "20250822_041417_Contract-Summary"
    # Construct final PDF path
    path_to_md_file = os.path.join(output_file_path, base_name + ".md")

    try:
        # Updated subprocess.run to capture stdout and stderr for better debugging
        result = subprocess.run(
            [
                "pandoc",
                path_to_doc,  # Input DOCX file
                "-f",
                "docx",
                "-t",
                "gfm",
                "-o",
                path_to_md_file,
                "--wrap=none",
                "--extract-media=./media",
            ],
            check=True,
            capture_output=True,
            text=True,  # This ensures output is a string instead of bytes
        )

        # Insert the converted Markdown file into the main job
        insert_file_to_main_job(job_id, log_queue, path_to_md_file)

    except subprocess.CalledProcessError as e:
        error_message = f"Conversion failed: {e}"
        log_queue.put(error_message)
        log_queue.put(f"STDOUT: {e.stdout}")  # Log stdout from the failed process
        log_queue.put(f"STDERR: {e.stderr}")  # Log stderr from the failed process
        return error_message

    return "Markdown conversion Complete"


def insert_file_to_main_job(job_id: str, log_queue, file_path: str) -> str:
    """This function inserts a file into the main job using the Language Studio API.
    This is to make sure the files are all available for download in the main job.
    By default, the files are saved in the output/contracts folder. Hence we need to pass it back to the main job.
    """
    log_queue.put(f"[INSERT FILE TO MAIN FUNCTION] JOB ID: {job_id}")

    api_api_url = "http://devdemo.languagestudio.com:4000/lsrestapi/v6/jobfile/insert"

    params = {
        "jobid": job_id,
        "filepath": file_path,
        "jobfiletypecode": 2,
    }

    response = requests.get(api_api_url, params=params)
    log_queue.put(f"Response from API: {response.text}")

    if response.status_code != 200:
        log_queue.put(f"Error: {response.status_code} - {response.text}")
        raise Exception(f"Failed to retrieve document template: {response.text}")

    return "File inserted to main job successfully"
