import os
import sys

if sys.version_info > (3, 11):
    import tomllib
else:
    import tomli as tomllib


def find_file_in_directory(directory_path: str, target_name: str) -> str | bool:
    """Search for a file with target_name (ignoring extension) in the given directory.
    Returns the absolute path to the file if found, else False.
    """
    directory_path = os.path.abspath(directory_path)
    for entry in os.listdir(directory_path):
        entry_path = os.path.join(directory_path, entry)
        if os.path.isfile(entry_path) and os.path.splitext(entry)[0] == target_name:
            return entry_path
    return False


def find_file_in_parents(start_path: str, target_name: str, levels: int = None) -> str | None:
    """Traverse up from start_path to find a file with target_name (ignoring extension).
    Uses find_file_in_directory for each folder found.
    levels: maximum number of upward traversals (None for unlimited).
    Returns the absolute path to the file if found, else None.
    """
    current_dir = os.path.abspath(start_path)
    traversals = 0
    while True:
        result = find_file_in_directory(current_dir, target_name)
        if result:
            return result
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir or (levels is not None and traversals >= levels):
            break
        current_dir = parent_dir
        traversals += 1
    return None


def open_path_configs(toml_path: str) -> dict | None:
    """Open and parse a TOML configuration file.
    Returns the parsed dictionary if successful, else None.
    """
    try:
        with open(toml_path, "rb") as f:
            config = tomllib.load(f)
        return config
    except Exception as e:
        print(f"Error reading TOML file at {toml_path}: {e}")
        return None
