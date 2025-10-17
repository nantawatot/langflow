OpenManus requires at least Python 3.12 to function.

To install, go to the ```/src/backend/base/langflow/omniscien_backend/open_manus/``` path

1. Create a uv venv in Python 3.12 with
```uv venv --python 3.12```

2. Activate the environment (while in the same DIR)
```source .venv/bin/activate```

3. Install the project's dependencies (in the same path)
```uv pip install -r requirements.txt```

4. Ensure that playwright (the browser automation tool) is installed to perform browser based-search.
```npx playwright install```

5. Update the mappings of the paths.toml file in
```/src/backend/base/langflow/omniscien_backend/open_manus/utils/paths.toml```

    To the Absolute Paths for the
    - Python executable (version 3.12)
    - main.py
    - results output filepath

    Below is an example for the content of the paths.toml file:
    ```
    result = "/home/user/Documents/otai-project-langflow/src/backend/base/langflow/omniscien_backend/open_manus/results/"

    main = "/home/user/Documents/otai-project-langflow/src/backend/base/langflow/omniscien_backend/open_manus/main.py"

    python = "/home/user/Documents/otai-project-langflow/src/backend/base/langflow/omniscien_backend/open_manus/.venv/bin/python3.12"
    ```
Your Version of Langflow should now be ready for the Langflow OpenManus component! :)