# Backend Libraries

This directory contains the libraries that power the **Omniscien Backend**.

## Documentation with Sphinx

The project uses [Sphinx](https://www.sphinx-doc.org/) to generate documentation, with publishing support to Confluence.

### How to Build the Documentation

1. Navigate to the `docs` directory:

   ```bash
   cd docs
   ```
2. Generate reStructuredText sources from the backend code:

   ```bash
   sphinx-apidoc -f -o source/ ../../omniscien_backend
   ```
3. Clean previous build artifacts and publish to Confluence:

   ```bash
   make clean && make confluence
   ```

### Notes

* Documentation is generated from Python docstrings (with support for Pydantic models and Google/NumPy-style
  docstrings).
* The output is published to Confluence under the configured **space** and **parent page** (see `conf.py` for details).

