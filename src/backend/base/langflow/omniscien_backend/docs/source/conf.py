# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root (three levels up) to sys.path so autodoc can find modules
sys.path.insert(0, os.path.abspath("../../../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Omniscien Backend"  # Project name shown in the docs
copyright = "2025, OTAI"  # Copyright notice
author = "OTAI"  # Author name

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Auto-generate documentation from Python docstrings
    "sphinxcontrib.autodoc_pydantic",  # Pydantic model support for autodoc
    "sphinx.ext.napoleon",  # Support for NumPy and Google-style docstrings
    "sphinxcontrib.confluencebuilder",  # Publish documentation directly to Confluence
]

# -- Confluence output configuration -----------------------------------------
# https://sphinxcontrib-confluencebuilder.readthedocs.io/

confluence_publish = True  # Enable publishing to Confluence
confluence_server_url = "http://172.17.101.29:8091/"  # Confluence server URL

# Authentication options – asking interactively (instead of hardcoding credentials)
confluence_ask_user = True
confluence_ask_password = True

# Target Confluence space and parent page
confluence_space_key = "OTAI"  # Confluence space key (short name)
confluence_parent_page = "LangFlow Documentation"  # Parent page under which pages will be created

# Page purge settings
confluence_purge = False  # If True, will delete old child pages not in this build

# Autodoc Pydantic options
autodoc_pydantic_model_show_json = False  # Whether to show JSON schemas for models
autodoc_pydantic_field_show_default = False  # Whether to show default values for fields

# -- Templates and patterns --------------------------------------------------

templates_path = ["_templates"]  # Path for custom HTML templates
exclude_patterns = []  # Patterns to ignore when looking for source files

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"  # HTML theme for Sphinx docs
html_static_path = ["_static"]  # Path for static files (CSS, JS, images, etc.)
