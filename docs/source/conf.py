# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import version as package_version

# Add the package to the path so Sphinx can import it
sys.path.insert(0, os.path.abspath("../../src"))

# Keep MNE's progress messages out of the example pages
os.environ["MNE_LOGGING_LEVEL"] = "WARNING"

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "rsa-nd"
copyright = "2026, Weiyong Xu"
author = "Weiyong Xu"

release = package_version("rsa-nd")
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Auto-generate API docs from docstrings
    "sphinx.ext.napoleon",  # Support for NumPy-style docstrings
    "sphinx.ext.intersphinx",  # Link to other project docs (MNE-Python)
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx_gallery.gen_gallery",  # Auto-generate example gallery
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Napoleon settings (NumPy-style docstrings) ------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Autodoc settings --------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# -- Sphinx-Gallery configuration -------------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",  # Path to example scripts
    "gallery_dirs": "auto_examples",  # Where to save generated gallery
    "filename_pattern": "/plot_",  # Only process files starting with plot_
    "ignore_pattern": r"__init__\.py",  # Ignore __init__.py
    "within_subsection_order": "FileNameSortKey",  # Overview first
    "download_all_examples": False,  # Don't create a zip of all examples
    "remove_config_comments": True,  # Remove sphinx-gallery config comments
    "expected_failing_examples": [],  # List of examples expected to fail
    "backreferences_dir": None,  # Disable backreferences for now
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "mne": ("https://mne.tools/stable/", None),  # Link to MNE-Python docs
}
