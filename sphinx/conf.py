# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'pheres'
copyright = '2020, Quentin Soubeyran'
author = 'Quentin Soubeyran'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    "recommonmark",
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

napoleon_use_ivar=False

autodoc_member_order = "bysource"
autodoc_default_flags = ["members", "show-inheritance"]
autodoc_type_aliases = {
    "TypeHint": "pheres.utils.TypeHint",
    "JSONValue": "pheres.typing.JSONValue",
    "JSONArray": "pheres.typing.JSONArray",
    "JSONObject": "pheres.typing.JSONObject",
    "JSONType": "pheres.typing.JSONType",
}

intersphinx_mapping = {
    "https://docs.python.org/3": None,
    "python": ("https://docs.python.org/3", None),
    "https://attrs.org/en/stable/": None,
    "attrs": ("https://attrs.org/en/stable/", None)
}
napoleon_use_ivar=True
napoleon_google_attr_annotations=True
default_role = "any"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']