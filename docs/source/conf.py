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
import os
import sys
import shutil

ALR_ROOT = os.path.abspath(os.path.join("..", ".."))
EXPERIMENT_DIR = os.path.join(ALR_ROOT, "experiments")
sys.path.insert(0, ALR_ROOT)

# -- Project information -----------------------------------------------------

project = "alr"
copyright = "2020, Jia Hong Fong"
author = "Jia Hong Fong"


# The full version, including alpha/beta/rc tags
def get_version(rel_path):
    for line in open(rel_path).read().splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


release = get_version(os.path.join(ALR_ROOT, "alr", "__init__.py"))


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "sphinx.ext.napoleon",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# html_theme = 'nature'
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- User added content ------------------------------------------------------
# ADDED: show init docstring in class documentation
autoclass_content = "both"
# ADDED: for rtd.io
master_doc = "index"
# ADDED: import experiments from ../../experiments
shutil.rmtree("experiments", ignore_errors=True)
shutil.copytree(EXPERIMENT_DIR, "experiments")
