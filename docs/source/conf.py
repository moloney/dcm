# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "dcm"
copyright = "2022, Brendan Moloney"
author = "Brendan Moloney"
release = "0.1.0-dev"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "autodocsumm",
]

templates_path = ["_templates"]
exclude_patterns = []

language = "en"

autodoc_default_options = {"inherited-members": True}
autodoc_typehints_format = "short"
autodoc_member_order = "groupwise"
autodoc_preserve_defaults = True
napoleon_attr_annotations = True
# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

autodoc_default_options = {
    "autosummary": True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

# -- Auto run sphinx-apidoc on each build -----------------------------------
def run_apidoc(_):
    from sphinx.ext.apidoc import main
    import os
    import sys

    doc_src = os.path.abspath(os.path.dirname(__file__))
    code_src = os.path.abspath(os.path.join(doc_src, "../../dcm"))
    sys.path.append(code_src)
    ignore_pattern = os.path.join(code_src, "tests", "**")
    main(["--force", "--separate", "-o", doc_src, code_src, ignore_pattern])


def setup(app):
    app.connect("builder-inited", run_apidoc)
