import os
import sys

# Point to the root directory
sys.path.insert(0, os.path.abspath('../..'))

# Point explicitly to the correct subdirectories
sys.path.insert(0, os.path.abspath('../../initial_condition'))
sys.path.insert(0, os.path.abspath('../../sim_folder/analysis'))
sys.path.insert(0, os.path.abspath('../../sim_folder/run'))
sys.path.insert(0, os.path.abspath('../../tests'))

project = 'CHEQUP'
copyright = '2026, Thibault Benahmed, Remi Lehe, Maxence Thevenet, Christian McCombs, Alexander Sinn'
author = 'Thibault Benahmed, Remi Lehe, Maxence Thevenet, Christian McCombs, Alexander Sinn'
release = 'v0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # Core library for capturing docstrings
    'sphinx.ext.napoleon',      # Support for Google/NumPy docstrings
    'sphinx.ext.viewcode',      # Add links to highlighted source code
    'sphinx.ext.mathjax',       # Render math formulas
    'nbsphinx',                 # Integrate Jupyter Notebooks
    'sphinx_copybutton',        # Add a "Copy" button to code blocks
    'sphinxcontrib.bibtex',     # Bibliography management (HiPACE++ style)
]

autodoc_mock_imports = []

# Configure bibliography file
bibtex_bibfiles = ['refs.bib']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']