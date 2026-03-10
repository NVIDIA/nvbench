import os

project = "cuda.bench Python API"
author = "NVIDIA Corporation"

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.autosummary"]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": True}

release = "0.2.0"

######################################################

# -- Options for HTML output -------------------------------------------------

html_theme = "nvidia_sphinx_theme"

html_logo = "_static/nvidia-logo.png"

html_baseurl = (
    os.environ.get("NVBENCH_DOCS_BASE_URL", "https://nvidia.github.io/nvbench/").rstrip(
        "/"
    )
    + "/"
)

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVIDIA/nvbench",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
    "navigation_depth": 4,
    "show_toc_level": 2,
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "sidebar_includehidden": True,
    "collapse_navigation": False,
    "switcher": {
        "json_url": f"{html_baseurl}nv-versions.json",
        "version_match": release,
    },
}

html_static_path = ["_static"] if os.path.exists("_static") else []

# Images directory
if os.path.exists("img"):
    html_static_path.append("img")
