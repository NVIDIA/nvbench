import os
import sys

project = "NVBench C++ API"
author = "NVIDIA Corporation"

extensions = ["breathe"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "_doxygen"]

release = "0.0.1"

_here = os.path.abspath(os.path.dirname(__file__))
_doxygen_xml = os.path.join(_here, "_doxygen", "xml")

breathe_projects = {"nvbench": _doxygen_xml}
breathe_default_project = "nvbench"
breathe_domain_by_extension = {"cuh": "cpp", "cxx": "cpp", "cu": "cpp"}

sys.path.insert(0, os.path.abspath(os.path.join(_here, "..", "..")))


def _patch_breathe_namespace_declarations() -> None:
    try:
        import breathe.renderer.sphinxrenderer as sphinxrenderer
        from docutils import nodes
        from sphinx import addnodes
    except Exception:
        return

    original = sphinxrenderer.SphinxRenderer.handle_declaration

    def handle_declaration(self, nodeDef, declaration, *args, **kwargs):
        is_namespace = getattr(nodeDef, "kind", None) == "namespace"
        if not is_namespace:
            return original(self, nodeDef, declaration, *args, **kwargs)

        name = (declaration or "").strip()
        if name.startswith("namespace "):
            name = name[len("namespace ") :].strip()
        if not name:
            name = "<anonymous>"

        keyword = addnodes.desc_sig_keyword("namespace", "namespace")
        sig_name = addnodes.desc_sig_name(name, name)
        return [keyword, nodes.Text(" "), sig_name]

    sphinxrenderer.SphinxRenderer.handle_declaration = handle_declaration


def setup(app):
    _patch_breathe_namespace_declarations()


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
