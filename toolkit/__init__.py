"""Shared helpers for open-ml-notebooks.

Three modules, deliberately small:

    style     one chart theme, so 44 notebooks look like one book
    data      cached downloads with the provenance rule enforced
    evaluate  scoring with uncertainty attached

Notebooks pick these up by walking up to the repo root:

    import sys, pathlib
    ROOT = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents]
                if (p / "toolkit").is_dir())
    sys.path.insert(0, str(ROOT))

Author: Elyes Lounissi
"""

__all__ = ["style", "data", "evaluate"]
__author__ = "Elyes Lounissi"
