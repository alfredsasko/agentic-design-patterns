from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Any


def load_root_agent(module_name: str) -> Any:
    """Import an example module and return its exported ``root_agent``."""
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_string = str(repo_root)
    if repo_root_string not in sys.path:
        sys.path.insert(0, repo_root_string)

    module = import_module(module_name)
    return module.root_agent
