"""Test support helpers shared across the repository test suite."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any


def load_module_from_path(module_name: str, module_path: pathlib.Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
