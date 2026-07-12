from __future__ import annotations

import sys
from pathlib import Path

# adk web loads this package directly, so ensure the repository root is importable
# before importing shared helpers or the example module.
REPO_ROOT = Path(__file__).resolve().parents[2]
repo_root_string = str(REPO_ROOT)
if repo_root_string not in sys.path:
    sys.path.insert(0, repo_root_string)

from hands_one_code_examples._shared.adk_web_loader import load_root_agent


root_agent = load_root_agent("hands_one_code_examples.14_1_rag")
