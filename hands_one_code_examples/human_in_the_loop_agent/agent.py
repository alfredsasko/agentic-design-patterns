from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path


# agent.py is located at:
# <repo>/hands_one_code_examples/13_1_human_in_the_loop_agent/agent.py
#
# parents[2] therefore points to <repo>.
REPO_ROOT = Path(__file__).resolve().parents[2]

repo_root_string = str(REPO_ROOT)
if repo_root_string not in sys.path:
    sys.path.insert(0, repo_root_string)


example_module = import_module(
    "hands_one_code_examples.13_1_human_in_the_loop"
)

root_agent = example_module.root_agent