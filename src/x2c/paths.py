"""Where the project folder is. Every module that reads or writes project files (specifications/, chip_build/,
deploy/, reports/, hardware/, ...) uses ROOT from here, so moving a module never changes where files go.

ROOT is the folder holding pyproject.toml, found by walking up from this file (works for the normal editable
install, `uv sync`). $X2C_ROOT overrides it; without either, the current folder is used.
"""
import os
from pathlib import Path


def _find_root():
    env = os.environ.get("X2C_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for p in Path(__file__).resolve().parents:
        if (p / "pyproject.toml").is_file():
            return p
    return Path.cwd().resolve()


ROOT = _find_root()
