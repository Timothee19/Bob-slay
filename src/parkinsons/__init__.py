"""Package root — Parkinson's UPDRS regression workspace.

Exposes `PROJECT_ROOT`: absolute path to the project root, derived from
this file's location. Import this in any module that needs project-relative
paths (data files, configs) instead of hard-coding CWD-relative strings.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
