"""Helpers for loading Skore Hub credentials from the repo's .skore file."""

from __future__ import annotations

import json
import os
from pathlib import Path


def load_skore_credentials() -> dict:
    """Load hub credentials from the .skore file at the repo root.

    Returns a dict with keys: hub_url, workspace, workspace_id, api_key.
    Also sets the SKORE_HUB_API_KEY environment variable so that skore's
    login plugin can pick it up without opening a browser.
    """
    skore_path = Path(__file__).resolve().parents[2] / ".skore"
    if not skore_path.is_file():
        raise FileNotFoundError(
            f".skore not found at {skore_path}. Run `python scripts/skore-agent`."
        )
    cfg = json.loads(skore_path.read_text())
    os.environ["SKORE_HUB_API_KEY"] = cfg["api_key"]
    return cfg
