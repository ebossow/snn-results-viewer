"""Small UI helper utilities for the Streamlit app."""
from __future__ import annotations

import yaml
from typing import Any


def config_to_yaml_text(config: dict | None) -> str:
    """Return a YAML string for display from a config dict.

    Safe to call with None.
    """
    if not config:
        return "# no config available"
    try:
        return yaml.safe_dump(config, sort_keys=False)
    except Exception:
        return str(config)
