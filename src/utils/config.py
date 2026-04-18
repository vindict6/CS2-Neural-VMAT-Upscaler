"""
Application configuration management.
Loads/saves settings from a JSON config file.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from ..core.upscaler import UpscaleSettings

logger = logging.getLogger("TextureForge.Config")

DEFAULT_CONFIG_DIR = Path.home() / ".textureforge"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.json"

DEFAULTS: Dict[str, Any] = {
    "window": {
        "width": 1600,
        "height": 1000,
        "maximized": False,
    },
    "paths": {
        "last_open_dir": "",
        "last_save_dir": "",
        "output_dir": "",
        "models_dir": "models",
    },
    "upscale": asdict(UpscaleSettings()),
    "ui": {
        "theme": "dark",
        "preview_mode": "slider",
        "show_info_panel": True,
        "show_batch_panel": True,
    },
}


class Config:
    """JSON-backed application configuration."""

    def __init__(self, path: str = None):
        self._path = Path(path) if path else DEFAULT_CONFIG_FILE
        self._data: Dict[str, Any] = {}
        self.load()

    def load(self):
        """Load config from disk, merging with defaults."""
        self._data = _deep_merge(DEFAULTS, {})
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self._data = _deep_merge(DEFAULTS, saved)
                logger.info(f"Config loaded from {self._path}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")

    def save(self):
        """Persist config to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, default=str)
        logger.debug(f"Config saved to {self._path}")

    def get(self, *keys, default=None):
        """Get a nested value: config.get('window', 'width')."""
        d = self._data
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, default)
            else:
                return default
        return d

    def set(self, *keys_and_value):
        """Set a nested value: config.set('window', 'width', 1920)."""
        if len(keys_and_value) < 2:
            return
        keys = keys_and_value[:-1]
        value = keys_and_value[-1]
        d = self._data
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    @property
    def data(self) -> dict:
        return self._data


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
