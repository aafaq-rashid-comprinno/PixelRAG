"""Parse pixelrag.yaml with parameter forwarding and validation."""

import logging
import os
from difflib import get_close_matches
from pathlib import Path

import yaml

from .sources import SOURCES

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "ingest": {"backend": "cdp", "quality": 85, "tile_height": 8192},
    "embed": {"model": "Qwen/Qwen3-VL-Embedding-2B", "device": "cuda"},
    "output": "./index",
}

# Valid values for validated fields
_VALID = {
    "source.type": list(SOURCES.keys()),
    "embed.device": ["auto", "cpu", "cuda", "mps"],
    "ingest.backend": ["cdp", "playwright"],
}

# Known keys per section (for typo detection)
_KNOWN_KEYS = {
    "source": {"type", "path", "urls_file", "zim_path", "preset", "start_url", "max_pages", "max_depth", "stay_on_domain", "exclude_patterns"},
    "embed": {"model", "device", "gpu_ids", "backend", "instruction", "batch_size"},
    "ingest": {"backend", "quality", "tile_height", "wait_network_idle", "viewport_width", "workers"},
}

_KNOWN_TOP_LEVEL = {"source", "embed", "ingest", "output"}


class ConfigValidationError(ValueError):
    """Raised when pixelrag.yaml has invalid values."""
    pass


def _validate(config: dict) -> list[str]:
    """Validate config and return list of warnings. Raises on hard errors."""
    warnings = []

    # Check unknown top-level keys
    for key in config:
        if key not in _KNOWN_TOP_LEVEL:
            suggestion = get_close_matches(key, _KNOWN_TOP_LEVEL, n=1, cutoff=0.6)
            hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
            warnings.append(f"Unknown top-level key '{key}'.{hint}")

    # Check unknown keys within sections
    for section, known in _KNOWN_KEYS.items():
        section_data = config.get(section, {})
        if not isinstance(section_data, dict):
            continue
        for key in section_data:
            if key not in known:
                suggestion = get_close_matches(key, known, n=1, cutoff=0.6)
                hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
                warnings.append(f"Unknown key '{key}' in '{section}' config.{hint}")

    # Validate constrained values
    for field_path, valid_values in _VALID.items():
        section, key = field_path.split(".")
        value = config.get(section, {}).get(key) if isinstance(config.get(section), dict) else None
        if value is not None and value not in valid_values:
            raise ConfigValidationError(
                f"Invalid {field_path}='{value}'. Must be one of: {valid_values}"
            )

    # Source must have type
    source = config.get("source", {})
    if isinstance(source, dict) and "type" not in source:
        warnings.append("'source.type' not specified, defaulting to 'local'.")

    return warnings


def load_config(path=None):
    if path is None:
        for c in [Path("pixelrag.yaml"), Path("pixelrag.yml")]:
            if c.exists():
                path = str(c)
                break
    if path and os.path.exists(path):
        with open(path) as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # Validate
    warnings = _validate(config)
    for w in warnings:
        logger.warning("pixelrag.yaml: %s", w)

    return {**DEFAULT_CONFIG, **config}


def make_source(config):
    source_config = dict(config.get("source", {}))
    source_type = source_config.pop("type", "local")
    # Expand ~ in any string values that look like paths (but not URLs)
    for k, v in source_config.items():
        if isinstance(v, str) and ("/" in v or "~" in v) and "://" not in v:
            source_config[k] = str(Path(v).expanduser())
    return SOURCES[source_type](**source_config)
