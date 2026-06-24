"""Tests for pixelrag.yaml config validation (issue #98)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "index" / "src"))
from pixelrag_index.config import ConfigValidationError, _validate, load_config


def test_valid_config_no_warnings():
    config = {
        "source": {"type": "local", "path": "./docs"},
        "embed": {"device": "auto"},
        "ingest": {"backend": "cdp"},
        "output": "./index",
    }
    warnings = _validate(config)
    assert warnings == []


def test_invalid_device_raises():
    config = {"embed": {"device": "tpu"}}
    with pytest.raises(ConfigValidationError, match="device.*tpu"):
        _validate(config)


def test_invalid_source_type_raises():
    config = {"source": {"type": "s3"}}
    with pytest.raises(ConfigValidationError, match="source.type.*s3"):
        _validate(config)


def test_invalid_backend_raises():
    config = {"ingest": {"backend": "selenium"}}
    with pytest.raises(ConfigValidationError, match="backend.*selenium"):
        _validate(config)


def test_typo_in_top_level_key_warns():
    config = {"souce": {"type": "local"}}  # typo: souce → source
    warnings = _validate(config)
    assert any("souce" in w for w in warnings)
    assert any("source" in w for w in warnings)  # suggests correction


def test_typo_in_section_key_warns():
    config = {"embed": {"devie": "auto"}}  # typo: devie → device
    warnings = _validate(config)
    assert any("devie" in w for w in warnings)
    assert any("device" in w for w in warnings)


def test_missing_source_type_warns():
    config = {"source": {"path": "./docs"}}
    warnings = _validate(config)
    assert any("source.type" in w for w in warnings)


def test_valid_values_pass():
    """All valid device/source/backend values should not raise."""
    for device in ["auto", "cpu", "cuda", "mps"]:
        _validate({"embed": {"device": device}})
    for stype in ["local", "web", "pdf", "kiwix"]:
        _validate({"source": {"type": stype}})
    for backend in ["cdp", "playwright"]:
        _validate({"ingest": {"backend": backend}})


def test_load_config_with_typo_warns(tmp_path, caplog):
    import logging

    config_path = tmp_path / "pixelrag.yaml"
    config_path.write_text("embed:\n  devie: auto\n")

    with caplog.at_level(logging.WARNING):
        load_config(str(config_path))

    assert any("devie" in r.message for r in caplog.records)
