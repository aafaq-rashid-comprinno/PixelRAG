"""Path-traversal hardening tests for eval/lib/pixel_query.py (issues #78, #79).

Tests the sanitization logic directly — verifying that pathlib.Path(x).name
strips traversal sequences from example_id before it reaches os.path.join.
Does NOT require a TTF font or actual rendering.
"""

import os
import pathlib

import pytest


@pytest.mark.parametrize(
    "example_id,expected_stem",
    [
        ("normal_id", "normal_id"),
        ("../../etc/passwd", "passwd"),
        ("../evil", "evil"),
        ("foo/bar/baz", "baz"),
        ("/absolute/path/id", "id"),
        ("..%2F..%2Fetc/passwd", "passwd"),  # URL-encoded separators still have /
        ("....//....//etc/shadow", "shadow"),
    ],
)
def test_safe_id_strips_traversal(example_id, expected_stem):
    """pathlib.Path(x).name must strip all directory components."""
    safe_id = pathlib.Path(example_id).name
    assert safe_id == expected_stem


@pytest.mark.parametrize(
    "example_id",
    [
        "../../etc/passwd",
        "../evil",
        "foo/bar/baz",
        "/absolute/path/id",
        "..\\..\\windows\\system32",
    ],
)
def test_constructed_path_stays_in_output_dir(tmp_path, example_id):
    """os.path.join with sanitized id must never escape output_dir."""
    output_dir = str(tmp_path)
    safe_id = pathlib.Path(example_id).name
    out_path = os.path.join(output_dir, f"{safe_id}_query.png")

    # Resolve and check containment
    resolved = pathlib.Path(out_path).resolve()
    assert str(resolved).startswith(str(tmp_path.resolve()))


def test_traversal_does_not_reach_parent(tmp_path):
    """Even deeply nested traversal must not escape."""
    evil_ids = [
        "../" * 10 + "etc/passwd",
        "..\\..\\..\\windows\\system32\\config",
    ]
    for example_id in evil_ids:
        safe_id = pathlib.Path(example_id).name
        out_path = os.path.join(str(tmp_path), f"{safe_id}_query.png")
        resolved = pathlib.Path(out_path).resolve()
        assert str(resolved).startswith(str(tmp_path.resolve())), (
            f"{example_id!r} escaped to {resolved}"
        )
