import datetime
import os

import pytest

from openrouter_image_mcp.paths import (
    default_image_path,
    free_target,
    is_taken,
    media_type_for,
    numbered,
    resolve_inside,
    slugify,
    with_extension,
)


def test_relative_path_resolves_inside_root(tmp_path):
    assert resolve_inside(str(tmp_path), "a/b.png") == os.path.realpath(tmp_path / "a" / "b.png")


def test_absolute_path_inside_root_is_allowed(tmp_path):
    inside = str(tmp_path / "x.png")
    assert resolve_inside(str(tmp_path), inside) == os.path.realpath(inside)


@pytest.mark.parametrize("path", ["../escape.png", "a/../../escape.png"])
def test_parent_traversal_is_refused(tmp_path, path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError):
        resolve_inside(str(root), path)


def test_absolute_path_outside_root_is_refused(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError):
        resolve_inside(str(root), str(tmp_path / "other" / "x.png"))


def test_sibling_with_common_prefix_is_refused(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ValueError):
        resolve_inside(str(root), str(tmp_path / "root-other" / "x.png"))


def test_default_path_uses_timestamp_and_slug():
    now = datetime.datetime(2026, 11, 2, 5, 1, 2)
    path = default_image_path("generated", "A Satellite, icon!", "png", now)
    assert path == os.path.join("generated", "20261102-050102-a-satellite-icon.png")


def test_slug_of_symbols_only_falls_back():
    assert slugify("!!!") == "image"


def test_numbering_only_for_several_images():
    assert numbered("a/b.png", 0, 1) == "a/b.png"
    assert numbered("a/b.png", 1, 3) == "a/b-2.png"


def test_reference_media_types():
    assert media_type_for("x.JPG") == "image/jpeg"
    with pytest.raises(ValueError):
        media_type_for("x.gif")


def test_extension_is_added_or_must_match_the_format():
    assert with_extension("out/a", "png") == "out/a.png"
    assert with_extension("out/a.JPG", "jpeg") == "out/a.JPG"
    with pytest.raises(ValueError):
        with_extension("out/a.png", "jpeg")
    with pytest.raises(ValueError):
        with_extension("out/a.json", "json")


def test_a_sidecar_alone_makes_a_name_taken(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    assert is_taken(str(tmp_path / "a.png"))
    assert not is_taken(str(tmp_path / "b.png"))


def test_free_target_skips_names_used_by_any_file_of_the_batch(tmp_path):
    target = str(tmp_path / "a.png")
    assert free_target(target, 1) == target
    (tmp_path / "a-2.png").write_bytes(b"x")
    assert free_target(target, 1) == target
    assert free_target(target, 2) == str(tmp_path / "a-v2.png")
