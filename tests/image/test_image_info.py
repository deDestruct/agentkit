from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agentkit.image import info
from agentkit.image.info import get_image_info


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_returns_metadata(rgb_png: Path) -> None:
    result = get_image_info(rgb_png)

    assert result["width"] == 4
    assert result["height"] == 2
    assert result["format"] == "PNG"
    assert result["mode"] == "RGB"
    assert result["size_bytes"] == rgb_png.stat().st_size
    assert Path(str(result["path"])).samefile(rgb_png)


def test_reports_jpeg_format(rgb_jpeg: Path) -> None:
    result = get_image_info(rgb_jpeg)

    assert result["format"] == "JPEG"
    assert result["width"] == 4
    assert result["height"] == 2


def test_reports_palette_mode(palette_png: Path) -> None:
    result = get_image_info(palette_png)

    assert result["mode"] == "P"
    assert result["format"] == "PNG"


def test_reports_bilevel_mode(image_factory) -> None:
    result = get_image_info(image_factory("bilevel.png", mode="1"))

    assert result["mode"] == "1"


def test_reads_multi_frame_gif(animated_gif: Path) -> None:
    result = get_image_info(animated_gif)

    assert result["format"] == "GIF"
    assert result["width"] == 4
    assert result["height"] == 2


def test_does_not_modify_source(rgb_png: Path) -> None:
    before = _digest(rgb_png)

    get_image_info(rgb_png)

    assert _digest(rgb_png) == before


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        get_image_info(tmp_path / "absent.png")


def test_directory_instead_of_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Not a regular file"):
        get_image_info(tmp_path)


def test_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.png"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="empty"):
        get_image_info(path)


def test_file_size_limit(monkeypatch: pytest.MonkeyPatch, rgb_png: Path) -> None:
    monkeypatch.setattr(info, "MAX_FILE_SIZE", 10)

    with pytest.raises(ValueError, match="exceeds"):
        get_image_info(rgb_png)


def test_pixel_limit(monkeypatch: pytest.MonkeyPatch, rgb_png: Path) -> None:
    monkeypatch.setattr(info, "MAX_PIXELS", 3)

    with pytest.raises(ValueError, match="exceeds"):
        get_image_info(rgb_png)


def test_invalid_image(garbage_file: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported, invalid or unsafe image"):
        get_image_info(garbage_file)
