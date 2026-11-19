from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from agentkit.image import annotate as annotate_module
from agentkit.image import crop as crop_module
from agentkit.image import info as info_module
from agentkit.image import resize as resize_module
from agentkit.image import rotate as rotate_module
from agentkit.image import sheet as sheet_module
from agentkit.image.annotate import annotate_image
from agentkit.image.crop import crop_image
from agentkit.image.info import get_image_info
from agentkit.image.resize import resize_image
from agentkit.image.rotate import rotate_image
from agentkit.image.sheet import sheet_images

CASES: dict[str, Callable[[Path, Path], object]] = {
    "info": lambda source, output: get_image_info(source),
    "crop": lambda source, output: crop_image(
        source=source, output=output, box=(0, 0, 2, 2)
    ),
    "resize": lambda source, output: resize_image(
        source=source, output=output, scale=0.5
    ),
    "rotate": lambda source, output: rotate_image(
        source=source, output=output, angle=90
    ),
    "annotate": lambda source, output: annotate_image(
        source=source,
        output=output,
        shape="line",
        coordinates=(0, 0, 2, 1),
        color="red",
    ),
    "sheet": lambda source, output: sheet_images(sources=[source], output=output),
}
MODULES = {
    "info": info_module,
    "crop": crop_module,
    "resize": resize_module,
    "rotate": rotate_module,
    "annotate": annotate_module,
    "sheet": sheet_module,
}
SOURCE_PIXEL_CASES = ("info", "crop", "resize", "rotate", "annotate")
IDS = list(CASES)


def _output_for(tmp_path: Path, name: str, leaf: str) -> Path:
    base = tmp_path / leaf
    return base / "sheets" if name == "sheet" else base / "out.png"


def _assert_published(name: str, output: Path) -> None:
    if name == "sheet":
        assert (output / "sheet_0001.png").is_file()
    elif name != "info":
        assert output.is_file()


def _assert_not_published(name: str, output: Path) -> None:
    if name == "sheet":
        if output.exists():
            assert not list(output.glob("sheet_*.png"))
    else:
        assert not output.exists()


def _assert_no_leftovers(
    name: str,
    output: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    directory = output if name == "sheet" else output.parent
    if directory.exists():
        assert_no_leftovers(directory)


@pytest.mark.parametrize("name", IDS)
def test_source_file_size_limit_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
    padded_png: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
    name: str,
) -> None:
    limit = padded_png.stat().st_size
    monkeypatch.setattr(MODULES[name], "MAX_FILE_SIZE", limit)

    accepted = _output_for(tmp_path, name, "accepted")
    CASES[name](padded_png, accepted)
    _assert_published(name, accepted)

    monkeypatch.setattr(MODULES[name], "MAX_FILE_SIZE", limit - 1)
    rejected = _output_for(tmp_path, name, "rejected")
    message = "exceeds" if name == "info" else "between 1 byte"
    with pytest.raises(ValueError, match=message):
        CASES[name](padded_png, rejected)

    _assert_not_published(name, rejected)
    _assert_no_leftovers(name, rejected, assert_no_leftovers)


@pytest.mark.parametrize("name", SOURCE_PIXEL_CASES)
def test_source_pixel_limit_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
    name: str,
) -> None:
    monkeypatch.setattr(MODULES[name], "MAX_PIXELS", 8)

    accepted = _output_for(tmp_path, name, "accepted")
    CASES[name](rgb_png, accepted)
    _assert_published(name, accepted)

    monkeypatch.setattr(MODULES[name], "MAX_PIXELS", 7)
    rejected = _output_for(tmp_path, name, "rejected")
    with pytest.raises(ValueError, match="pixel"):
        CASES[name](rgb_png, rejected)

    _assert_not_published(name, rejected)
    _assert_no_leftovers(name, rejected, assert_no_leftovers)


def test_resize_result_pixel_limit_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    monkeypatch.setattr(resize_module, "MAX_PIXELS", 32)

    accepted = tmp_path / "accepted" / "out.png"
    resize_image(source=rgb_png, output=accepted, scale=2.0)
    assert accepted.is_file()

    monkeypatch.setattr(resize_module, "MAX_PIXELS", 31)
    rejected = tmp_path / "rejected" / "out.png"
    with pytest.raises(ValueError, match="Resulting image exceeds pixel limit"):
        resize_image(source=rgb_png, output=rejected, scale=2.0)

    assert not rejected.exists()
    if rejected.parent.exists():
        assert_no_leftovers(rejected.parent)


def test_sheet_canvas_pixel_limit_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    monkeypatch.setattr(sheet_module, "MAX_PIXELS", 1920 * 1080)

    accepted = tmp_path / "accepted" / "sheets"
    sheet_images(sources=[rgb_png], output=accepted)
    assert (accepted / "sheet_0001.png").is_file()

    monkeypatch.setattr(sheet_module, "MAX_PIXELS", 1920 * 1080 - 1)
    rejected = tmp_path / "rejected" / "sheets"
    with pytest.raises(ValueError, match="Sheet exceeds pixel limit"):
        sheet_images(sources=[rgb_png], output=rejected)

    if rejected.exists():
        assert not list(rejected.glob("sheet_*.png"))
        assert_no_leftovers(rejected)


@pytest.mark.parametrize("name", IDS)
def test_empty_source_is_rejected(
    empty_file: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
    name: str,
) -> None:
    output = _output_for(tmp_path, name, "empty")
    message = "empty" if name == "info" else "between 1 byte"

    with pytest.raises(ValueError, match=message):
        CASES[name](empty_file, output)

    _assert_not_published(name, output)
    _assert_no_leftovers(name, output, assert_no_leftovers)
