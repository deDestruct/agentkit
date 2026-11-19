from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image

from agentkit.image import rotate as rotate_module
from agentkit.image.rotate import ANGLES, rotate_image

MAPPING = {
    90: lambda x, y, width, height: (height - 1 - y, x),
    180: lambda x, y, width, height: (width - 1 - x, height - 1 - y),
    270: lambda x, y, width, height: (y, width - 1 - x),
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exposes_supported_angles() -> None:
    assert ANGLES == (90, 180, 270)


@pytest.mark.parametrize("angle", [90, 180, 270])
def test_rotates_pixels_clockwise(
    rgb_png: Path,
    tmp_path: Path,
    read_pixels: Callable,
    angle: int,
) -> None:
    output = tmp_path / f"rotated_{angle}.png"
    source_pixels = read_pixels(rgb_png)
    width = len(source_pixels[0])
    height = len(source_pixels)

    result = rotate_image(source=rgb_png, output=output, angle=angle)

    assert result["status"] == "ok"
    assert Path(result["output"]).samefile(output)
    rotated = read_pixels(output)
    for x in range(width):
        for y in range(height):
            new_x, new_y = MAPPING[angle](x, y, width, height)
            assert rotated[new_y][new_x] == source_pixels[y][x]


@pytest.mark.parametrize(
    ("angle", "expected_size"),
    [(90, (3, 5)), (180, (5, 3)), (270, (3, 5))],
)
def test_swaps_dimensions_for_non_square_images(
    image_factory,
    tmp_path: Path,
    angle: int,
    expected_size: tuple[int, int],
) -> None:
    source = image_factory("wide.png", size=(5, 3))
    output = tmp_path / f"rotated_{angle}.png"

    rotate_image(source=source, output=output, angle=angle)

    with Image.open(output) as image:
        assert image.size == expected_size


@pytest.mark.parametrize(
    ("name", "expected_format"),
    [
        ("out.png", "PNG"),
        ("out.jpg", "JPEG"),
        ("out.webp", "WEBP"),
        ("out.bmp", "BMP"),
        ("out.tiff", "TIFF"),
    ],
)
def test_writes_supported_formats(
    rgb_png: Path,
    tmp_path: Path,
    name: str,
    expected_format: str,
) -> None:
    output = tmp_path / name

    rotate_image(source=rgb_png, output=output, angle=90)

    with Image.open(output) as image:
        assert image.format == expected_format
        assert image.size == (2, 4)


def test_preserves_alpha_for_png(
    rgba_png: Path,
    tmp_path: Path,
    read_pixels: Callable,
) -> None:
    output = tmp_path / "rotated-alpha.png"
    source_pixels = read_pixels(rgba_png)

    rotate_image(source=rgba_png, output=output, angle=90)

    rotated = read_pixels(output)
    width = len(source_pixels[0])
    height = len(source_pixels)
    for x in range(width):
        for y in range(height):
            assert rotated[x][height - 1 - y] == source_pixels[y][x]


def test_rejects_transparency_for_jpeg(rgba_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JPEG requires"):
        rotate_image(source=rgba_png, output=tmp_path / "alpha.jpg", angle=90)


@pytest.mark.parametrize("angle", [45, 0, -90, 360])
def test_rejects_unsupported_angles(
    rgb_png: Path,
    tmp_path: Path,
    angle: int,
) -> None:
    with pytest.raises(ValueError, match="90, 180, 270"):
        rotate_image(source=rgb_png, output=tmp_path / "out.png", angle=angle)


@pytest.mark.parametrize("angle", [90.0, "90", True, None])
def test_rejects_non_integer_angle(
    rgb_png: Path,
    tmp_path: Path,
    angle: Any,
) -> None:
    with pytest.raises(TypeError, match="Angle must be an integer"):
        rotate_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            angle=angle,
        )


@pytest.mark.parametrize("quality", [1, 50, 95])
def test_accepts_jpeg_quality(rgb_png: Path, tmp_path: Path, quality: int) -> None:
    output = tmp_path / f"quality_{quality}.jpg"

    rotate_image(source=rgb_png, output=output, angle=180, quality=quality)

    with Image.open(output) as image:
        assert image.format == "JPEG"
        assert image.size == (4, 2)


@pytest.mark.parametrize("quality", [0, 96])
def test_rejects_quality_out_of_range(
    rgb_png: Path,
    tmp_path: Path,
    quality: int,
) -> None:
    with pytest.raises(ValueError, match="between 1 and 95"):
        rotate_image(
            source=rgb_png,
            output=tmp_path / "out.jpg",
            angle=90,
            quality=quality,
        )


def test_rejects_non_integer_quality(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="integer"):
        rotate_image(
            source=rgb_png,
            output=tmp_path / "out.jpg",
            angle=90,
            quality=cast(Any, True),
        )


def test_rejects_quality_for_png(rgb_png: Path, tmp_path: Path) -> None:
    """Characterizes current behaviour; the contract is undecided (see tests/image/README.md)."""
    with pytest.raises(ValueError, match="only for JPEG"):
        rotate_image(source=rgb_png, output=tmp_path / "out.png", angle=90, quality=80)


def test_rejects_same_source_and_output(rgb_png: Path) -> None:
    with pytest.raises(ValueError, match="must be different"):
        rotate_image(source=rgb_png, output=rgb_png, angle=90)


def test_rejects_existing_output(rgb_png: Path, tmp_path: Path) -> None:
    output = tmp_path / "exists.png"
    output.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError, match="already exists"):
        rotate_image(source=rgb_png, output=output, angle=90)

    assert output.read_bytes() == b"sentinel"


def test_rejects_unsupported_extension(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported output format"):
        rotate_image(source=rgb_png, output=tmp_path / "out.tga", angle=90)


def test_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        rotate_image(source=tmp_path / "absent.png", output=tmp_path / "out.png", angle=90)


def test_rejects_directory_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Not a regular file"):
        rotate_image(source=tmp_path, output=tmp_path / "out.png", angle=90)


def test_rejects_invalid_image(garbage_file: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid or unsafe image"):
        rotate_image(source=garbage_file, output=tmp_path / "out.png", angle=90)


def test_rejects_multi_frame_image(animated_gif: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Multi-frame"):
        rotate_image(source=animated_gif, output=tmp_path / "out.png", angle=90)


def test_enforces_file_size_limit(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(rotate_module, "MAX_FILE_SIZE", 10)

    with pytest.raises(ValueError, match="between 1 byte and 100 MiB"):
        rotate_image(source=rgb_png, output=tmp_path / "out.png", angle=90)


def test_enforces_pixel_limit(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(rotate_module, "MAX_PIXELS", 3)

    with pytest.raises(ValueError, match="pixel limit"):
        rotate_image(source=rgb_png, output=tmp_path / "out.png", angle=90)


def test_enforces_output_file_size_limit(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(rotate_module, "MAX_FILE_SIZE", 300)

    with pytest.raises(ValueError, match="Output file must be between"):
        rotate_image(source=rgb_png, output=tmp_path / "out.jpg", angle=90)


def test_creates_nested_output_directory(rgb_png: Path, tmp_path: Path) -> None:
    output = tmp_path / "nested" / "deep" / "out.png"

    rotate_image(source=rgb_png, output=output, angle=90)

    assert output.is_file()


def test_does_not_modify_source_and_leaves_no_artifacts(
    rgb_png: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable,
) -> None:
    before = _digest(rgb_png)

    rotate_image(source=rgb_png, output=tmp_path / "out.png", angle=270)

    assert _digest(rgb_png) == before
    assert_no_leftovers(tmp_path)
