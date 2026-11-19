from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image

from agentkit.image import crop as crop_module
from agentkit.image.crop import crop_image


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_region(source: Path, box: tuple[int, int, int, int]) -> list[list]:
    with Image.open(source) as image:
        return [
            [image.getpixel((x, y)) for x in range(box[0], box[2])]
            for y in range(box[1], box[3])
        ]


def test_crops_expected_region(
    rgb_png: Path,
    tmp_path: Path,
    read_pixels: Callable,
) -> None:
    output = tmp_path / "cropped.png"

    result = crop_image(source=rgb_png, output=output, box=(1, 0, 3, 2))

    assert result["status"] == "ok"
    assert Path(result["output"]).samefile(output)
    assert read_pixels(output) == _expected_region(rgb_png, (1, 0, 3, 2))


def test_accepts_list_box(
    rgb_png: Path,
    tmp_path: Path,
    read_pixels: Callable,
) -> None:
    output = tmp_path / "list.png"

    box: Any = [0, 0, 2, 2]

    crop_image(source=rgb_png, output=output, box=box)

    assert read_pixels(output) == _expected_region(rgb_png, (0, 0, 2, 2))


def test_crops_full_image(rgb_png: Path, tmp_path: Path, read_pixels: Callable) -> None:
    output = tmp_path / "full.png"

    crop_image(source=rgb_png, output=output, box=(0, 0, 4, 2))

    assert read_pixels(output) == read_pixels(rgb_png)


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

    crop_image(source=rgb_png, output=output, box=(0, 0, 2, 2))

    with Image.open(output) as image:
        assert image.format == expected_format
        assert image.size == (2, 2)


def test_preserves_alpha_for_png(
    rgba_png: Path,
    tmp_path: Path,
    read_pixels: Callable,
) -> None:
    output = tmp_path / "cropped-alpha.png"

    crop_image(source=rgba_png, output=output, box=(0, 0, 4, 2))

    assert read_pixels(output) == read_pixels(rgba_png)


def test_rejects_transparency_for_jpeg(rgba_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JPEG requires"):
        crop_image(source=rgba_png, output=tmp_path / "alpha.jpg", box=(0, 0, 2, 2))


@pytest.mark.parametrize("quality", [1, 50, 95])
def test_accepts_jpeg_quality(
    rgb_jpeg: Path,
    tmp_path: Path,
    quality: int,
) -> None:
    output = tmp_path / f"quality_{quality}.jpg"

    crop_image(source=rgb_jpeg, output=output, box=(0, 0, 2, 2), quality=quality)

    with Image.open(output) as image:
        assert image.format == "JPEG"
        assert image.size == (2, 2)


@pytest.mark.parametrize("quality", [0, 96, -1])
def test_rejects_quality_out_of_range(
    rgb_png: Path,
    tmp_path: Path,
    quality: int,
) -> None:
    with pytest.raises(ValueError, match="between 1 and 95"):
        crop_image(
            source=rgb_png,
            output=tmp_path / "quality.jpg",
            box=(0, 0, 2, 2),
            quality=quality,
        )


def test_rejects_non_integer_quality(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="integer"):
        crop_image(
            source=rgb_png,
            output=tmp_path / "quality.jpg",
            box=(0, 0, 2, 2),
            quality=cast(Any, True),
        )


def test_rejects_quality_for_png(rgb_png: Path, tmp_path: Path) -> None:
    """Characterizes current behaviour; the contract is undecided (see tests/image/README.md)."""
    with pytest.raises(ValueError, match="only for JPEG"):
        crop_image(
            source=rgb_png,
            output=tmp_path / "quality.png",
            box=(0, 0, 2, 2),
            quality=80,
        )


def test_rejects_inverted_rectangle(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid crop rectangle"):
        crop_image(source=rgb_png, output=tmp_path / "out.png", box=(2, 0, 1, 2))


def test_rejects_degenerate_rectangle(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid crop rectangle"):
        crop_image(source=rgb_png, output=tmp_path / "out.png", box=(1, 1, 1, 2))


def test_rejects_negative_coordinates(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="negative"):
        crop_image(source=rgb_png, output=tmp_path / "out.png", box=(-1, 0, 2, 2))


def test_rejects_out_of_bounds(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exceeds image bounds"):
        crop_image(source=rgb_png, output=tmp_path / "out.png", box=(0, 0, 5, 2))


@pytest.mark.parametrize(
    "box",
    [
        (0, 0, 2),
        (0, 0, 2, 2.0),
        (0, 0, True, 2),
        "0,0,2,2",
    ],
)
def test_rejects_malformed_box(rgb_png: Path, tmp_path: Path, box: Any) -> None:
    with pytest.raises(TypeError, match="exactly four integers"):
        crop_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            box=box,
        )


def test_rejects_same_source_and_output(rgb_png: Path) -> None:
    with pytest.raises(ValueError, match="must be different"):
        crop_image(source=rgb_png, output=rgb_png, box=(0, 0, 2, 2))


def test_rejects_existing_output(
    rgb_png: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable,
) -> None:
    output = tmp_path / "exists.png"
    output.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError, match="already exists"):
        crop_image(source=rgb_png, output=output, box=(0, 0, 2, 2))

    assert output.read_bytes() == b"sentinel"
    assert_no_leftovers(tmp_path)


def test_rejects_unsupported_extension(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported output format"):
        crop_image(source=rgb_png, output=tmp_path / "out.tga", box=(0, 0, 2, 2))


def test_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        crop_image(
            source=tmp_path / "absent.png",
            output=tmp_path / "out.png",
            box=(0, 0, 2, 2),
        )


def test_rejects_directory_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Not a regular file"):
        crop_image(source=tmp_path, output=tmp_path / "out.png", box=(0, 0, 2, 2))


def test_rejects_invalid_image(garbage_file: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid or unsafe image"):
        crop_image(source=garbage_file, output=tmp_path / "out.png", box=(0, 0, 2, 2))


def test_rejects_multi_frame_image(animated_gif: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Multi-frame"):
        crop_image(source=animated_gif, output=tmp_path / "out.png", box=(0, 0, 2, 2))


def test_enforces_file_size_limit(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(crop_module, "MAX_FILE_SIZE", 10)

    with pytest.raises(ValueError, match="between 1 byte and 100 MiB"):
        crop_image(source=rgb_png, output=tmp_path / "out.png", box=(0, 0, 2, 2))


def test_enforces_pixel_limit(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(crop_module, "MAX_PIXELS", 3)

    with pytest.raises(ValueError, match="pixel limit"):
        crop_image(source=rgb_png, output=tmp_path / "out.png", box=(0, 0, 2, 2))


def test_enforces_output_file_size_limit(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable,
) -> None:
    measured = tmp_path / "measured.jpg"
    crop_image(source=rgb_png, output=measured, box=(0, 0, 4, 2))
    encoded_size = measured.stat().st_size
    measured.unlink()

    monkeypatch.setattr(crop_module, "MAX_FILE_SIZE", encoded_size)
    accepted = tmp_path / "accepted.jpg"
    crop_image(source=rgb_png, output=accepted, box=(0, 0, 4, 2))
    assert accepted.stat().st_size == encoded_size

    monkeypatch.setattr(crop_module, "MAX_FILE_SIZE", encoded_size - 1)
    rejected = tmp_path / "rejected.jpg"
    with pytest.raises(ValueError, match="Output file must be between"):
        crop_image(source=rgb_png, output=rejected, box=(0, 0, 4, 2))

    assert not rejected.exists()
    assert_no_leftovers(tmp_path)


def test_creates_nested_output_directory(
    rgb_png: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable,
) -> None:
    output = tmp_path / "nested" / "deep" / "out.png"

    crop_image(source=rgb_png, output=output, box=(0, 0, 2, 2))

    assert output.is_file()
    assert_no_leftovers(tmp_path)


def test_accepts_uppercase_extension(rgb_png: Path, tmp_path: Path) -> None:
    output = tmp_path / "OUT.PNG"

    crop_image(source=rgb_png, output=output, box=(0, 0, 2, 2))

    with Image.open(output) as image:
        assert image.format == "PNG"


def test_does_not_modify_source_and_leaves_no_artifacts(
    rgb_png: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable,
) -> None:
    before = _digest(rgb_png)

    crop_image(source=rgb_png, output=tmp_path / "out.png", box=(0, 0, 2, 2))

    assert _digest(rgb_png) == before
    assert_no_leftovers(tmp_path)
