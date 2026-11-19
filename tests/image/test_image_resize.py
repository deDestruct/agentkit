from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image

from agentkit.image import resize as resize_module
from agentkit.image.resize import resize_image


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _alpha_values(image: Image.Image) -> list[int]:
    values: list[int] = []
    for x in range(image.width):
        for y in range(image.height):
            value = image.getpixel((x, y))
            if isinstance(value, int):
                values.append(value)
    return values


@pytest.mark.parametrize(
    ("scale", "expected_size"),
    [
        (0.5, (2, 1)),
        (1.5, (6, 3)),
        (2.0, (8, 4)),
        (1.0, (4, 2)),
    ],
)
def test_scales_preserving_aspect_ratio(
    rgb_png: Path,
    tmp_path: Path,
    scale: float,
    expected_size: tuple[int, int],
) -> None:
    output = tmp_path / f"scaled_{scale}.png"

    result = resize_image(source=rgb_png, output=output, scale=scale)

    assert result["status"] == "ok"
    with Image.open(output) as image:
        assert image.size == expected_size


def test_handles_tall_images(tmp_path: Path) -> None:
    source = tmp_path / "tall.png"
    with Image.new("RGB", (2, 6), "gray") as image:
        image.save(source)
    output = tmp_path / "scaled.png"

    resize_image(source=source, output=output, scale=0.5)

    with Image.open(output) as image:
        assert image.size == (1, 3)


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

    resize_image(source=rgb_png, output=output, scale=2.0)

    with Image.open(output) as image:
        assert image.format == expected_format
        assert image.size == (8, 4)


def test_converts_opaque_palette_to_rgb(
    opaque_palette_png: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "converted.png"

    resize_image(source=opaque_palette_png, output=output, scale=2.0)

    with Image.open(output) as image:
        assert image.mode == "RGB"


def test_converts_transparent_palette_to_rgba(
    palette_png: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "converted.png"

    resize_image(source=palette_png, output=output, scale=2.0)

    with Image.open(output) as image:
        assert image.mode == "RGBA"


def test_converts_bilevel_to_grayscale(image_factory, tmp_path: Path) -> None:
    source = image_factory("bilevel.png", mode="1")
    output = tmp_path / "converted.png"

    resize_image(source=source, output=output, scale=2.0)

    with Image.open(output) as image:
        assert image.mode == "L"


def test_preserves_alpha_channel(rgba_png: Path, tmp_path: Path) -> None:
    output = tmp_path / "scaled-alpha.png"

    resize_image(source=rgba_png, output=output, scale=2.0)

    with Image.open(output) as image:
        assert image.mode == "RGBA"
        assert min(_alpha_values(image.getchannel("A"))) < 255


def test_rejects_transparency_for_jpeg(rgba_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JPEG requires"):
        resize_image(source=rgba_png, output=tmp_path / "alpha.jpg", scale=2.0)


@pytest.mark.parametrize("scale", [0, -1, -0.5])
def test_rejects_non_positive_scale(
    rgb_png: Path,
    tmp_path: Path,
    scale: float,
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        resize_image(source=rgb_png, output=tmp_path / "out.png", scale=scale)


@pytest.mark.parametrize("scale", [float("inf"), float("nan")])
def test_rejects_non_finite_scale(
    rgb_png: Path,
    tmp_path: Path,
    scale: float,
) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        resize_image(source=rgb_png, output=tmp_path / "out.png", scale=scale)


def test_rejects_non_numeric_scale(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="must be a number"):
        resize_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            scale=cast(Any, "2"),
        )


def test_rejects_boolean_scale(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="must be a number"):
        resize_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            scale=cast(Any, True),
        )


def test_rejects_huge_integer_scale(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="too large"):
        resize_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            scale=cast(Any, 10**400),
        )


def test_rejects_non_finite_resulting_dimensions(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dimensions are too large"):
        resize_image(source=rgb_png, output=tmp_path / "out.png", scale=1e308)


def test_rejects_scale_below_one_pixel(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="too small"):
        resize_image(source=rgb_png, output=tmp_path / "out.png", scale=0.1)


def test_enforces_source_pixel_limit(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(resize_module, "MAX_PIXELS", 5)

    with pytest.raises(ValueError, match="Source image exceeds pixel limit"):
        resize_image(source=rgb_png, output=tmp_path / "out.png", scale=1.0)


def test_enforces_result_pixel_limit(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(resize_module, "MAX_PIXELS", 10)

    with pytest.raises(ValueError, match="Resulting image exceeds pixel limit"):
        resize_image(source=rgb_png, output=tmp_path / "out.png", scale=2.0)


def test_enforces_file_size_limit(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(resize_module, "MAX_FILE_SIZE", 10)

    with pytest.raises(ValueError, match="between 1 byte and 100 MiB"):
        resize_image(source=rgb_png, output=tmp_path / "out.png", scale=1.0)


def test_enforces_output_file_size_limit(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(resize_module, "MAX_FILE_SIZE", 300)

    with pytest.raises(ValueError, match="Output file must be between"):
        resize_image(source=rgb_png, output=tmp_path / "out.jpg", scale=2.0)


@pytest.mark.parametrize("quality", [0, 96])
def test_rejects_quality_out_of_range(
    rgb_png: Path,
    tmp_path: Path,
    quality: int,
) -> None:
    with pytest.raises(ValueError, match="between 1 and 95"):
        resize_image(
            source=rgb_png,
            output=tmp_path / "out.jpg",
            scale=1.0,
            quality=quality,
        )


def test_rejects_non_integer_quality(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="integer"):
        resize_image(
            source=rgb_png,
            output=tmp_path / "out.jpg",
            scale=1.0,
            quality=True,  # type: ignore[arg-type]
        )


def test_rejects_quality_for_png(rgb_png: Path, tmp_path: Path) -> None:
    """Characterizes current behaviour; the contract is undecided (see tests/image/README.md)."""
    with pytest.raises(ValueError, match="only for JPEG"):
        resize_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            scale=1.0,
            quality=80,
        )


def test_accepts_jpeg_quality(rgb_png: Path, tmp_path: Path) -> None:
    output = tmp_path / "quality.jpg"

    resize_image(source=rgb_png, output=output, scale=2.0, quality=95)

    with Image.open(output) as image:
        assert image.format == "JPEG"


def test_rejects_same_source_and_output(rgb_png: Path) -> None:
    with pytest.raises(ValueError, match="must be different"):
        resize_image(source=rgb_png, output=rgb_png, scale=2.0)


def test_rejects_existing_output(rgb_png: Path, tmp_path: Path) -> None:
    output = tmp_path / "exists.png"
    output.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError, match="already exists"):
        resize_image(source=rgb_png, output=output, scale=2.0)

    assert output.read_bytes() == b"sentinel"


def test_rejects_unsupported_extension(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported output format"):
        resize_image(source=rgb_png, output=tmp_path / "out.tga", scale=2.0)


def test_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resize_image(source=tmp_path / "absent.png", output=tmp_path / "out.png", scale=2.0)


def test_rejects_directory_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Not a regular file"):
        resize_image(source=tmp_path, output=tmp_path / "out.png", scale=2.0)


def test_rejects_invalid_image(garbage_file: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid or unsafe image"):
        resize_image(source=garbage_file, output=tmp_path / "out.png", scale=2.0)


def test_rejects_multi_frame_image(animated_gif: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Multi-frame"):
        resize_image(source=animated_gif, output=tmp_path / "out.png", scale=2.0)


def test_creates_nested_output_directory(rgb_png: Path, tmp_path: Path) -> None:
    output = tmp_path / "nested" / "deep" / "out.png"

    resize_image(source=rgb_png, output=output, scale=2.0)

    assert output.is_file()


def test_does_not_modify_source_and_leaves_no_artifacts(
    rgb_png: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable,
) -> None:
    before = _digest(rgb_png)

    resize_image(source=rgb_png, output=tmp_path / "out.png", scale=2.0)

    assert _digest(rgb_png) == before
    assert_no_leftovers(tmp_path)
