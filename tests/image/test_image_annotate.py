from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image, ImageColor

from agentkit.image import annotate as annotate_module
from agentkit.image.annotate import PALETTE, annotate_image

RED = (255, 0, 0)
WHITE = (255, 255, 255)


def _solid(tmp_path: Path, name: str, size: tuple[int, int], color: str = "white") -> Path:
    path = tmp_path / name
    with Image.new("RGB", size, color) as image:
        image.save(path)
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_draws_rectangle_outline(tmp_path: Path) -> None:
    source = _solid(tmp_path, "canvas.png", (20, 20))
    output = tmp_path / "annotated.png"

    result = annotate_image(
        source=source,
        output=output,
        shape="rectangle",
        coordinates=(2, 2, 12, 12),
        color="red",
    )

    assert result["status"] == "ok"
    with Image.open(output) as image:
        assert image.getpixel((2, 2)) == RED
        assert image.getpixel((11, 11)) == RED
        assert image.getpixel((6, 6)) == WHITE
        assert image.getpixel((1, 1)) == WHITE


def test_rectangle_bounds_are_exclusive(tmp_path: Path) -> None:
    source = _solid(tmp_path, "canvas.png", (20, 20))
    output = tmp_path / "annotated.png"

    annotate_image(
        source=source,
        output=output,
        shape="rectangle",
        coordinates=(2, 2, 12, 12),
        color="red",
    )

    with Image.open(output) as image:
        assert image.getpixel((12, 12)) == WHITE


def test_clamps_stroke_width_for_small_rectangle(tmp_path: Path) -> None:
    source = _solid(tmp_path, "canvas.png", (8, 8))
    output = tmp_path / "annotated.png"

    annotate_image(
        source=source,
        output=output,
        shape="rectangle",
        coordinates=(2, 2, 3, 3),
        color="red",
    )

    with Image.open(output) as image:
        assert image.getpixel((2, 2)) == RED
        assert image.getpixel((3, 3)) == WHITE


def test_draws_line(tmp_path: Path) -> None:
    source = _solid(tmp_path, "canvas.png", (10, 10))
    output = tmp_path / "annotated.png"

    annotate_image(
        source=source,
        output=output,
        shape="line",
        coordinates=(0, 0, 4, 4),
        color="red",
    )

    with Image.open(output) as image:
        assert image.getpixel((0, 0)) == RED
        assert image.getpixel((4, 4)) == RED
        assert image.getpixel((9, 9)) == WHITE


def test_supports_every_palette_color(tmp_path: Path) -> None:
    source = _solid(tmp_path, "canvas.png", (20, 20))

    for color in PALETTE:
        output = tmp_path / f"{color}.png"
        annotate_image(
            source=source,
            output=output,
            shape="line",
            coordinates=(0, 0, 9, 9),
            color=color,
        )
        with Image.open(output) as image:
            assert image.getpixel((0, 0)) == ImageColor.getrgb(PALETTE[color])


def test_converts_palette_image(palette_png: Path, tmp_path: Path) -> None:
    output = tmp_path / "annotated.png"

    annotate_image(
        source=palette_png,
        output=output,
        shape="rectangle",
        coordinates=(0, 0, 4, 2),
        color="red",
    )

    with Image.open(output) as image:
        assert image.mode in ("RGB", "RGBA")


def test_converts_bilevel_image(image_factory, tmp_path: Path) -> None:
    source = image_factory("bilevel.png", mode="1")
    output = tmp_path / "annotated.png"

    annotate_image(
        source=source,
        output=output,
        shape="line",
        coordinates=(0, 0, 3, 1),
        color="red",
    )

    with Image.open(output) as image:
        assert image.mode == "RGB"


def test_converts_grayscale_image(tmp_path: Path) -> None:
    source = tmp_path / "gray.png"
    with Image.new("L", (8, 8), 128) as image:
        image.save(source)
    output = tmp_path / "annotated.png"

    annotate_image(
        source=source,
        output=output,
        shape="line",
        coordinates=(0, 0, 3, 3),
        color="red",
    )

    with Image.open(output) as image:
        assert image.mode == "RGB"


def test_preserves_transparency(rgba_png: Path, tmp_path: Path) -> None:
    output = tmp_path / "annotated.png"

    annotate_image(
        source=rgba_png,
        output=output,
        shape="rectangle",
        coordinates=(0, 0, 2, 2),
        color="red",
    )

    with Image.open(rgba_png) as original, Image.open(output) as annotated:
        assert annotated.mode == "RGBA"
        original_alpha = original.getchannel("A")
        annotated_alpha = annotated.getchannel("A")
        assert annotated_alpha.getpixel((3, 0)) == original_alpha.getpixel((3, 0))
        assert annotated_alpha.getpixel((0, 0)) == 255


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

    annotate_image(
        source=rgb_png,
        output=output,
        shape="line",
        coordinates=(0, 0, 3, 1),
        color="green",
    )

    with Image.open(output) as image:
        assert image.format == expected_format


@pytest.mark.parametrize("quality", [1, 95])
def test_accepts_jpeg_quality(rgb_png: Path, tmp_path: Path, quality: int) -> None:
    output = tmp_path / f"quality_{quality}.jpg"

    annotate_image(
        source=rgb_png,
        output=output,
        shape="line",
        coordinates=(0, 0, 3, 1),
        color="red",
        quality=quality,
    )

    assert output.is_file()


@pytest.mark.parametrize("quality", [0, 96])
def test_rejects_quality_out_of_range(
    rgb_png: Path,
    tmp_path: Path,
    quality: int,
) -> None:
    with pytest.raises(ValueError, match="between 1 and 95"):
        annotate_image(
            source=rgb_png,
            output=tmp_path / "out.jpg",
            shape="line",
            coordinates=(0, 0, 3, 1),
            color="red",
            quality=quality,
        )


def test_rejects_non_integer_quality(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="integer"):
        annotate_image(
            source=rgb_png,
            output=tmp_path / "out.jpg",
            shape="line",
            coordinates=(0, 0, 3, 1),
            color="red",
            quality=cast(Any, True),
        )


def test_rejects_quality_for_png(rgb_png: Path, tmp_path: Path) -> None:
    """Characterizes current behaviour; the contract is undecided (see tests/image/README.md)."""
    with pytest.raises(ValueError, match="only for JPEG"):
        annotate_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            shape="line",
            coordinates=(0, 0, 3, 1),
            color="red",
            quality=80,
        )


def test_rejects_unknown_shape(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rectangle or line"):
        annotate_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            shape="circle",
            coordinates=(0, 0, 2, 2),
            color="red",
        )


def test_rejects_unknown_color(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Color must be one of"):
        annotate_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            shape="line",
            coordinates=(0, 0, 2, 2),
            color="pink",
        )


@pytest.mark.parametrize(
    "coordinates",
    [
        (0, 0, 2),
        (0, 0, 2, 2.0),
        (0, 0, True, 2),
        "0,0,2,2",
    ],
)
def test_rejects_malformed_coordinates(
    rgb_png: Path,
    tmp_path: Path,
    coordinates: Any,
) -> None:
    with pytest.raises(TypeError, match="exactly four integers"):
        annotate_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            shape="line",
            coordinates=coordinates,
            color="red",
        )


def test_rejects_negative_coordinates(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="negative"):
        annotate_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            shape="line",
            coordinates=(-1, 0, 2, 2),
            color="red",
        )


def test_rejects_inverted_rectangle(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="x2 > x1"):
        annotate_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            shape="rectangle",
            coordinates=(2, 0, 2, 2),
            color="red",
        )


def test_rejects_degenerate_line(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="endpoints must be different"):
        annotate_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            shape="line",
            coordinates=(1, 1, 1, 1),
            color="red",
        )


def test_rejects_rectangle_outside_bounds(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exceeds image bounds"):
        annotate_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            shape="rectangle",
            coordinates=(0, 0, 5, 2),
            color="red",
        )


def test_rejects_line_outside_bounds(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"within 0\.\.3"):
        annotate_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            shape="line",
            coordinates=(0, 0, 4, 1),
            color="red",
        )


@pytest.mark.parametrize("extension", ["jpg", "bmp"])
def test_rejects_transparency_for_lossy_formats(
    rgba_png: Path,
    tmp_path: Path,
    extension: str,
) -> None:
    with pytest.raises(ValueError, match="Use PNG, WEBP or TIFF"):
        annotate_image(
            source=rgba_png,
            output=tmp_path / f"out.{extension}",
            shape="line",
            coordinates=(0, 0, 2, 1),
            color="red",
        )


def test_rejects_same_source_and_output(rgb_png: Path) -> None:
    with pytest.raises(ValueError, match="must be different"):
        annotate_image(
            source=rgb_png,
            output=rgb_png,
            shape="line",
            coordinates=(0, 0, 2, 1),
            color="red",
        )


def test_rejects_existing_output(rgb_png: Path, tmp_path: Path) -> None:
    output = tmp_path / "exists.png"
    output.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError, match="already exists"):
        annotate_image(
            source=rgb_png,
            output=output,
            shape="line",
            coordinates=(0, 0, 2, 1),
            color="red",
        )

    assert output.read_bytes() == b"sentinel"


def test_rejects_unsupported_extension(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported output format"):
        annotate_image(
            source=rgb_png,
            output=tmp_path / "out.tga",
            shape="line",
            coordinates=(0, 0, 2, 1),
            color="red",
        )


def test_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        annotate_image(
            source=tmp_path / "absent.png",
            output=tmp_path / "out.png",
            shape="line",
            coordinates=(0, 0, 2, 1),
            color="red",
        )


def test_rejects_directory_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Not a regular file"):
        annotate_image(
            source=tmp_path,
            output=tmp_path / "out.png",
            shape="line",
            coordinates=(0, 0, 2, 1),
            color="red",
        )


def test_rejects_invalid_image(garbage_file: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid or unsafe image"):
        annotate_image(
            source=garbage_file,
            output=tmp_path / "out.png",
            shape="line",
            coordinates=(0, 0, 2, 1),
            color="red",
        )


def test_rejects_multi_frame_image(animated_gif: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Multi-frame"):
        annotate_image(
            source=animated_gif,
            output=tmp_path / "out.png",
            shape="line",
            coordinates=(0, 0, 2, 1),
            color="red",
        )


def test_enforces_file_size_limit(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(annotate_module, "MAX_FILE_SIZE", 10)

    with pytest.raises(ValueError, match="between 1 byte and 100 MiB"):
        annotate_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            shape="line",
            coordinates=(0, 0, 2, 1),
            color="red",
        )


def test_enforces_pixel_limit(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(annotate_module, "MAX_PIXELS", 3)

    with pytest.raises(ValueError, match="pixel limit"):
        annotate_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            shape="line",
            coordinates=(0, 0, 2, 1),
            color="red",
        )


def test_creates_nested_output_directory(rgb_png: Path, tmp_path: Path) -> None:
    output = tmp_path / "nested" / "deep" / "out.png"

    annotate_image(
        source=rgb_png,
        output=output,
        shape="line",
        coordinates=(0, 0, 2, 1),
        color="red",
    )

    assert output.is_file()


def test_enforces_output_file_size_limit(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(annotate_module, "MAX_FILE_SIZE", 300)

    with pytest.raises(ValueError, match="Output file must be between"):
        annotate_image(
            source=rgb_png,
            output=tmp_path / "out.jpg",
            shape="line",
            coordinates=(0, 0, 3, 1),
            color="red",
        )


def test_rejects_invalid_stroke_width(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(annotate_module, "STROKE_WIDTH", 0)

    with pytest.raises(ValueError, match="Stroke width"):
        annotate_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            shape="line",
            coordinates=(0, 0, 2, 1),
            color="red",
        )


def test_does_not_modify_source_and_leaves_no_artifacts(
    rgb_png: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable,
) -> None:
    before = _digest(rgb_png)

    annotate_image(
        source=rgb_png,
        output=tmp_path / "out.png",
        shape="line",
        coordinates=(0, 0, 2, 1),
        color="yellow",
    )

    assert _digest(rgb_png) == before
    assert_no_leftovers(tmp_path)
