from __future__ import annotations

import errno
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image, ImageFont

from agentkit.image import grid as grid_module
from agentkit.image.grid import grid_image

RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (41, 98, 255)


def _solid(
    tmp_path: Path,
    name: str,
    size: tuple[int, int],
    color: str = "white",
) -> Path:
    path = tmp_path / name
    with Image.new("RGB", size, color) as image:
        image.save(path)
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _blend(
    background: tuple[int, int, int],
    color: tuple[int, int, int],
    opacity: int,
) -> tuple[int, int, int]:
    alpha = round(255 * opacity / 100)
    return (
        round((background[0] * (255 - alpha) + color[0] * alpha) / 255),
        round((background[1] * (255 - alpha) + color[1] * alpha) / 255),
        round((background[2] * (255 - alpha) + color[2] * alpha) / 255),
    )


@pytest.mark.parametrize(
    ("name", "expected_format"),
    [("out.png", "PNG"), ("out.jpg", "JPEG")],
)
def test_writes_supported_formats(
    rgb_png: Path,
    tmp_path: Path,
    name: str,
    expected_format: str,
) -> None:
    output = tmp_path / name

    result = grid_image(source=rgb_png, output=output, labels=False)

    assert result["status"] == "ok"
    assert Path(result["output"]).samefile(output)
    with Image.open(output) as image:
        assert image.format == expected_format
        assert image.size == (4, 2)


def test_draws_grid_lines_at_step_positions(tmp_path: Path) -> None:
    source = _solid(tmp_path, "canvas.png", (20, 20))
    output = tmp_path / "grid.png"

    grid_image(source, output, step=5, opacity=100, labels=False)

    with Image.open(output) as image:
        assert image.getpixel((0, 0)) == RED
        for x in (5, 10, 15):
            assert image.getpixel((x, 1)) == RED
        for y in (5, 10, 15):
            assert image.getpixel((1, y)) == RED

        assert image.getpixel((6, 1)) == WHITE
        assert image.getpixel((1, 6)) == WHITE
        # No forced lines at the right or bottom edges.
        assert image.getpixel((19, 1)) == WHITE
        assert image.getpixel((1, 19)) == WHITE
        assert image.getpixel((19, 19)) == WHITE


def test_default_step_is_100(tmp_path: Path) -> None:
    source = _solid(tmp_path, "canvas.png", (250, 150))
    output = tmp_path / "grid.png"

    grid_image(source, output, opacity=100, labels=False)

    with Image.open(output) as image:
        assert image.getpixel((100, 1)) == RED
        assert image.getpixel((200, 1)) == RED
        assert image.getpixel((50, 1)) == WHITE
        assert image.getpixel((150, 1)) == WHITE


def test_default_color_is_red(tmp_path: Path) -> None:
    source = _solid(tmp_path, "canvas.png", (20, 20))
    output = tmp_path / "grid.png"

    grid_image(source, output, step=10, opacity=100, labels=False)

    with Image.open(output) as image:
        assert image.getpixel((10, 1)) == RED


def test_default_opacity_is_35(tmp_path: Path) -> None:
    source = _solid(tmp_path, "canvas.png", (20, 20))
    output = tmp_path / "grid.png"

    grid_image(source, output, step=10, labels=False)

    with Image.open(output) as image:
        assert image.getpixel((10, 1)) == _blend(WHITE, RED, 35)


def test_blends_custom_color_and_opacity(tmp_path: Path) -> None:
    source = _solid(tmp_path, "canvas.png", (30, 20))
    output = tmp_path / "grid.png"

    grid_image(
        source,
        output,
        step=10,
        color="blue",
        opacity=50,
        labels=False,
    )

    with Image.open(output) as image:
        assert image.getpixel((10, 1)) == _blend(WHITE, BLUE, 50)


def test_draws_labels_by_default(tmp_path: Path) -> None:
    source = _solid(tmp_path, "canvas.png", (120, 80))
    labeled = tmp_path / "labeled.png"
    plain = tmp_path / "plain.png"

    grid_image(source, labeled, step=40, opacity=100)
    grid_image(source, plain, step=40, opacity=100, labels=False)

    with Image.open(labeled) as labeled_image, Image.open(plain) as plain_image:
        assert labeled_image.getpixel((2, 2)) == BLACK
        assert plain_image.getpixel((2, 2)) == WHITE
        assert labeled_image.getpixel((2, 2)) != plain_image.getpixel((2, 2))


def test_no_labels_disables_label_overlay(tmp_path: Path) -> None:
    source = _solid(tmp_path, "canvas.png", (120, 80))
    output = tmp_path / "plain.png"

    grid_image(source, output, step=40, opacity=100, labels=False)

    with Image.open(output) as image:
        assert image.getpixel((2, 2)) == WHITE
        assert image.getpixel((40, 1)) == RED


@pytest.mark.parametrize("background", ["white", "black"])
def test_labels_remain_readable_on_contrasting_backgrounds(
    tmp_path: Path,
    background: str,
) -> None:
    source = _solid(tmp_path, f"{background}.png", (120, 80), color=background)
    output = tmp_path / f"{background}-grid.png"

    grid_image(source, output, step=100, opacity=100)

    with Image.open(output) as image:
        pixels = [
            cast(tuple[int, int, int], image.getpixel((x, y)))
            for y in range(2, 18)
            for x in range(2, 40)
        ]
    luminances = [sum(pixel) / 3 for pixel in pixels]
    assert max(luminances) - min(luminances) > 100


def test_skips_labels_that_would_overlap() -> None:
    font = ImageFont.load_default(size=grid_module.LABEL_FONT_SIZE)

    labels = grid_module._label_layout(120, 120, 10, font)

    assert labels
    boxes = [box for _, box in labels]
    for index, first in enumerate(boxes):
        for second in boxes[index + 1 :]:
            assert not grid_module._boxes_overlap(
                first,
                second,
                grid_module.LABEL_GAP,
            )
    assert len(labels) < len(range(0, 120, 10)) + len(range(10, 120, 10))


def test_small_step_skips_labels_but_keeps_grid_spacing(tmp_path: Path) -> None:
    source = _solid(tmp_path, "canvas.png", (120, 120))
    output = tmp_path / "grid.png"

    grid_image(source, output, step=10, opacity=100)

    with Image.open(output) as image:
        for x in (10, 20, 30, 100):
            assert image.getpixel((x, 119)) == RED
        assert image.getpixel((119, 110)) == RED


def test_zero_opacity_leaves_pixels_unchanged(
    tmp_path: Path,
    read_pixels: Callable[[str | Path], list[list[Any]]],
) -> None:
    source = tmp_path / "source.png"
    with Image.new("RGB", (40, 30)) as image:
        for x in range(40):
            for y in range(30):
                image.putpixel((x, y), (x * 5 % 256, y * 7 % 256, (x + y) * 3 % 256))
        image.save(source)
    output = tmp_path / "zero.png"

    grid_image(source, output, opacity=0)

    assert read_pixels(output) == read_pixels(source)
    with Image.open(output) as image:
        assert image.size == (40, 30)


def test_zero_opacity_preserves_lossless_png_mode(
    palette_png: Path,
    tmp_path: Path,
    read_pixels: Callable[[str | Path], list[list[Any]]],
) -> None:
    output = tmp_path / "zero-palette.png"

    grid_image(palette_png, output, opacity=0)

    with Image.open(palette_png) as original, Image.open(output) as result:
        assert result.mode == original.mode
        assert result.size == original.size
    assert read_pixels(output) == read_pixels(palette_png)


@pytest.mark.parametrize("name", ["out.png", "out.jpg"])
def test_preserves_image_dimensions(tmp_path: Path, name: str) -> None:
    source = _solid(tmp_path, "odd.png", (37, 19))
    output = tmp_path / name

    grid_image(source, output)

    with Image.open(output) as image:
        assert image.size == (37, 19)


@pytest.mark.parametrize("step", [0, -1])
def test_rejects_non_positive_step(
    rgb_png: Path,
    tmp_path: Path,
    step: int,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        grid_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            step=step,
            labels=False,
        )


@pytest.mark.parametrize("step", [1.5, True, "10"])
def test_rejects_non_integer_step(
    rgb_png: Path,
    tmp_path: Path,
    step: Any,
) -> None:
    with pytest.raises(TypeError, match="integer"):
        grid_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            step=step,
            labels=False,
        )


@pytest.mark.parametrize("opacity", [-1, 101])
def test_rejects_opacity_out_of_range(
    rgb_png: Path,
    tmp_path: Path,
    opacity: int,
) -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        grid_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            opacity=opacity,
            labels=False,
        )


@pytest.mark.parametrize("opacity", [1.5, True, "35"])
def test_rejects_non_integer_opacity(
    rgb_png: Path,
    tmp_path: Path,
    opacity: Any,
) -> None:
    with pytest.raises(TypeError, match="integer"):
        grid_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            opacity=opacity,
            labels=False,
        )


def test_rejects_unknown_color(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Color must be one of"):
        grid_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            color="pink",
            labels=False,
        )


@pytest.mark.parametrize("quality", [1, 95])
def test_accepts_jpeg_quality(
    rgb_png: Path,
    tmp_path: Path,
    quality: int,
) -> None:
    output = tmp_path / f"quality_{quality}.jpg"

    grid_image(
        source=rgb_png,
        output=output,
        quality=quality,
        labels=False,
    )

    with Image.open(output) as image:
        assert image.format == "JPEG"
        assert image.size == (4, 2)


def test_default_jpeg_quality_matches_explicit_90(tmp_path: Path) -> None:
    source = _solid(tmp_path, "canvas.png", (64, 64))
    default = tmp_path / "default.jpg"
    explicit = tmp_path / "explicit.jpg"

    grid_image(source, default, step=16, labels=False)
    grid_image(source, explicit, step=16, labels=False, quality=90)

    assert default.read_bytes() == explicit.read_bytes()


@pytest.mark.parametrize("quality", [0, 96, -1])
def test_rejects_quality_out_of_range(
    rgb_png: Path,
    tmp_path: Path,
    quality: int,
) -> None:
    with pytest.raises(ValueError, match="between 1 and 95"):
        grid_image(
            source=rgb_png,
            output=tmp_path / "out.jpg",
            quality=quality,
            labels=False,
        )


def test_rejects_non_integer_quality(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="integer"):
        grid_image(
            source=rgb_png,
            output=tmp_path / "out.jpg",
            quality=cast(Any, True),
            labels=False,
        )


def test_rejects_quality_for_png(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only for JPEG"):
        grid_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            quality=80,
            labels=False,
        )


def test_rejects_same_source_and_output(rgb_png: Path) -> None:
    with pytest.raises(ValueError, match="must be different"):
        grid_image(source=rgb_png, output=rgb_png, labels=False)


def test_rejects_existing_output(
    rgb_png: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    output = tmp_path / "exists.png"
    output.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError, match="already exists"):
        grid_image(source=rgb_png, output=output, labels=False)

    assert output.read_bytes() == b"sentinel"
    assert_no_leftovers(tmp_path)


def test_rejects_unsupported_extension(rgb_png: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported output format"):
        grid_image(
            source=rgb_png,
            output=tmp_path / "out.tga",
            labels=False,
        )


def test_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        grid_image(
            source=tmp_path / "absent.png",
            output=tmp_path / "out.png",
            labels=False,
        )


def test_rejects_directory_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Not a regular file"):
        grid_image(source=tmp_path, output=tmp_path / "out.png", labels=False)


def test_rejects_empty_source(
    empty_file: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    output = tmp_path / "out.png"

    with pytest.raises(ValueError, match="between 1 byte"):
        grid_image(source=empty_file, output=output, labels=False)

    assert not output.exists()
    assert_no_leftovers(tmp_path)


def test_rejects_invalid_image(garbage_file: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid or unsafe image"):
        grid_image(
            source=garbage_file,
            output=tmp_path / "out.png",
            labels=False,
        )


def test_rejects_multi_frame_image(animated_gif: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Multi-frame"):
        grid_image(
            source=animated_gif,
            output=tmp_path / "out.png",
            labels=False,
        )


@pytest.mark.parametrize("extension", ["jpg", "bmp"])
def test_rejects_transparency_for_lossy_formats(
    rgba_png: Path,
    tmp_path: Path,
    extension: str,
) -> None:
    with pytest.raises(ValueError, match="Use PNG, WEBP or TIFF"):
        grid_image(
            source=rgba_png,
            output=tmp_path / f"out.{extension}",
            labels=False,
        )


def test_preserves_alpha_for_png(rgba_png: Path, tmp_path: Path) -> None:
    output = tmp_path / "alpha-grid.png"

    grid_image(rgba_png, output, step=2, opacity=100, labels=False)

    with Image.open(rgba_png) as original, Image.open(output) as result:
        assert result.mode == "RGBA"
        assert result.getpixel((3, 1)) == original.getpixel((3, 1))
        assert cast(tuple[int, int, int, int], result.getpixel((2, 1)))[3] == 255


def test_enforces_source_file_size_limit(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(grid_module, "MAX_FILE_SIZE", 10)

    with pytest.raises(ValueError, match="between 1 byte and 100 MiB"):
        grid_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            labels=False,
        )


def test_enforces_pixel_limit(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(grid_module, "MAX_PIXELS", 3)

    with pytest.raises(ValueError, match="pixel limit"):
        grid_image(
            source=rgb_png,
            output=tmp_path / "out.png",
            labels=False,
        )


def test_enforces_output_file_size_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    source = _solid(tmp_path, "canvas.png", (64, 64))
    measured = tmp_path / "measured.jpg"
    grid_image(source, measured, step=16, labels=False)
    encoded_size = measured.stat().st_size
    measured.unlink()

    monkeypatch.setattr(grid_module, "MAX_FILE_SIZE", encoded_size)
    accepted = tmp_path / "accepted.jpg"
    grid_image(source, accepted, step=16, labels=False)
    assert accepted.stat().st_size == encoded_size

    monkeypatch.setattr(grid_module, "MAX_FILE_SIZE", encoded_size - 1)
    rejected = tmp_path / "rejected.jpg"
    with pytest.raises(ValueError, match="Output file must be between"):
        grid_image(source, rejected, step=16, labels=False)

    assert not rejected.exists()
    assert_no_leftovers(tmp_path)


def test_creates_nested_output_directory(
    rgb_png: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    output = tmp_path / "nested" / "deep" / "out.png"

    grid_image(source=rgb_png, output=output, labels=False)

    assert output.is_file()
    assert_no_leftovers(tmp_path)


def test_does_not_modify_source_and_leaves_no_artifacts(
    rgb_png: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    before = _digest(rgb_png)

    grid_image(
        source=rgb_png,
        output=tmp_path / "out.png",
        step=2,
        opacity=100,
        labels=True,
    )

    assert _digest(rgb_png) == before
    assert_no_leftovers(tmp_path)


def test_hardlink_failure_fails_without_publishing(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    output = tmp_path / "out.png"

    def unsupported_link(source: Any, destination: Any) -> None:
        raise OSError(errno.EOPNOTSUPP, "Hard links are not supported")

    monkeypatch.setattr(grid_module.os, "link", unsupported_link)

    with pytest.raises(OSError, match="not supported"):
        grid_image(source=rgb_png, output=output, labels=False)

    assert not output.exists()
    assert_no_leftovers(tmp_path)


def test_racing_writer_is_not_overwritten(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    output = tmp_path / "result.png"
    sentinel = b"competitor wrote this file"
    real_link = os.link

    def racing_link(source: Any, destination: Any) -> None:
        destination_path = Path(destination)
        if destination_path == output:
            destination_path.write_bytes(sentinel)
        real_link(source, destination)

    monkeypatch.setattr(grid_module.os, "link", racing_link)

    with pytest.raises(FileExistsError):
        grid_image(source=rgb_png, output=output, labels=False)

    assert output.read_bytes() == sentinel
    assert_no_leftovers(tmp_path)


def test_dangling_symlink_output_is_rejected(
    rgb_png: Path,
    symlink_or_skip: Callable[[Path, Path], None],
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    output = tmp_path / "dangling.png"
    missing = tmp_path / "missing-target.png"
    symlink_or_skip(output, missing)

    with pytest.raises(FileExistsError, match="already exists"):
        grid_image(source=rgb_png, output=output, labels=False)

    assert output.is_symlink()
    assert not output.exists()
    assert not missing.exists()
    assert_no_leftovers(tmp_path)


def test_cli_draws_labels_by_default(
    run_cli: Callable[..., Any],
    tmp_path: Path,
) -> None:
    source = _solid(tmp_path, "source.png", (120, 80))
    output = tmp_path / "cli-labeled.png"

    result = run_cli(
        "image",
        "grid",
        str(source),
        str(output),
        "--step",
        "40",
        "--opacity",
        "100",
    )

    assert result.returncode == 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert Path(payload["output"]).samefile(output)
    with Image.open(output) as image:
        assert image.size == (120, 80)
        assert image.getpixel((2, 2)) == BLACK


def test_cli_no_labels_disables_labels(
    run_cli: Callable[..., Any],
    tmp_path: Path,
) -> None:
    source = _solid(tmp_path, "source.png", (120, 80))
    output = tmp_path / "cli-plain.png"

    result = run_cli(
        "image",
        "grid",
        str(source),
        str(output),
        "--step",
        "40",
        "--opacity",
        "100",
        "--no-labels",
    )

    assert result.returncode == 0
    assert result.stderr == ""
    with Image.open(output) as image:
        assert image.getpixel((2, 2)) == WHITE
        assert image.getpixel((40, 1)) == RED


def test_cli_help_exposes_grid_defaults(run_cli: Callable[..., Any]) -> None:
    result = run_cli("image", "grid", "--help")

    assert result.returncode == 0
    assert result.stderr == ""
    assert "default: 100" in result.stdout
    assert "default: 35" in result.stdout
    assert "default: red" in result.stdout
    assert "--no-labels" in result.stdout
    assert "default: 90" in result.stdout


def test_cli_rejects_invalid_step_as_json(
    run_cli: Callable[..., Any],
    tmp_path: Path,
) -> None:
    source = _solid(tmp_path, "source.png", (20, 20))

    result = run_cli(
        "image",
        "grid",
        str(source),
        str(tmp_path / "out.png"),
        "--step",
        "0",
    )

    assert result.returncode == 1
    assert result.stdout == ""
    payload = json.loads(result.stderr)
    assert payload["status"] == "error"
    assert payload["type"] == "ValueError"
    assert "positive integer" in payload["error"]
    assert "Traceback" not in result.stderr


def test_cli_rejects_unknown_color_as_argument_error(
    run_cli: Callable[..., Any],
    tmp_path: Path,
) -> None:
    source = _solid(tmp_path, "source.png", (20, 20))

    result = run_cli(
        "image",
        "grid",
        str(source),
        str(tmp_path / "out.png"),
        "--color",
        "pink",
    )

    assert result.returncode == 2
    assert result.stdout == ""
    payload = json.loads(result.stderr)
    assert payload["status"] == "error"
    assert payload["type"] == "CLIArgumentError"
