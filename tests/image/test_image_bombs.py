from __future__ import annotations

import struct
import zlib
from collections.abc import Callable
from pathlib import Path

import pytest

from agentkit.image.annotate import annotate_image
from agentkit.image.crop import crop_image
from agentkit.image.info import get_image_info
from agentkit.image.resize import resize_image
from agentkit.image.rotate import rotate_image
from agentkit.image.sheet import sheet_images


def _chunk(tag: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", checksum)


def _png_header(width: int, height: int) -> bytes:
    """Valid PNG container whose IHDR advertises the size but stores no pixels."""
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(b""))
        + _chunk(b"IEND", b"")
    )


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
IDS = list(CASES)


def _output_for(tmp_path: Path, name: str) -> Path:
    return tmp_path / ("sheets" if name == "sheet" else "out.png")


def _assert_nothing_published(name: str, output: Path) -> None:
    if name == "sheet":
        assert not list(output.glob("sheet_*.png"))
    else:
        assert not output.exists()


@pytest.mark.parametrize(
    ("width", "height"),
    [(10_000, 10_000), (20_000, 20_000)],
    ids=["warning", "error"],
)
@pytest.mark.parametrize("name", IDS)
def test_rejects_decompression_bomb_without_allocating(
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
    name: str,
    width: int,
    height: int,
) -> None:
    source = tmp_path / f"bomb_{width}.png"
    source.write_bytes(_png_header(width, height))
    output = _output_for(tmp_path, name)
    message = "Unsupported, invalid or unsafe" if name == "info" else "Invalid or unsafe"

    with pytest.raises(ValueError, match=message):
        CASES[name](source, output)

    _assert_nothing_published(name, output)
    if output.exists():
        assert_no_leftovers(output)


@pytest.mark.parametrize("name", IDS)
def test_rejects_real_pixel_limit_without_allocating(
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
    name: str,
) -> None:
    source = tmp_path / "too_many_pixels.png"
    source.write_bytes(_png_header(8_000, 8_000))
    output = _output_for(tmp_path, name)

    with pytest.raises(ValueError, match="pixel"):
        CASES[name](source, output)

    _assert_nothing_published(name, output)
    if output.exists():
        assert_no_leftovers(output)
