from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, PngImagePlugin


def _fill_unique(image: Image.Image) -> None:
    for x in range(image.width):
        for y in range(image.height):
            channels = [20 + x * 40, 30 + y * 90, 200]
            if image.mode == "RGBA":
                channels.append(40 * (x + 1))
            image.putpixel((x, y), tuple(channels))


@pytest.fixture
def image_factory(tmp_path: Path) -> Callable[..., Path]:
    def create(
        name: str = "source.png",
        size: tuple[int, int] = (4, 2),
        mode: str = "RGB",
        *,
        transparent: bool = False,
        **save_options: Any,
    ) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new(mode, size)
        try:
            if mode in ("RGB", "RGBA"):
                _fill_unique(image)
            elif mode == "P":
                image.putpalette([255, 255, 255, 0, 0, 0])
                for x in range(size[0]):
                    for y in range(size[1]):
                        image.putpixel((x, y), (x + y) % 2)
                if transparent:
                    image.info["transparency"] = 0
            elif mode == "1":
                for x in range(size[0]):
                    for y in range(size[1]):
                        image.putpixel((x, y), 255 if (x + y) % 2 else 0)
            else:
                raise ValueError(f"Unsupported fixture mode: {mode}")
            image.save(path, **save_options)
        finally:
            image.close()
        return path

    return create


@pytest.fixture
def rgb_png(image_factory: Callable[..., Path]) -> Path:
    return image_factory()


@pytest.fixture
def rgba_png(image_factory: Callable[..., Path]) -> Path:
    return image_factory("alpha.png", mode="RGBA")


@pytest.fixture
def palette_png(image_factory: Callable[..., Path]) -> Path:
    return image_factory("palette.png", mode="P", transparent=True)


@pytest.fixture
def opaque_palette_png(image_factory: Callable[..., Path]) -> Path:
    return image_factory("opaque_palette.png", mode="P")


@pytest.fixture
def rgb_jpeg(image_factory: Callable[..., Path]) -> Path:
    return image_factory("source.jpg", quality=95)


@pytest.fixture
def animated_gif(tmp_path: Path) -> Path:
    path = tmp_path / "animation.gif"
    first = Image.new("RGB", (4, 2), "red")
    second = Image.new("RGB", (4, 2), "blue")
    try:
        first.save(path, save_all=True, append_images=[second], duration=50, loop=0)
    finally:
        first.close()
        second.close()
    return path


@pytest.fixture
def garbage_file(tmp_path: Path) -> Path:
    path = tmp_path / "garbage.png"
    path.write_bytes(b"this is not an image")
    return path


@pytest.fixture
def empty_file(tmp_path: Path) -> Path:
    path = tmp_path / "empty.png"
    path.write_bytes(b"")
    return path


@pytest.fixture
def padded_png(tmp_path: Path) -> Path:
    """PNG whose ancillary text makes the file larger than its pixel payload."""
    path = tmp_path / "padded.png"
    image = Image.new("RGB", (4, 2))
    try:
        _fill_unique(image)
        info = PngImagePlugin.PngInfo()
        info.add_text("comment", "x" * 50_000)
        image.save(path, pnginfo=info)
    finally:
        image.close()
    return path


@pytest.fixture
def symlink_or_skip() -> Callable[[Path, Path], None]:
    def create(link: Path, target: Path) -> None:
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlink creation is not available: {exc}")

    return create


@pytest.fixture
def read_pixels() -> Callable[[str | Path], list[list[Any]]]:
    def read(path: str | Path) -> list[list[Any]]:
        with Image.open(path) as image:
            image.load()
            return [
                [image.getpixel((x, y)) for x in range(image.width)]
                for y in range(image.height)
            ]

    return read


@pytest.fixture
def assert_no_leftovers() -> Callable[[Path], None]:
    def check(directory: Path) -> None:
        leftovers = [
            entry.name
            for entry in Path(directory).rglob("*")
            if entry.name.startswith(".agentkit-") or entry.name.endswith(".tmp")
        ]
        assert not leftovers, f"temporary artifacts left: {leftovers}"

    return check


@pytest.fixture
def run_cli() -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(*arguments: object) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            "-m",
            "agentkit.cli",
            *(str(argument) for argument in arguments),
        ]
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=60,
        )

    return run
