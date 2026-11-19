from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

CASES = [
    ("agentkit.image.crop", "crop_image", {"box": (0, 0, 2, 2)}),
    ("agentkit.image.resize", "resize_image", {"scale": 0.5}),
    ("agentkit.image.rotate", "rotate_image", {"angle": 90}),
    (
        "agentkit.image.annotate",
        "annotate_image",
        {"shape": "rectangle", "coordinates": (0, 0, 2, 2), "color": "red"},
    ),
]
IDS = ["crop", "resize", "rotate", "annotate"]
SENTINEL = b"competitor wrote this file"


def _install_racing_link(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    expected_destination: Path,
    sentinel: bytes,
) -> None:
    module = importlib.import_module(module_name)
    real_link = os.link

    def racing_link(source: Any, destination: Any) -> None:
        destination_path = Path(destination)
        if destination_path == expected_destination:
            destination_path.write_bytes(sentinel)
        real_link(source, destination)

    monkeypatch.setattr(module, "os", SimpleNamespace(link=racing_link))


@pytest.mark.parametrize(("module_name", "function_name", "kwargs"), CASES, ids=IDS)
def test_single_file_publication_never_overwrites_racing_file(
    monkeypatch: pytest.MonkeyPatch,
    image_factory: Callable[..., Path],
    tmp_path: Path,
    assert_no_leftovers: Callable,
    module_name: str,
    function_name: str,
    kwargs: dict[str, Any],
) -> None:
    source = image_factory()
    output = tmp_path / "result.png"
    _install_racing_link(monkeypatch, module_name, output, SENTINEL)
    function = getattr(importlib.import_module(module_name), function_name)

    with pytest.raises(FileExistsError):
        function(source=source, output=output, **kwargs)

    assert output.read_bytes() == SENTINEL
    assert_no_leftovers(tmp_path)


@pytest.mark.parametrize(("module_name", "function_name", "kwargs"), CASES, ids=IDS)
def test_single_file_publication_is_visible_only_after_completion(
    image_factory: Callable[..., Path],
    tmp_path: Path,
    assert_no_leftovers: Callable,
    module_name: str,
    function_name: str,
    kwargs: dict[str, Any],
) -> None:
    source = image_factory()
    output = tmp_path / "result.png"
    function = getattr(importlib.import_module(module_name), function_name)

    function(source=source, output=output, **kwargs)

    assert output.is_file()
    assert not output.name.startswith(".agentkit-")
    assert_no_leftovers(tmp_path)


def test_sheet_publication_never_overwrites_racing_file(
    monkeypatch: pytest.MonkeyPatch,
    image_factory: Callable[..., Path],
    tmp_path: Path,
    assert_no_leftovers: Callable,
) -> None:
    sources: list[str | Path] = [image_factory(f"{index}.png") for index in range(2)]
    output = tmp_path / "sheets"
    output.mkdir()
    destination = output / "sheet_0001.png"
    _install_racing_link(monkeypatch, "agentkit.image.sheet", destination, SENTINEL)

    from agentkit.image.sheet import sheet_images

    with pytest.raises(OSError, match="Could not publish"):
        sheet_images(sources=sources, output=output)

    assert destination.read_bytes() == SENTINEL
    assert_no_leftovers(output)


def test_sheet_publication_leaves_no_artifacts_on_success(
    image_factory: Callable[..., Path],
    tmp_path: Path,
    assert_no_leftovers: Callable,
) -> None:
    from agentkit.image.sheet import sheet_images

    sources: list[str | Path] = [image_factory(f"{index}.png") for index in range(2)]
    output = tmp_path / "sheets"

    sheet_images(sources=sources, output=output)

    assert (output / "sheet_0001.png").is_file()
    assert_no_leftovers(output)
