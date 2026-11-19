from __future__ import annotations

import errno
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


def _install_checking_link(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    seen: list[Path],
) -> None:
    module = importlib.import_module(module_name)
    real_link = os.link

    def checking_link(source: Any, destination: Any) -> None:
        destination_path = Path(destination)
        assert not destination_path.exists(), f"destination already exists: {destination}"
        assert not destination_path.is_symlink(), f"destination is a symlink: {destination}"
        seen.append(destination_path)
        real_link(source, destination)

    monkeypatch.setattr(module.os, "link", checking_link)


def _install_unsupported_link(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    module = importlib.import_module(module_name)

    def unsupported_link(source: Any, destination: Any) -> None:
        raise OSError(errno.EOPNOTSUPP, "Hard links are not supported")

    monkeypatch.setattr(module.os, "link", unsupported_link)


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
    seen: list[Path] = []
    _install_checking_link(monkeypatch, module_name, seen)
    function = getattr(importlib.import_module(module_name), function_name)

    function(source=source, output=output, **kwargs)

    assert seen == [output]
    assert output.is_file()
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


def test_sheet_second_page_race_keeps_first_page_and_cleans_temp(
    monkeypatch: pytest.MonkeyPatch,
    image_factory: Callable[..., Path],
    tmp_path: Path,
    assert_no_leftovers: Callable,
) -> None:
    sources: list[str | Path] = [image_factory(f"{index}.png") for index in range(13)]
    output = tmp_path / "sheets"
    output.mkdir()
    destination = output / "sheet_0002.png"
    _install_racing_link(monkeypatch, "agentkit.image.sheet", destination, SENTINEL)

    from agentkit.image.sheet import sheet_images

    with pytest.raises(OSError, match="Could not publish"):
        sheet_images(sources=sources, output=output, per_sheet=12)

    assert (output / "sheet_0001.png").is_file()
    assert destination.read_bytes() == SENTINEL
    assert not (output / "sheet_0003.png").exists()
    assert_no_leftovers(output)


@pytest.mark.parametrize(("module_name", "function_name", "kwargs"), CASES, ids=IDS)
def test_unsupported_hardlink_fails_without_publishing(
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
    _install_unsupported_link(monkeypatch, module_name)
    function = getattr(importlib.import_module(module_name), function_name)

    with pytest.raises(OSError, match="not supported"):
        function(source=source, output=output, **kwargs)

    assert not output.exists()
    assert_no_leftovers(tmp_path)


def test_sheet_unsupported_hardlink_fails_without_publishing(
    monkeypatch: pytest.MonkeyPatch,
    image_factory: Callable[..., Path],
    tmp_path: Path,
    assert_no_leftovers: Callable,
) -> None:
    sources: list[str | Path] = [image_factory(f"{index}.png") for index in range(2)]
    output = tmp_path / "sheets"
    _install_unsupported_link(monkeypatch, "agentkit.image.sheet")

    from agentkit.image.sheet import sheet_images

    with pytest.raises(OSError, match="Could not publish"):
        sheet_images(sources=sources, output=output)

    assert not (output / "sheet_0001.png").exists()
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


def test_all_sheet_destinations_are_absent_immediately_before_link(
    monkeypatch: pytest.MonkeyPatch,
    image_factory: Callable[..., Path],
    tmp_path: Path,
    assert_no_leftovers: Callable,
) -> None:
    sources: list[str | Path] = [image_factory(f"{index}.png") for index in range(13)]
    output = tmp_path / "sheets"
    seen: list[Path] = []
    _install_checking_link(monkeypatch, "agentkit.image.sheet", seen)

    from agentkit.image.sheet import sheet_images

    result = sheet_images(sources=sources, output=output, per_sheet=12)

    assert [path.name for path in seen] == ["sheet_0001.png", "sheet_0002.png"]
    assert all(path.is_file() for path in seen)
    assert result["count"] == 13
    assert_no_leftovers(output)


@pytest.mark.parametrize(("module_name", "function_name", "kwargs"), CASES, ids=IDS)
def test_dangling_symlink_output_is_rejected(
    image_factory: Callable[..., Path],
    symlink_or_skip: Callable[[Path, Path], None],
    tmp_path: Path,
    assert_no_leftovers: Callable,
    module_name: str,
    function_name: str,
    kwargs: dict[str, Any],
) -> None:
    source = image_factory()
    output = tmp_path / "dangling.png"
    missing = tmp_path / "missing-target.png"
    symlink_or_skip(output, missing)
    function = getattr(importlib.import_module(module_name), function_name)

    with pytest.raises(FileExistsError, match="already exists"):
        function(source=source, output=output, **kwargs)

    assert output.is_symlink()
    assert not output.exists()
    assert not missing.exists()
    assert_no_leftovers(tmp_path)


def test_dangling_symlink_sheet_is_rejected(
    image_factory: Callable[..., Path],
    symlink_or_skip: Callable[[Path, Path], None],
    tmp_path: Path,
    assert_no_leftovers: Callable,
) -> None:
    source = image_factory("only.png")
    output = tmp_path / "sheets"
    output.mkdir()
    missing = tmp_path / "missing-target.png"
    destination = output / "sheet_0001.png"
    symlink_or_skip(destination, missing)

    from agentkit.image.sheet import sheet_images

    with pytest.raises(FileExistsError, match="already exists"):
        sheet_images(sources=[source], output=output)

    assert destination.is_symlink()
    assert not destination.exists()
    assert not missing.exists()
    assert_no_leftovers(output)


@pytest.mark.parametrize(("module_name", "function_name", "kwargs"), CASES, ids=IDS)
def test_path_alias_to_source_is_rejected(
    image_factory: Callable[..., Path],
    tmp_path: Path,
    module_name: str,
    function_name: str,
    kwargs: dict[str, Any],
) -> None:
    source = image_factory()
    alias_directory = tmp_path / "alias"
    alias_directory.mkdir()
    output = alias_directory / ".." / source.name
    function = getattr(importlib.import_module(module_name), function_name)

    with pytest.raises(ValueError, match="must be different"):
        function(source=source, output=output, **kwargs)

    assert source.is_file()
