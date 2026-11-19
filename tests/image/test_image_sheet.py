from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image

from agentkit.image import sheet as sheet_module
from agentkit.image.sheet import sheet_images

CELL_WIDTH = 455
CELL_HEIGHT = 333
IMAGE_HEIGHT = 303
LEFT = (20, 495, 970, 1445)
TOP = (20, 373, 726)
COLORS = [
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
]


def _sheets(result: dict[str, object]) -> list[dict[str, Any]]:
    sheets = result["sheets"]
    assert isinstance(sheets, list)
    return cast("list[dict[str, Any]]", sheets)


def _names(sheet: dict[str, Any]) -> list[str]:
    return [Path(str(entry["source"])).name for entry in sheet["images"]]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _solid_sources(tmp_path: Path, count: int) -> list[Path]:
    sources = []
    for index in range(count):
        path = tmp_path / f"{index:02d}.png"
        with Image.new("RGB", (CELL_WIDTH, IMAGE_HEIGHT), COLORS[index]) as image:
            image.save(path)
        sources.append(path)
    return sources


def _expected_box(slot: int) -> tuple[int, int, int, int]:
    row, column = divmod(slot, 4)
    left = LEFT[column]
    top = TOP[row]
    return left, top, left + CELL_WIDTH, top + IMAGE_HEIGHT


def test_creates_one_sheet_for_few_images(image_factory, tmp_path: Path) -> None:
    sources = [image_factory(f"{index}.png") for index in range(4)]
    output = tmp_path / "sheets"

    result = sheet_images(sources=sources, output=output)

    assert result["status"] == "ok"
    assert result["count"] == 4
    assert result["per_sheet"] == 12
    assert result["columns"] == 4
    assert result["rows"] == 3
    sheets = _sheets(result)
    assert len(sheets) == 1
    assert Path(str(sheets[0]["output"])).name == "sheet_0001.png"
    assert [entry["index"] for entry in sheets[0]["images"]] == [1, 2, 3, 4]
    with Image.open(str(sheets[0]["output"])) as image:
        assert image.size == (1920, 1080)


def test_splits_images_between_sheets(image_factory, tmp_path: Path) -> None:
    sources = [image_factory(f"{index:02d}.png") for index in range(13)]
    output = tmp_path / "sheets"

    result = sheet_images(sources=sources, output=output, per_sheet=12)

    sheets = _sheets(result)
    assert len(sheets) == 2
    assert [Path(str(sheet["output"])).name for sheet in sheets] == [
        "sheet_0001.png",
        "sheet_0002.png",
    ]
    assert [entry["index"] for entry in sheets[1]["images"]] == [13]


def test_renders_sources_in_expected_cells_on_every_page(tmp_path: Path) -> None:
    path_sources = _solid_sources(tmp_path, 13)
    sources: list[str | Path] = list(path_sources)

    result = sheet_images(
        sources=sources,
        output=tmp_path / "sheets",
        per_sheet=12,
    )

    sheets = _sheets(result)
    assert len(sheets) == 2
    for page, sheet in enumerate(sheets):
        start = page * 12
        expected_colors = COLORS[start:start + 12]
        expected_names = [Path(path).name for path in sources[start:start + 12]]
        entries = sheet["images"]

        assert _names(sheet) == expected_names
        assert [entry["index"] for entry in entries] == list(
            range(start + 1, start + len(expected_colors) + 1)
        )
        with Image.open(str(sheet["output"])) as image:
            assert image.size == (1920, 1080)
            for slot, (entry, color) in enumerate(
                zip(entries, expected_colors, strict=True)
            ):
                left, top, right, bottom = _expected_box(slot)
                assert tuple(entry["box"]) == (left, top, right, bottom)
                assert image.getpixel((left, top)) == color
                assert image.getpixel((right - 1, top)) == color
                assert image.getpixel((left, bottom - 1)) == color
                assert image.getpixel((right - 1, bottom - 1)) == color
                assert image.getpixel(
                    ((left + right) // 2, (top + bottom) // 2)
                ) == color
                assert image.getpixel((right, top)) == (255, 255, 255)
                assert image.getpixel((left, bottom)) == (255, 255, 255)


def test_accepts_single_directory(image_factory, tmp_path: Path) -> None:
    directory = tmp_path / "inputs"
    for index in range(3):
        image_factory(f"inputs/{index}.png")

    result = sheet_images(sources=directory, output=tmp_path / "sheets")

    assert result["count"] == 3


def test_accepts_plain_string_path(image_factory, tmp_path: Path) -> None:
    source = image_factory("single.png")

    result = sheet_images(sources=str(source), output=tmp_path / "sheets")

    assert result["count"] == 1


def test_sorts_directory_case_insensitively(image_factory, tmp_path: Path) -> None:
    directory = tmp_path / "inputs"
    for name in ("B.png", "a.png", "c.png"):
        image_factory(f"inputs/{name}")

    result = sheet_images(sources=directory, output=tmp_path / "sheets")

    assert _names(_sheets(result)[0]) == ["a.png", "B.png", "c.png"]


def test_per_sheet_one(image_factory, tmp_path: Path) -> None:
    sources = [image_factory(f"{index}.png") for index in range(2)]

    result = sheet_images(sources=sources, output=tmp_path / "sheets", per_sheet=1)

    assert result["columns"] == 1
    assert result["rows"] == 1
    assert len(_sheets(result)) == 2


def test_manifest_boxes_are_inside_canvas(image_factory, tmp_path: Path) -> None:
    sources = [image_factory(f"{index}.png") for index in range(4)]

    result = sheet_images(sources=sources, output=tmp_path / "sheets")
    sheet = _sheets(result)[0]
    boxes = [entry["box"] for entry in sheet["images"]]

    for left, top, right, bottom in boxes:
        assert 0 <= left < right <= 1920
        assert 0 <= top < bottom <= 1080
    for index, first in enumerate(boxes):
        for second in boxes[index + 1:]:
            separated = first[2] <= second[0] or second[2] <= first[0]
            separated = separated or first[3] <= second[1] or second[3] <= first[1]
            assert separated


def test_reuses_existing_output_directory(image_factory, tmp_path: Path) -> None:
    source = image_factory("only.png")
    output = tmp_path / "sheets"
    output.mkdir()

    sheet_images(sources=[source], output=output)

    assert (output / "sheet_0001.png").is_file()


def test_rejects_existing_sheet_file(image_factory, tmp_path: Path) -> None:
    source = image_factory("only.png")
    output = tmp_path / "sheets"
    output.mkdir()
    existing = output / "sheet_0001.png"
    existing.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError, match="already exists"):
        sheet_images(sources=[source], output=output)

    assert existing.read_bytes() == b"sentinel"


@pytest.mark.parametrize("per_sheet", [0, -1])
def test_rejects_non_positive_per_sheet(
    image_factory,
    tmp_path: Path,
    per_sheet: int,
) -> None:
    source = image_factory("only.png")

    with pytest.raises(ValueError, match="must be positive"):
        sheet_images(sources=[source], output=tmp_path / "sheets", per_sheet=per_sheet)


@pytest.mark.parametrize("per_sheet", [1.5, True, "12"])
def test_rejects_non_integer_per_sheet(
    image_factory,
    tmp_path: Path,
    per_sheet: Any,
) -> None:
    source = image_factory("only.png")

    with pytest.raises(TypeError, match="must be an integer"):
        sheet_images(
            sources=[source],
            output=tmp_path / "sheets",
            per_sheet=per_sheet,
        )


def test_rejects_grid_that_cannot_fit(image_factory, tmp_path: Path) -> None:
    source = image_factory("only.png")

    with pytest.raises(ValueError, match="Too many images per sheet"):
        sheet_images(sources=[source], output=tmp_path / "sheets", per_sheet=1000)


def test_rejects_empty_directory(tmp_path: Path) -> None:
    directory = tmp_path / "empty"
    directory.mkdir()

    with pytest.raises(ValueError, match="no supported images"):
        sheet_images(sources=directory, output=tmp_path / "sheets")


def test_rejects_directory_without_supported_images(tmp_path: Path) -> None:
    directory = tmp_path / "inputs"
    directory.mkdir()
    (directory / "notes.txt").write_text("text", encoding="utf-8")

    with pytest.raises(ValueError, match="no supported images"):
        sheet_images(sources=directory, output=tmp_path / "sheets")


def test_rejects_empty_source_list(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one image path"):
        sheet_images(sources=[], output=tmp_path / "sheets")


def test_rejects_directory_mixed_with_file(image_factory, tmp_path: Path) -> None:
    directory = tmp_path / "inputs"
    image_factory("inputs/inside.png")
    outside = image_factory("outside.png")

    with pytest.raises(ValueError, match="Expected an image file"):
        sheet_images(sources=[directory, outside], output=tmp_path / "sheets")


def test_rejects_multi_frame_image(animated_gif: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Multi-frame"):
        sheet_images(sources=[animated_gif], output=tmp_path / "sheets")


def test_enforces_source_file_size_limit(
    monkeypatch: pytest.MonkeyPatch,
    image_factory,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sheet_module, "MAX_FILE_SIZE", 10)
    source = image_factory("only.png")

    with pytest.raises(ValueError, match="between 1 byte and 100 MiB"):
        sheet_images(sources=[source], output=tmp_path / "sheets")


def test_enforces_canvas_pixel_limit(
    monkeypatch: pytest.MonkeyPatch,
    image_factory,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sheet_module, "MAX_PIXELS", 1000)
    source = image_factory("only.png")

    with pytest.raises(ValueError, match="Sheet exceeds pixel limit"):
        sheet_images(sources=[source], output=tmp_path / "sheets")


def test_enforces_output_file_size_limit(
    monkeypatch: pytest.MonkeyPatch,
    image_factory,
    tmp_path: Path,
    assert_no_leftovers: Callable,
) -> None:
    sources = [image_factory(f"{index}.png") for index in range(2)]

    measured = tmp_path / "measured" / "sheets"
    sheet_images(sources=sources, output=measured)
    encoded_size = (measured / "sheet_0001.png").stat().st_size

    monkeypatch.setattr(sheet_module, "MAX_FILE_SIZE", encoded_size)
    accepted = tmp_path / "accepted" / "sheets"
    sheet_images(sources=sources, output=accepted)
    assert (accepted / "sheet_0001.png").stat().st_size == encoded_size

    monkeypatch.setattr(sheet_module, "MAX_FILE_SIZE", encoded_size - 1)
    rejected = tmp_path / "rejected" / "sheets"
    with pytest.raises(ValueError, match="Sheet exceeds 100 MiB limit"):
        sheet_images(sources=sources, output=rejected)

    assert not list(rejected.glob("sheet_*.png"))
    assert_no_leftovers(rejected)


def test_rejects_invalid_canvas_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    image_factory,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sheet_module, "SHEET_WIDTH", 0)
    source = image_factory("only.png")

    with pytest.raises(ValueError, match="dimensions must be positive"):
        sheet_images(sources=[source], output=tmp_path / "sheets")


def test_rejects_invalid_spacing(
    monkeypatch: pytest.MonkeyPatch,
    image_factory,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sheet_module, "LABEL_HEIGHT", 0)
    source = image_factory("only.png")

    with pytest.raises(ValueError, match="Invalid sheet spacing"):
        sheet_images(sources=[source], output=tmp_path / "sheets")


def test_rejects_unsafe_source_image(tmp_path: Path) -> None:
    directory = tmp_path / "inputs"
    directory.mkdir()
    (directory / "broken.png").write_bytes(b"not an image")

    with pytest.raises(ValueError, match="Invalid or unsafe image"):
        sheet_images(sources=directory, output=tmp_path / "sheets")


def test_does_not_modify_sources_and_leaves_no_artifacts(
    image_factory,
    tmp_path: Path,
    assert_no_leftovers: Callable,
) -> None:
    sources = [image_factory(f"{index}.png") for index in range(3)]
    before = {path: _digest(path) for path in sources}
    output = tmp_path / "sheets"

    sheet_images(sources=sources, output=output)

    assert {path: _digest(path) for path in sources} == before
    assert_no_leftovers(output)
