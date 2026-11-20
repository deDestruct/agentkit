from __future__ import annotations

import errno
import hashlib
import json
import struct
import subprocess
import zlib
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from PIL import Image

from agentkit.image import find as find_module
from agentkit.image.find import find_image, parse_color

TARGET = (255, 0, 0)
BACKGROUND = (0, 0, 0)


def _make_image(
    tmp_path: Path,
    name: str,
    mode: str,
    pixels: list[list[Any]],
    *,
    transparency: int | tuple[int, int, int] | None = None,
) -> Path:
    height = len(pixels)
    width = len(pixels[0])
    path = tmp_path / name
    with Image.new(mode, (width, height)) as image:
        if mode == "P":
            image.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255])
        for y, row in enumerate(pixels):
            for x, value in enumerate(row):
                image.putpixel((x, y), value)
        if transparency is not None:
            image.info["transparency"] = transparency
        image.save(path)
    return path


def _mask_image(
    tmp_path: Path,
    mask: list[list[bool]],
    name: str = "mask.png",
) -> Path:
    return _make_image(
        tmp_path,
        name,
        "RGB",
        [
            [TARGET if value else BACKGROUND for value in row]
            for row in mask
        ],
    )


def _find(
    source: Path,
    *,
    name: str = "result.json",
    color: str = "#FF0000",
    **kwargs: Any,
) -> dict[str, Any]:
    output = source.parent / name
    find_image(source, output, color=color, **kwargs)
    return json.loads(output.read_text(encoding="utf-8"))


def _reference(
    width: int,
    height: int,
    mask: list[list[bool]],
    min_area: int,
) -> dict[str, Any]:
    matched = {
        (x, y)
        for y, row in enumerate(mask)
        for x, value in enumerate(row)
        if value
    }
    seen: set[tuple[int, int]] = set()
    components: list[dict[str, Any]] = []

    for y in range(height):
        for x in range(width):
            if (x, y) not in matched or (x, y) in seen:
                continue

            stack = [(x, y)]
            seen.add((x, y))
            coordinates: list[tuple[int, int]] = []
            while stack:
                current_x, current_y = stack.pop()
                coordinates.append((current_x, current_y))
                for offset_y in (-1, 0, 1):
                    for offset_x in (-1, 0, 1):
                        if offset_x == 0 and offset_y == 0:
                            continue
                        neighbour = (
                            current_x + offset_x,
                            current_y + offset_y,
                        )
                        if neighbour in matched and neighbour not in seen:
                            seen.add(neighbour)
                            stack.append(neighbour)

            pixels = len(coordinates)
            components.append(
                {
                    "box": [
                        min(x for x, _ in coordinates),
                        min(y for _, y in coordinates),
                        max(x for x, _ in coordinates) + 1,
                        max(y for _, y in coordinates) + 1,
                    ],
                    "pixels": pixels,
                    "centroid": [
                        round(
                            sum(x for x, _ in coordinates) / pixels,
                            2,
                        ),
                        round(
                            sum(y for _, y in coordinates) / pixels,
                            2,
                        ),
                    ],
                    "first": min(
                        y * width + x for x, y in coordinates
                    ),
                }
            )

    retained = [
        component
        for component in components
        if component["pixels"] >= min_area
    ]
    retained.sort(
        key=lambda component: (
            -component["pixels"],
            component["box"][1],
            component["box"][0],
            component["box"][3],
            component["box"][2],
            component["first"],
        )
    )
    retained_pixels = sum(
        component["pixels"] for component in retained
    )
    for component in retained:
        del component["first"]

    return {
        "summary": {
            "region_pixels": width * height,
            "matched_pixels": len(matched),
            "matched_percentage": round(
                100 * len(matched) / (width * height),
                2,
            ),
            "component_count": len(components),
            "retained_component_count": len(retained),
            "retained_pixels": retained_pixels,
        },
        "components": retained,
    }


def _assert_matches_reference(
    artifact: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert artifact["summary"] == expected["summary"]
    assert artifact["components"] == expected["components"]


def _transpose(mask: list[list[bool]]) -> list[list[bool]]:
    return [
        [mask[y][x] for y in range(len(mask))]
        for x in range(len(mask[0]))
    ]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(path: Path) -> set[Path]:
    return {entry.relative_to(path) for entry in path.rglob("*")}


def _chunk(tag: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(tag + data) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", checksum)
    )


def _png_header(width: int, height: int) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(b""))
        + _chunk(b"IEND", b"")
    )


def _assert_cli_error(
    result: subprocess.CompletedProcess[str],
    *,
    code: int,
    error_type: str,
    message: str | None = None,
) -> None:
    assert result.returncode == code
    assert result.stdout == ""
    payload = json.loads(result.stderr)
    assert payload["status"] == "error"
    assert payload["type"] == error_type
    if message is not None:
        assert message in payload["error"]
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("#26E526", (38, 229, 38)),
        ("#26e526", (38, 229, 38)),
        ("38,229,38", (38, 229, 38)),
        (" 38, 229, 38 ", (38, 229, 38)),
    ],
)
def test_accepts_supported_color_syntax(
    value: str,
    expected: tuple[int, int, int],
) -> None:
    assert parse_color(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "#26E52",
        "#26E5267",
        "26E526",
        "#GGGGGG",
        "rgb(38,229,38)",
        "38,229",
        "38,229,38,1",
        "256,0,0",
        "-1,0,0",
        "+1,0,0",
        "a,b,c",
        "1, 2, ",
    ],
)
def test_rejects_malformed_colors(value: str) -> None:
    with pytest.raises(ValueError, match="Color"):
        parse_color(value)


def test_rejects_non_string_color() -> None:
    with pytest.raises(TypeError, match="string"):
        parse_color(cast(Any, 123))


def test_color_matching_uses_inclusive_chebyshev_boundary(
    tmp_path: Path,
) -> None:
    values = [
        (10, 20, 30),
        (0, 0, 0),
        (15, 20, 30),
        (0, 0, 0),
        (10, 25, 30),
        (0, 0, 0),
        (10, 20, 35),
        (0, 0, 0),
        (16, 20, 30),
    ]
    source = _make_image(
        tmp_path,
        "boundary.png",
        "RGB",
        [values],
    )

    result = _find(
        source,
        color="10,20,30",
        tolerance=5,
        min_area=1,
    )

    assert result["summary"]["matched_pixels"] == 4
    assert result["summary"]["component_count"] == 4
    assert [component["box"] for component in result["components"]] == [
        [0, 0, 1, 1],
        [2, 0, 3, 1],
        [4, 0, 5, 1],
        [6, 0, 7, 1],
    ]


def test_tolerance_zero_matches_only_exact_color(tmp_path: Path) -> None:
    source = _make_image(
        tmp_path,
        "exact.png",
        "RGB",
        [[(10, 20, 30), (11, 20, 30), (10, 21, 30)]],
    )

    result = _find(
        source,
        color="10,20,30",
        tolerance=0,
        min_area=1,
    )

    assert result["summary"]["matched_pixels"] == 1
    assert result["components"][0]["box"] == [0, 0, 1, 1]


def test_tolerance_255_matches_every_nontransparent_pixel(
    tmp_path: Path,
) -> None:
    source = _make_image(
        tmp_path,
        "opaque.png",
        "RGBA",
        [[
            (255, 0, 0, 0),
            (10, 20, 30, 128),
            (1, 2, 3, 255),
        ]],
    )

    result = _find(
        source,
        color="#FF0000",
        tolerance=255,
        min_area=1,
    )

    assert result["summary"]["matched_pixels"] == 2
    assert result["components"] == [
        {
            "box": [1, 0, 3, 1],
            "pixels": 2,
            "centroid": [1.5, 0.0],
        }
    ]


def test_partial_alpha_is_compared_without_compositing(
    tmp_path: Path,
) -> None:
    source = _make_image(
        tmp_path,
        "partial.png",
        "RGBA",
        [[(10, 20, 30, 128), (10, 20, 30, 0)]],
    )

    result = _find(
        source,
        color="10,20,30",
        tolerance=0,
        min_area=1,
    )

    assert result["summary"]["matched_pixels"] == 1
    assert result["components"][0]["box"] == [0, 0, 1, 1]


def test_palette_transparency_is_excluded(tmp_path: Path) -> None:
    source = _make_image(
        tmp_path,
        "palette.png",
        "P",
        [[0, 1, 2, 1]],
        transparency=0,
    )

    result = _find(
        source,
        color="255,0,0",
        tolerance=0,
        min_area=1,
    )

    assert result["summary"]["matched_pixels"] == 0
    assert result["summary"]["component_count"] == 0
    assert result["summary"]["retained_component_count"] == 0
    assert result["components"] == []


def test_opaque_palette_and_grayscale_are_converted(
    tmp_path: Path,
) -> None:
    palette_source = _make_image(
        tmp_path,
        "opaque-palette.png",
        "P",
        [[0, 1, 2]],
    )
    grayscale_source = _make_image(
        tmp_path,
        "grayscale.png",
        "L",
        [[200, 100, 0]],
    )

    palette_result = _find(
        palette_source,
        name="palette.json",
        color="255,0,0",
        tolerance=0,
        min_area=1,
    )
    grayscale_result = _find(
        grayscale_source,
        name="grayscale.json",
        color="200,200,200",
        tolerance=0,
        min_area=1,
    )

    assert palette_result["summary"]["matched_pixels"] == 1
    assert palette_result["components"][0]["box"] == [0, 0, 1, 1]
    assert grayscale_result["summary"]["matched_pixels"] == 1
    assert grayscale_result["components"][0]["box"] == [0, 0, 1, 1]


def test_rgb_trns_transparency_is_excluded(tmp_path: Path) -> None:
    source = _make_image(
        tmp_path,
        "trns.png",
        "RGB",
        [[(10, 20, 30), (40, 50, 60)]],
        transparency=(10, 20, 30),
    )

    result = _find(
        source,
        color="10,20,30",
        tolerance=0,
        min_area=1,
    )

    assert result["summary"]["matched_pixels"] == 0
    assert result["components"] == []


def test_single_pixel_contract(tmp_path: Path) -> None:
    source = _make_image(
        tmp_path,
        "single.png",
        "RGB",
        [
            [BACKGROUND, BACKGROUND, BACKGROUND],
            [BACKGROUND, BACKGROUND, TARGET],
        ],
    )

    result = _find(source, tolerance=0, min_area=1)

    assert result == {
        "status": "ok",
        "width": 3,
        "height": 2,
        "region": [0, 0, 3, 2],
        "color": [255, 0, 0],
        "tolerance": 0,
        "min_area": 1,
        "summary": {
            "region_pixels": 6,
            "matched_pixels": 1,
            "matched_percentage": 16.67,
            "component_count": 1,
            "retained_component_count": 1,
            "retained_pixels": 1,
        },
        "components": [
            {
                "box": [2, 1, 3, 2],
                "pixels": 1,
                "centroid": [2.0, 1.0],
            }
        ],
    }


def test_artifact_serializes_one_component_per_line(
    tmp_path: Path,
) -> None:
    source = _mask_image(
        tmp_path,
        [[True] * 5 + [False] * 3 + [True] * 3 + [False] * 2 + [True] * 2],
    )
    output = tmp_path / "lines.json"

    find_image(source, output, color="#FF0000", tolerance=0, min_area=1)

    text = output.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "{"
    start = lines.index('  "components": [')
    end = lines.index("  ]")
    component_lines = lines[start + 1 : end]
    assert len(component_lines) == 3

    components = [
        json.loads(line.removesuffix(",")) for line in component_lines
    ]
    for line in component_lines:
        assert line.startswith('    {"box":')

    artifact = json.loads(text)
    assert artifact["components"] == components
    assert [component["pixels"] for component in components] == [5, 3, 2]
    assert text.endswith("\n")


def test_artifact_without_matches_is_valid_json(
    tmp_path: Path,
) -> None:
    source = _mask_image(tmp_path, [[False, False, False]])
    output = tmp_path / "empty.json"

    find_image(source, output, color="#FF0000", tolerance=0)

    text = output.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert '  "components": []' in lines
    assert json.loads(text)["components"] == []
    assert text.endswith("\n")


def test_exact_statistics_for_l_shape(tmp_path: Path) -> None:
    source = _make_image(
        tmp_path,
        "l-shape.png",
        "RGB",
        [
            [BACKGROUND] * 5,
            [BACKGROUND, TARGET, TARGET, BACKGROUND, BACKGROUND],
            [BACKGROUND, TARGET, BACKGROUND, BACKGROUND, BACKGROUND],
            [BACKGROUND] * 5,
        ],
    )

    result = _find(source, tolerance=0, min_area=1)

    assert result["components"] == [
        {
            "box": [1, 1, 3, 3],
            "pixels": 3,
            "centroid": [1.33, 1.33],
        }
    ]


@pytest.mark.parametrize(
    "mask",
    [
        [
            [True, True, True],
            [False, False, False],
        ],
        [
            [True, False, False],
            [True, False, False],
        ],
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
        ],
        [
            [True, False, True],
            [False, False, False],
        ],
        [
            [True, True, False],
            [True, False, False],
            [False, False, True],
            [False, False, True],
        ],
    ],
    ids=["horizontal", "vertical", "diagonal", "disconnected", "irregular"],
)
def test_connectivity_matches_reference(
    tmp_path: Path,
    mask: list[list[bool]],
) -> None:
    source = _mask_image(tmp_path, mask)

    result = _find(source, tolerance=0, min_area=1)

    _assert_matches_reference(
        result,
        _reference(len(mask[0]), len(mask), mask, 1),
    )


def test_disconnected_components_with_overlapping_boxes(
    tmp_path: Path,
) -> None:
    mask = [[False] * 5 for _ in range(5)]
    for x in range(5):
        mask[0][x] = True
        mask[4][x] = True
    for y in range(5):
        mask[y][0] = True
        mask[y][4] = True
    mask[2][2] = True
    source = _mask_image(tmp_path, mask)

    result = _find(source, tolerance=0, min_area=1)

    assert result["summary"]["component_count"] == 2
    assert [component["pixels"] for component in result["components"]] == [
        16,
        1,
    ]
    assert result["components"][0]["box"] == [0, 0, 5, 5]
    assert result["components"][1]["box"] == [2, 2, 3, 3]


@pytest.mark.parametrize(
    ("width", "height", "tile", "seed", "density", "min_area"),
    [
        (9, 6, 5, 0, 0.35, 1),
        (9, 6, 5, 1, 0.65, 3),
        (9, 6, 4, 2, 0.9, 1),
        (6, 9, 5, 3, 0.35, 1),
        (6, 9, 5, 4, 0.65, 3),
        (6, 9, 4, 5, 0.9, 1),
        (12, 12, 7, 6, 0.5, 2),
        (12, 12, 3, 7, 0.8, 1),
    ],
)
def test_randomized_masks_match_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    width: int,
    height: int,
    tile: int,
    seed: int,
    density: float,
    min_area: int,
) -> None:
    import random

    generator = random.Random(seed)
    mask = [
        [generator.random() < density for _ in range(width)]
        for _ in range(height)
    ]
    source = _mask_image(tmp_path, mask)
    monkeypatch.setattr(find_module, "TILE_PIXELS", tile)

    result = _find(source, tolerance=0, min_area=min_area)

    _assert_matches_reference(
        result,
        _reference(width, height, mask, min_area),
    )


@pytest.mark.parametrize(
    ("width", "height", "tile", "seed"),
    [
        (9, 6, 4, 11),
        (6, 9, 4, 12),
        (12, 12, 5, 13),
    ],
)
def test_randomized_label_reuse_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    width: int,
    height: int,
    tile: int,
    seed: int,
) -> None:
    import random

    generator = random.Random(seed)
    mask = [
        [generator.random() < 0.5 for _ in range(width)]
        for _ in range(height)
    ]
    source = _mask_image(tmp_path, mask)
    monkeypatch.setattr(find_module, "TILE_PIXELS", tile)
    min_area = width * height + 1

    result = _find(source, tolerance=0, min_area=min_area)

    _assert_matches_reference(
        result,
        _reference(width, height, mask, min_area),
    )
    assert result["summary"]["retained_component_count"] == 0
    assert result["summary"]["retained_pixels"] == 0


def test_scan_directions_produce_equivalent_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mask = [
        [True, False, False, True, False, False, True],
        [True, True, False, False, True, False, False],
        [False, True, False, True, True, True, False],
        [False, False, True, False, True, False, True],
    ]
    horizontal_source = _mask_image(tmp_path, mask, "horizontal.png")
    vertical_source = _mask_image(
        tmp_path,
        _transpose(mask),
        "vertical.png",
    )
    monkeypatch.setattr(find_module, "TILE_PIXELS", 3)

    horizontal = _find(
        horizontal_source,
        name="horizontal.json",
        tolerance=0,
        min_area=1,
    )
    vertical = _find(
        vertical_source,
        name="vertical.json",
        tolerance=0,
        min_area=1,
    )

    assert horizontal["summary"] == vertical["summary"]
    mapped = [
        {
            "box": [
                component["box"][1],
                component["box"][0],
                component["box"][3],
                component["box"][2],
            ],
            "pixels": component["pixels"],
            "centroid": [
                component["centroid"][1],
                component["centroid"][0],
            ],
        }
        for component in vertical["components"]
    ]
    assert mapped == horizontal["components"]


def test_min_area_boundary_and_default(tmp_path: Path) -> None:
    mask = [[False] * 10 for _ in range(5)]
    for y in range(3):
        for x in range(3):
            mask[y][x] = True
    for y in range(2):
        for x in range(6, 10):
            mask[y][x] = True
    source = _mask_image(tmp_path, mask)

    default = _find(source, tolerance=0, name="default.json")
    all_components = _find(
        source,
        tolerance=0,
        min_area=1,
        name="all.json",
    )
    filtered = _find(
        source,
        tolerance=0,
        min_area=9,
        name="filtered.json",
    )

    assert default["summary"] == filtered["summary"]
    assert default["summary"]["matched_pixels"] == 17
    assert default["summary"]["component_count"] == 2
    assert default["summary"]["retained_component_count"] == 1
    assert default["summary"]["retained_pixels"] == 9
    assert [component["pixels"] for component in default["components"]] == [
        9
    ]
    assert all_components["summary"]["retained_component_count"] == 2
    assert all_components["summary"]["retained_pixels"] == 17


def test_raw_statistics_are_preserved_when_all_components_filtered(
    tmp_path: Path,
) -> None:
    source = _mask_image(
        tmp_path,
        [[True, False, True, False, True]],
    )

    result = _find(source, tolerance=0, min_area=2)

    assert result["summary"] == {
        "region_pixels": 5,
        "matched_pixels": 3,
        "matched_percentage": 60.0,
        "component_count": 3,
        "retained_component_count": 0,
        "retained_pixels": 0,
    }
    assert result["components"] == []


def test_returns_more_than_ten_components(tmp_path: Path) -> None:
    mask = [[index % 2 == 0 for index in range(25)]]
    source = _mask_image(tmp_path, mask)

    result = _find(source, tolerance=0, min_area=1)

    assert result["summary"]["retained_component_count"] == 13
    assert len(result["components"]) == 13


def test_components_sort_by_pixels_then_top_then_left(
    tmp_path: Path,
) -> None:
    mask = [[False] * 12 for _ in range(8)]
    for x in range(5):
        mask[0][x] = True
    for x in range(3):
        mask[0][7 + x] = True
    for x in range(3):
        mask[4][x] = True
    source = _mask_image(tmp_path, mask)

    result = _find(source, tolerance=0, min_area=1)

    assert [component["pixels"] for component in result["components"]] == [
        5,
        3,
        3,
    ]
    assert [component["box"] for component in result["components"]] == [
        [0, 0, 5, 1],
        [7, 0, 10, 1],
        [0, 4, 3, 5],
    ]


def test_region_restricts_matching_and_keeps_absolute_coordinates(
    tmp_path: Path,
) -> None:
    source = _make_image(
        tmp_path,
        "region.png",
        "RGB",
        [
            [TARGET, BACKGROUND, BACKGROUND, BACKGROUND, BACKGROUND, BACKGROUND],
            [BACKGROUND] * 6,
            [BACKGROUND] * 6,
            [BACKGROUND, BACKGROUND, BACKGROUND, TARGET, TARGET, BACKGROUND],
            [BACKGROUND, BACKGROUND, BACKGROUND, TARGET, TARGET, BACKGROUND],
            [BACKGROUND] * 6,
        ],
    )

    result = _find(
        source,
        tolerance=0,
        min_area=1,
        region=(3, 3, 5, 5),
    )

    assert result["region"] == [3, 3, 5, 5]
    assert result["summary"]["region_pixels"] == 4
    assert result["summary"]["matched_pixels"] == 4
    assert result["components"] == [
        {
            "box": [3, 3, 5, 5],
            "pixels": 4,
            "centroid": [3.5, 3.5],
        }
    ]


def test_region_uses_exclusive_right_and_bottom(tmp_path: Path) -> None:
    source = _make_image(
        tmp_path,
        "exclusive.png",
        "RGB",
        [
            [BACKGROUND] * 5,
            [BACKGROUND, TARGET, TARGET, BACKGROUND, BACKGROUND],
            [BACKGROUND, TARGET, TARGET, BACKGROUND, BACKGROUND],
            [BACKGROUND] * 5,
        ],
    )

    result = _find(
        source,
        tolerance=0,
        min_area=1,
        region=(2, 2, 3, 3),
    )

    assert result["summary"]["matched_pixels"] == 1
    assert result["components"][0]["box"] == [2, 2, 3, 3]


@pytest.mark.parametrize(
    ("region", "message"),
    [
        ((0, 0, 0, 1), "Region requires"),
        ((0, 0, 1, 0), "Region requires"),
        ((2, 0, 1, 1), "Region requires"),
        ((-1, 0, 1, 1), "negative"),
        ((0, -1, 1, 1), "negative"),
        ((0, 0, 5, 1), "exceeds image bounds"),
        ((0, 0, 1, 3), "exceeds image bounds"),
    ],
)
def test_rejects_invalid_regions(
    rgb_png: Path,
    region: tuple[int, int, int, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        find_image(
            rgb_png,
            rgb_png.parent / "out.json",
            color="#FF0000",
            region=region,
        )


@pytest.mark.parametrize(
    "region",
    [
        (0, 0, 1),
        (0, 0, 1, 1, 2),
        (0, 0, 1.0, 1),
        "0,0,1,1",
        (True, 0, 1, 1),
    ],
)
def test_rejects_malformed_regions(
    rgb_png: Path,
    region: Any,
) -> None:
    with pytest.raises(TypeError, match="four integers"):
        find_image(
            rgb_png,
            rgb_png.parent / "out.json",
            color="#FF0000",
            region=region,
        )


@pytest.mark.parametrize("tolerance", [-1, 256])
def test_rejects_tolerance_out_of_range(
    rgb_png: Path,
    tolerance: int,
) -> None:
    with pytest.raises(ValueError, match="between 0 and 255"):
        find_image(
            rgb_png,
            rgb_png.parent / "out.json",
            color="#FF0000",
            tolerance=tolerance,
        )


@pytest.mark.parametrize("tolerance", [1.5, True, "24"])
def test_rejects_non_integer_tolerance(
    rgb_png: Path,
    tolerance: Any,
) -> None:
    with pytest.raises(TypeError, match="integer"):
        find_image(
            rgb_png,
            rgb_png.parent / "out.json",
            color="#FF0000",
            tolerance=tolerance,
        )


@pytest.mark.parametrize("min_area", [0, -1])
def test_rejects_non_positive_min_area(
    rgb_png: Path,
    min_area: int,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        find_image(
            rgb_png,
            rgb_png.parent / "out.json",
            color="#FF0000",
            min_area=min_area,
        )


@pytest.mark.parametrize("min_area", [1.5, True, "9"])
def test_rejects_non_integer_min_area(
    rgb_png: Path,
    min_area: Any,
) -> None:
    with pytest.raises(TypeError, match="integer"):
        find_image(
            rgb_png,
            rgb_png.parent / "out.json",
            color="#FF0000",
            min_area=min_area,
        )


def test_label_slots_are_reused_after_filtered_finalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mask = [[False] * 5 for _ in range(6)]
    for x in (0, 2, 4):
        mask[0][x] = True
        mask[2][x] = True
    for y in (4, 5):
        for x in (0, 1):
            mask[y][x] = True
    source = _mask_image(tmp_path, mask)
    monkeypatch.setattr(find_module, "MAX_COMPONENT_LABELS", 3)

    result = _find(source, tolerance=0, min_area=4)

    _assert_matches_reference(
        result,
        _reference(5, 6, mask, 4),
    )


def test_repeated_merges_preserve_component_statistics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mask = [
        [True, False, False, False, True],
        [True, True, True, True, True],
        [True, False, False, False, True],
        [True, True, True, True, True],
    ]
    source = _mask_image(tmp_path, mask)
    monkeypatch.setattr(find_module, "MAX_COMPONENT_LABELS", 2)

    result = _find(source, tolerance=0, min_area=1)

    _assert_matches_reference(
        result,
        _reference(5, 4, mask, 1),
    )


def test_empty_scanlines_split_components(
    tmp_path: Path,
) -> None:
    mask = [
        [True, True, False],
        [False, False, False],
        [True, True, False],
    ]
    source = _mask_image(tmp_path, mask)

    result = _find(source, tolerance=0, min_area=1)

    _assert_matches_reference(result, _reference(3, 3, mask, 1))
    assert result["summary"]["component_count"] == 2


@pytest.mark.parametrize(
    ("width", "height"),
    [(6, 4), (4, 6)],
    ids=["horizontal", "vertical"],
)
def test_merged_component_frees_only_its_root_slot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    width: int,
    height: int,
) -> None:
    mask = [
        [True, False, True],
        [True, True, True],
        [False, False, False],
        [True, False, True],
        [True, True, True],
        [False, False, False],
    ]
    if width < height:
        mask = _transpose(mask)
    source = _mask_image(tmp_path, mask)
    monkeypatch.setattr(find_module, "MAX_COMPONENT_LABELS", 5)
    monkeypatch.setattr(find_module, "TILE_PIXELS", 3)

    result = _find(source, tolerance=0, min_area=100)

    _assert_matches_reference(
        result,
        _reference(len(mask[0]), len(mask), mask, 100),
    )


@pytest.mark.parametrize(
    "vertical",
    [False, True],
    ids=["horizontal", "vertical"],
)
def test_label_slots_are_reused_across_disconnected_components(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    vertical: bool,
) -> None:
    if vertical:
        mask = [[False] * 5 for _ in range(7)]
        for row, column in ((0, 0), (1, 2), (2, 4)):
            mask[row][column] = True
    else:
        mask = [[False] * 5 for _ in range(5)]
        for row, column in ((0, 0), (2, 1), (4, 4)):
            mask[row][column] = True
    source = _mask_image(tmp_path, mask)
    monkeypatch.setattr(find_module, "MAX_COMPONENT_LABELS", 1)
    monkeypatch.setattr(find_module, "TILE_PIXELS", 2)

    result = _find(source, tolerance=0, min_area=2)

    _assert_matches_reference(
        result,
        _reference(len(mask[0]), len(mask), mask, 2),
    )
    assert result["summary"]["component_count"] == 3


def test_component_label_limit_failure_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    source = _mask_image(
        tmp_path,
        [[True, False, True, False, True, False, True]],
    )
    output = tmp_path / "out.json"
    monkeypatch.setattr(find_module, "MAX_COMPONENT_LABELS", 2)

    with pytest.raises(ValueError, match="Component label limit"):
        find_image(
            source,
            output,
            color="#FF0000",
            tolerance=0,
            min_area=1,
        )

    assert not output.exists()
    assert_no_leftovers(tmp_path)


def test_source_file_size_limit_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
    padded_png: Path,
    tmp_path: Path,
) -> None:
    limit = padded_png.stat().st_size
    monkeypatch.setattr(find_module, "MAX_FILE_SIZE", limit)

    accepted = tmp_path / "accepted.json"
    find_image(padded_png, accepted, color="#000000")
    assert accepted.is_file()

    monkeypatch.setattr(find_module, "MAX_FILE_SIZE", limit - 1)
    rejected = tmp_path / "rejected.json"
    with pytest.raises(ValueError, match="between 1 byte"):
        find_image(padded_png, rejected, color="#000000")
    assert not rejected.exists()


def test_source_pixel_limit_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(find_module, "MAX_PIXELS", 8)

    accepted = tmp_path / "accepted.json"
    find_image(rgb_png, accepted, color="#000000")
    assert accepted.is_file()

    monkeypatch.setattr(find_module, "MAX_PIXELS", 7)
    rejected = tmp_path / "rejected.json"
    with pytest.raises(ValueError, match="pixel"):
        find_image(rgb_png, rejected, color="#000000")
    assert not rejected.exists()


@pytest.mark.parametrize(
    ("width", "height"),
    [(10_000, 10_000), (20_000, 20_000)],
    ids=["warning", "error"],
)
def test_rejects_decompression_bombs(
    tmp_path: Path,
    width: int,
    height: int,
) -> None:
    source = tmp_path / f"bomb_{width}.png"
    source.write_bytes(_png_header(width, height))

    with pytest.raises(ValueError, match="Invalid or unsafe image"):
        find_image(
            source,
            tmp_path / "out.json",
            color="#FF0000",
        )


def test_rejects_real_pixel_limit_before_decoding(
    tmp_path: Path,
) -> None:
    source = tmp_path / "too_many_pixels.png"
    source.write_bytes(_png_header(8_000, 8_000))

    with pytest.raises(ValueError, match="pixel"):
        find_image(
            source,
            tmp_path / "out.json",
            color="#FF0000",
        )


def test_rejects_empty_source(empty_file: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="between 1 byte"):
        find_image(
            empty_file,
            tmp_path / "out.json",
            color="#FF0000",
        )


def test_rejects_corrupted_source(
    garbage_file: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Invalid or unsafe image"):
        find_image(
            garbage_file,
            tmp_path / "out.json",
            color="#FF0000",
        )


def test_rejects_multi_frame_source(
    animated_gif: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Multi-frame"):
        find_image(
            animated_gif,
            tmp_path / "out.json",
            color="#FF0000",
        )


def test_output_size_limit_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    mask = [[index % 2 == 0 for index in range(25)]]
    source = _mask_image(tmp_path, mask)

    measured = tmp_path / "measured.json"
    find_image(source, measured, color="#FF0000", tolerance=0, min_area=1)
    measured_size = measured.stat().st_size

    monkeypatch.setattr(find_module, "MAX_OUTPUT_SIZE", measured_size)
    accepted = tmp_path / "accepted.json"
    find_image(source, accepted, color="#FF0000", tolerance=0, min_area=1)
    assert accepted.stat().st_size == measured_size

    monkeypatch.setattr(
        find_module,
        "MAX_OUTPUT_SIZE",
        measured_size - 1,
    )
    rejected = tmp_path / "rejected.json"
    with pytest.raises(ValueError, match="Output exceeds"):
        find_image(
            source,
            rejected,
            color="#FF0000",
            tolerance=0,
            min_area=1,
        )
    assert not rejected.exists()
    assert_no_leftovers(tmp_path)


def test_rejects_same_source_and_output(rgb_png: Path) -> None:
    with pytest.raises(ValueError, match="must be different"):
        find_image(rgb_png, rgb_png, color="#FF0000")


def test_rejects_unsupported_output_extension(
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=r"\.json"):
        find_image(
            rgb_png,
            tmp_path / "out.png",
            color="#FF0000",
        )


def test_rejects_existing_output(
    rgb_png: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    output = tmp_path / "exists.json"
    output.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError, match="already exists"):
        find_image(rgb_png, output, color="#FF0000")

    assert output.read_bytes() == b"sentinel"
    assert_no_leftovers(tmp_path)


def test_rejects_dangling_symlink_output(
    rgb_png: Path,
    tmp_path: Path,
    symlink_or_skip: Callable[[Path, Path], None],
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    output = tmp_path / "dangling.json"
    missing = tmp_path / "missing.json"
    symlink_or_skip(output, missing)

    with pytest.raises(FileExistsError, match="already exists"):
        find_image(rgb_png, output, color="#FF0000")

    assert output.is_symlink()
    assert not output.exists()
    assert not missing.exists()
    assert_no_leftovers(tmp_path)


def test_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_image(
            tmp_path / "absent.png",
            tmp_path / "out.json",
            color="#FF0000",
        )


def test_rejects_directory_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Not a regular file"):
        find_image(
            tmp_path,
            tmp_path / "out.json",
            color="#FF0000",
        )


def test_source_is_unchanged_and_no_artifacts_are_left(
    rgb_png: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    before = _digest(rgb_png)
    entries = _snapshot(tmp_path)

    output = tmp_path / "result.json"
    find_image(rgb_png, output, color="#000000")

    assert _digest(rgb_png) == before
    assert _snapshot(tmp_path) == entries | {output.relative_to(tmp_path)}
    assert_no_leftovers(tmp_path)


def test_publication_never_overwrites_racing_file(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    output = tmp_path / "race.json"
    real_link = find_module.os.link

    def racing_link(source: Any, destination: Any) -> None:
        if Path(destination) == output:
            output.write_bytes(b"competitor")
        real_link(source, destination)

    monkeypatch.setattr(
        find_module,
        "os",
        SimpleNamespace(link=racing_link),
    )

    with pytest.raises(FileExistsError):
        find_image(rgb_png, output, color="#FF0000")

    assert output.read_bytes() == b"competitor"
    assert_no_leftovers(tmp_path)


def test_publication_becomes_visible_only_after_completion(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    output = tmp_path / "result.json"
    seen: list[Path] = []
    real_link = find_module.os.link

    def checking_link(source: Any, destination: Any) -> None:
        destination_path = Path(destination)
        assert not destination_path.exists()
        assert not destination_path.is_symlink()
        seen.append(destination_path)
        real_link(source, destination)

    monkeypatch.setattr(find_module.os, "link", checking_link)

    find_image(rgb_png, output, color="#FF0000")

    assert seen == [output]
    assert output.is_file()
    assert_no_leftovers(tmp_path)


def test_unsupported_hardlink_fails_without_publishing(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    output = tmp_path / "result.json"

    def unsupported_link(source: Any, destination: Any) -> None:
        raise OSError(errno.EOPNOTSUPP, "Hard links are not supported")

    monkeypatch.setattr(find_module.os, "link", unsupported_link)

    with pytest.raises(OSError, match="not supported"):
        find_image(rgb_png, output, color="#FF0000")

    assert not output.exists()
    assert_no_leftovers(tmp_path)


def test_cli_success_writes_compact_response_and_artifact(
    run_cli: Callable[..., Any],
    tmp_path: Path,
) -> None:
    source = _make_image(
        tmp_path,
        "cli.png",
        "RGB",
        [[TARGET, BACKGROUND, TARGET]],
    )
    output = tmp_path / "artifact.json"

    result = run_cli(
        "image",
        "find",
        str(source),
        str(output),
        "--color",
        "#FF0000",
        "--tolerance",
        "0",
        "--min-area",
        "1",
    )

    assert result.returncode == 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload == {"status": "ok", "output": str(output)}
    assert "\n" not in result.stdout.strip()
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["summary"]["retained_component_count"] == 2


def test_cli_rejects_malformed_color_as_argument_error(
    run_cli: Callable[..., Any],
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    result = run_cli(
        "image",
        "find",
        str(rgb_png),
        str(tmp_path / "out.json"),
        "--color",
        "pink",
    )

    _assert_cli_error(result, code=2, error_type="CLIArgumentError")


def test_cli_rejects_tolerance_out_of_range(
    run_cli: Callable[..., Any],
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    result = run_cli(
        "image",
        "find",
        str(rgb_png),
        str(tmp_path / "out.json"),
        "--color",
        "#FF0000",
        "--tolerance",
        "256",
    )

    _assert_cli_error(
        result,
        code=1,
        error_type="ValueError",
        message="between 0 and 255",
    )


def test_cli_rejects_non_positive_min_area(
    run_cli: Callable[..., Any],
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    result = run_cli(
        "image",
        "find",
        str(rgb_png),
        str(tmp_path / "out.json"),
        "--color",
        "#FF0000",
        "--min-area",
        "0",
    )

    _assert_cli_error(
        result,
        code=1,
        error_type="ValueError",
        message="positive integer",
    )


def test_cli_rejects_malformed_region_syntax(
    run_cli: Callable[..., Any],
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    result = run_cli(
        "image",
        "find",
        str(rgb_png),
        str(tmp_path / "out.json"),
        "--color",
        "#FF0000",
        "--region",
        "0,0,1",
    )

    _assert_cli_error(result, code=2, error_type="CLIArgumentError")


def test_cli_rejects_invalid_region_geometry(
    run_cli: Callable[..., Any],
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    result = run_cli(
        "image",
        "find",
        str(rgb_png),
        str(tmp_path / "out.json"),
        "--color",
        "#FF0000",
        "--region",
        "0,0,0,1",
    )

    _assert_cli_error(
        result,
        code=1,
        error_type="ValueError",
        message="Region requires",
    )


def test_cli_missing_source_is_json_error(
    run_cli: Callable[..., Any],
    tmp_path: Path,
) -> None:
    result = run_cli(
        "image",
        "find",
        str(tmp_path / "absent.png"),
        str(tmp_path / "out.json"),
        "--color",
        "#FF0000",
    )

    _assert_cli_error(result, code=1, error_type="FileNotFoundError")


def test_cli_existing_output_is_json_error(
    run_cli: Callable[..., Any],
    rgb_png: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "exists.json"
    output.write_bytes(b"sentinel")

    result = run_cli(
        "image",
        "find",
        str(rgb_png),
        str(output),
        "--color",
        "#FF0000",
    )

    _assert_cli_error(result, code=1, error_type="FileExistsError")
    assert output.read_bytes() == b"sentinel"


def test_cli_help_exposes_find(run_cli: Callable[..., Any]) -> None:
    result = run_cli("image", "find", "--help")

    assert result.returncode == 0
    assert result.stderr == ""
    assert "usage:" in result.stdout
    assert "--color" in result.stdout
    assert "--tolerance" in result.stdout
    assert "--region" in result.stdout
    assert "--min-area" in result.stdout

    listing = run_cli("image", "--help")
    assert listing.returncode == 0
    assert "find" in listing.stdout
