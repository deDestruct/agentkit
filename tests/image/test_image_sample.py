from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image

from agentkit.image import sample as sample_module
from agentkit.image.sample import sample_image


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


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(path: Path) -> set[Path]:
    return {entry.relative_to(path) for entry in path.rglob("*")}


def _chunk(tag: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", checksum)


def _png_header(width: int, height: int) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(b""))
        + _chunk(b"IEND", b"")
    )


def _gradient_grid(tmp_path: Path, size: int, step: int) -> Path:
    return _make_image(
        tmp_path,
        "grid.png",
        "RGB",
        [
            [(x * step, y * step, 0) for x in range(size)]
            for y in range(size)
        ],
    )


def _assert_inclusive_limit(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    attribute: str,
    limit: int,
    match: str,
) -> None:
    monkeypatch.setattr(sample_module, attribute, limit)
    assert sample_image(source, points=[(0, 0)])["status"] == "ok"

    monkeypatch.setattr(sample_module, attribute, limit - 1)
    with pytest.raises(ValueError, match=match):
        sample_image(source, points=[(0, 0)])


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


def test_points_return_exact_rgb(rgb_png: Path) -> None:
    result = sample_image(rgb_png, points=[(3, 0), (0, 1), (3, 1)])

    assert result == {
        "status": "ok",
        "points": [
            {"x": 3, "y": 0, "rgba": [140, 30, 200, 255]},
            {"x": 0, "y": 1, "rgba": [20, 120, 200, 255]},
            {"x": 3, "y": 1, "rgba": [140, 120, 200, 255]},
        ],
    }


def test_points_return_exact_rgba(tmp_path: Path) -> None:
    source = _make_image(
        tmp_path,
        "rgba.png",
        "RGBA",
        [[(1, 2, 3, 0), (4, 5, 6, 128), (7, 8, 9, 255)]],
    )

    result = sample_image(source, points=[(0, 0), (1, 0), (2, 0)])

    assert [point["rgba"] for point in result["points"]] == [
        [1, 2, 3, 0],
        [4, 5, 6, 128],
        [7, 8, 9, 255],
    ]


@pytest.mark.parametrize(
    ("mode", "pixels", "expected"),
    [
        (
            "L",
            [[0, 128, 255]],
            [[0, 0, 0, 255], [128, 128, 128, 255], [255, 255, 255, 255]],
        ),
        (
            "1",
            [[0, 255]],
            [[0, 0, 0, 255], [255, 255, 255, 255]],
        ),
        (
            "LA",
            [[(10, 0), (20, 128)]],
            [[10, 10, 10, 0], [20, 20, 20, 128]],
        ),
    ],
)
def test_points_normalize_supported_modes(
    tmp_path: Path,
    mode: str,
    pixels: list[list[Any]],
    expected: list[list[int]],
) -> None:
    source = _make_image(tmp_path, f"{mode}.png", mode, pixels)
    points = [(index, 0) for index in range(len(expected))]

    result = sample_image(source, points=points)

    assert [point["rgba"] for point in result["points"]] == expected


def test_points_normalize_palette(opaque_palette_png: Path) -> None:
    result = sample_image(
        opaque_palette_png,
        points=[(0, 0), (1, 0), (0, 1), (1, 1)],
    )

    assert [point["rgba"] for point in result["points"]] == [
        [255, 255, 255, 255],
        [0, 0, 0, 255],
        [0, 0, 0, 255],
        [255, 255, 255, 255],
    ]


def test_points_normalize_palette_transparency(palette_png: Path) -> None:
    result = sample_image(
        palette_png,
        points=[(0, 0), (1, 0), (1, 1)],
    )

    assert [point["rgba"] for point in result["points"]] == [
        [255, 255, 255, 0],
        [0, 0, 0, 255],
        [255, 255, 255, 0],
    ]


def test_points_preserve_order_and_duplicates(rgb_png: Path) -> None:
    result = sample_image(rgb_png, points=[(3, 0), (0, 1), (3, 0)])

    assert [(point["x"], point["y"]) for point in result["points"]] == [
        (3, 0),
        (0, 1),
        (3, 0),
    ]
    assert result["points"][0]["rgba"] == result["points"][2]["rgba"]
    assert result["points"][0]["rgba"] == [140, 30, 200, 255]


@pytest.mark.parametrize(
    ("count", "message"),
    [
        (10, None),
        (11, "between 1 and 10"),
        (0, "between 1 and 10"),
    ],
    ids=["exactly-ten", "more-than-ten", "empty"],
)
def test_point_count_limits(
    rgb_png: Path,
    count: int,
    message: str | None,
) -> None:
    points = [(index % 4, (index // 4) % 2) for index in range(count)]

    if message is None:
        assert len(sample_image(rgb_png, points=points)["points"]) == 10
    else:
        with pytest.raises(ValueError, match=message):
            sample_image(rgb_png, points=points)


@pytest.mark.parametrize(
    "points",
    [
        (0, 0),
        [(0,)],
        [(0, 0, 0)],
        [("0", "0")],
        [0, 0],
        [(0, 0), None],
        "0,0",
    ],
)
def test_rejects_malformed_point_entries(
    rgb_png: Path,
    points: Any,
) -> None:
    with pytest.raises(TypeError, match=r"two integers|coordinate pairs"):
        sample_image(rgb_png, points=points)


def test_rejects_bool_point_and_region_coordinates(rgb_png: Path) -> None:
    with pytest.raises(TypeError, match="two integers"):
        sample_image(rgb_png, points=cast(Any, [(True, 0)]))

    with pytest.raises(TypeError, match="four integers"):
        sample_image(rgb_png, region=cast(Any, (False, 0, 1, 1)))


@pytest.mark.parametrize(
    ("point", "message"),
    [
        ((-1, 0), "negative"),
        ((0, -1), "negative"),
        ((4, 0), "outside image bounds"),
        ((0, 2), "outside image bounds"),
    ],
)
def test_rejects_negative_and_out_of_bounds_points(
    rgb_png: Path,
    point: tuple[int, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        sample_image(rgb_png, points=[point])


def test_neighbors_exact_matrix_and_center(tmp_path: Path) -> None:
    source = _gradient_grid(tmp_path, 5, 10)

    result = sample_image(source, points=[(2, 2)], neighbors=True)

    point = result["points"][0]
    assert point["rgba"] == [20, 20, 0, 255]
    assert point["neighbors"] == [
        ["#0A0A00FF", "#140A00FF", "#1E0A00FF"],
        ["#0A1400FF", "#141400FF", "#1E1400FF"],
        ["#0A1E00FF", "#141E00FF", "#1E1E00FF"],
    ]


def test_neighbors_corner_and_edge_nulls(tmp_path: Path) -> None:
    source = _gradient_grid(tmp_path, 3, 16)

    result = sample_image(
        source,
        points=[(0, 0), (2, 0), (0, 2), (2, 2), (1, 0), (0, 1)],
        neighbors=True,
    )
    entries = result["points"]

    assert entries[0]["neighbors"] == [
        [None, None, None],
        [None, "#000000FF", "#100000FF"],
        [None, "#001000FF", "#101000FF"],
    ]
    assert entries[1]["neighbors"] == [
        [None, None, None],
        ["#100000FF", "#200000FF", None],
        ["#101000FF", "#201000FF", None],
    ]
    assert entries[2]["neighbors"] == [
        [None, "#001000FF", "#101000FF"],
        [None, "#002000FF", "#102000FF"],
        [None, None, None],
    ]
    assert entries[3]["neighbors"] == [
        ["#101000FF", "#201000FF", None],
        ["#102000FF", "#202000FF", None],
        [None, None, None],
    ]
    assert sum(value is None for row in entries[4]["neighbors"] for value in row) == 3
    assert sum(value is None for row in entries[5]["neighbors"] for value in row) == 3


def test_neighbors_single_pixel_image(tmp_path: Path) -> None:
    source = _make_image(
        tmp_path,
        "single.png",
        "RGBA",
        [[(1, 2, 3, 4)]],
    )

    point = sample_image(source, points=[(0, 0)], neighbors=True)["points"][0]

    assert point["rgba"] == [1, 2, 3, 4]
    assert point["neighbors"] == [
        [None, None, None],
        [None, "#01020304", None],
        [None, None, None],
    ]


def test_neighbors_preserve_alpha(tmp_path: Path) -> None:
    source = _make_image(
        tmp_path,
        "alpha.png",
        "RGBA",
        [[(10, 20, 30, 40)] * 3 for _ in range(3)],
    )

    point = sample_image(source, points=[(1, 1)], neighbors=True)["points"][0]

    assert point["neighbors"] == [
        ["#0A141E28", "#0A141E28", "#0A141E28"],
        ["#0A141E28", "#0A141E28", "#0A141E28"],
        ["#0A141E28", "#0A141E28", "#0A141E28"],
    ]


def test_neighbors_absent_when_not_requested(rgb_png: Path) -> None:
    point = sample_image(rgb_png, points=[(1, 1)])["points"][0]

    assert "neighbors" not in point


def test_neighbors_reuse_converted_neighborhood_for_center(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("center must come from the neighborhood")

    monkeypatch.setattr(sample_module, "_rgba_at", fail)

    result = sample_image(rgb_png, points=[(0, 0)], neighbors=True)

    assert result["points"][0]["rgba"] == [20, 30, 200, 255]


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"points": [(0, 0)], "region": (0, 0, 1, 1)},
    ],
    ids=["neither", "both"],
)
def test_rejects_wrong_mode_combinations(
    rgb_png: Path,
    options: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="Exactly one"):
        sample_image(rgb_png, **options)


def test_rejects_neighbors_with_region_in_domain(rgb_png: Path) -> None:
    with pytest.raises(
        ValueError,
        match="Neighbors can only be used with points",
    ):
        sample_image(
            rgb_png,
            region=(0, 0, 1, 1),
            neighbors=True,
        )


def test_rejects_non_boolean_neighbors(rgb_png: Path) -> None:
    with pytest.raises(TypeError, match="boolean"):
        sample_image(
            rgb_png,
            points=[(0, 0)],
            neighbors=cast(Any, 1),
        )


def test_ten_neighborhoods_have_ninety_positions(tmp_path: Path) -> None:
    source = _gradient_grid(tmp_path, 5, 10)
    points = [
        (1, 1),
        (2, 1),
        (3, 1),
        (1, 2),
        (2, 2),
        (3, 2),
        (1, 3),
        (2, 3),
        (3, 3),
        (1, 1),
    ]

    result = sample_image(source, points=points, neighbors=True)

    assert len(result["points"]) == 10
    for point in result["points"]:
        matrix = point["neighbors"]
        assert len(matrix) == 3
        assert all(len(row) == 3 for row in matrix)
        assert all(value is not None for row in matrix for value in row)
    assert sum(
        len(row)
        for point in result["points"]
        for row in point["neighbors"]
    ) == 90


def test_region_quantization_bucket_assignment(tmp_path: Path) -> None:
    values = [0, 15, 16, 31, 32, 47, 48, 255]
    source = _make_image(
        tmp_path,
        "buckets.png",
        "RGB",
        [[(value, 0, 0) for value in values]],
    )

    result = sample_image(source, region=(0, 0, 8, 1))

    assert result == {
        "status": "ok",
        "region": [0, 0, 8, 1],
        "pixel_count": 8,
        "analyzed_pixel_count": 8,
        "quantization_step": 16,
        "color_group_count": 5,
        "coverage_percentage": 100.0,
        "colors": [
            {"rgb": [8, 8, 8], "count": 2, "percentage": 25.0},
            {"rgb": [24, 8, 8], "count": 2, "percentage": 25.0},
            {"rgb": [40, 8, 8], "count": 2, "percentage": 25.0},
            {"rgb": [56, 8, 8], "count": 1, "percentage": 12.5},
            {"rgb": [248, 8, 8], "count": 1, "percentage": 12.5},
        ],
    }


def test_region_exact_groups_percentages_and_order(tmp_path: Path) -> None:
    source = _make_image(
        tmp_path,
        "groups.png",
        "RGB",
        [[(0, 0, 0)] * 5 + [(255, 0, 0)] * 3 + [(0, 255, 0)] * 2],
    )

    result = sample_image(source, region=(0, 0, 10, 1))

    assert result["pixel_count"] == 10
    assert result["analyzed_pixel_count"] == 10
    assert result["color_group_count"] == 3
    assert result["coverage_percentage"] == 100.0
    assert result["colors"] == [
        {"rgb": [8, 8, 8], "count": 5, "percentage": 50.0},
        {"rgb": [248, 8, 8], "count": 3, "percentage": 30.0},
        {"rgb": [8, 248, 8], "count": 2, "percentage": 20.0},
    ]


def test_region_ties_sort_by_representative_rgb(tmp_path: Path) -> None:
    source = _make_image(
        tmp_path,
        "ties.png",
        "RGB",
        [[(255, 0, 0)] * 2 + [(0, 255, 0)] * 2],
    )

    result = sample_image(source, region=(0, 0, 4, 1))

    assert result["colors"] == [
        {"rgb": [8, 248, 8], "count": 2, "percentage": 50.0},
        {"rgb": [248, 8, 8], "count": 2, "percentage": 50.0},
    ]


def test_region_returns_at_most_ten_groups(tmp_path: Path) -> None:
    values = [value * 16 for value in range(12)]
    source = _make_image(
        tmp_path,
        "many.png",
        "RGB",
        [[(value, 0, 0) for value in values]],
    )

    result = sample_image(source, region=(0, 0, 12, 1))

    assert result["pixel_count"] == 12
    assert result["analyzed_pixel_count"] == 12
    assert result["color_group_count"] == 12
    assert result["coverage_percentage"] == 83.33
    assert len(result["colors"]) == 10
    assert result["colors"][0] == {
        "rgb": [8, 8, 8],
        "count": 1,
        "percentage": 8.33,
    }
    assert result["colors"][-1] == {
        "rgb": [152, 8, 8],
        "count": 1,
        "percentage": 8.33,
    }


def test_region_excludes_fully_transparent_pixels(tmp_path: Path) -> None:
    source = _make_image(
        tmp_path,
        "alpha.png",
        "RGBA",
        [[
            (10, 20, 30, 0),
            (40, 50, 60, 255),
            (70, 80, 90, 128),
            (100, 110, 120, 0),
        ]],
    )

    result = sample_image(source, region=(0, 0, 4, 1))

    assert result["pixel_count"] == 4
    assert result["analyzed_pixel_count"] == 2
    assert result["color_group_count"] == 2
    assert result["coverage_percentage"] == 100.0
    assert result["colors"] == [
        {"rgb": [40, 56, 56], "count": 1, "percentage": 50.0},
        {"rgb": [72, 88, 88], "count": 1, "percentage": 50.0},
    ]


def test_region_fully_transparent(tmp_path: Path) -> None:
    source = _make_image(
        tmp_path,
        "transparent.png",
        "RGBA",
        [
            [(10, 20, 30, 0), (40, 50, 60, 0)],
            [(70, 80, 90, 0), (100, 110, 120, 0)],
        ],
    )

    result = sample_image(source, region=(0, 0, 2, 2))

    assert result["pixel_count"] == 4
    assert result["analyzed_pixel_count"] == 0
    assert result["color_group_count"] == 0
    assert result["coverage_percentage"] == 0.0
    assert result["colors"] == []


def test_region_palette_png_transparency(tmp_path: Path) -> None:
    source = _make_image(
        tmp_path,
        "palette.png",
        "P",
        [[0, 1, 2, 1]],
        transparency=0,
    )

    result = sample_image(source, region=(0, 0, 4, 1))

    assert result["pixel_count"] == 4
    assert result["analyzed_pixel_count"] == 3
    assert result["color_group_count"] == 2
    assert result["coverage_percentage"] == 100.0
    assert result["colors"] == [
        {"rgb": [8, 248, 8], "count": 2, "percentage": 66.67},
        {"rgb": [8, 8, 248], "count": 1, "percentage": 33.33},
    ]


def test_region_rgb_png_trns_transparency(tmp_path: Path) -> None:
    source = _make_image(
        tmp_path,
        "trns.png",
        "RGB",
        [[
            (10, 20, 30),
            (40, 50, 60),
            (70, 80, 90),
            (10, 20, 30),
        ]],
        transparency=(10, 20, 30),
    )

    result = sample_image(source, region=(0, 0, 4, 1))

    assert result["pixel_count"] == 4
    assert result["analyzed_pixel_count"] == 2
    assert result["color_group_count"] == 2
    assert result["coverage_percentage"] == 100.0
    assert result["colors"] == [
        {"rgb": [40, 56, 56], "count": 1, "percentage": 50.0},
        {"rgb": [72, 88, 88], "count": 1, "percentage": 50.0},
    ]


def test_region_uses_exclusive_right_and_bottom(tmp_path: Path) -> None:
    pixels = [[(0, 255, 0)] * 4 for _ in range(3)]
    pixels[1][1] = (255, 0, 0)
    pixels[1][2] = (255, 0, 0)
    source = _make_image(tmp_path, "exclusive.png", "RGB", pixels)

    result = sample_image(source, region=(1, 1, 3, 2))

    assert result["pixel_count"] == 2
    assert result["analyzed_pixel_count"] == 2
    assert result["color_group_count"] == 1
    assert result["coverage_percentage"] == 100.0
    assert result["colors"] == [
        {"rgb": [248, 8, 8], "count": 2, "percentage": 100.0},
    ]


@pytest.mark.parametrize(
    ("region", "message"),
    [
        ((0, 0, 0, 1), "Region requires"),
        ((0, 0, 1, 0), "Region requires"),
        ((2, 0, 1, 1), "Region requires"),
        ((0, 2, 1, 1), "Region requires"),
        ((-1, 0, 1, 1), "negative"),
        ((0, -1, 1, 1), "negative"),
        ((0, 0, 5, 1), "exceeds image bounds"),
        ((0, 0, 1, 3), "exceeds image bounds"),
    ],
)
def test_region_rejects_invalid_geometry(
    rgb_png: Path,
    region: tuple[int, int, int, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        sample_image(rgb_png, region=region)


@pytest.mark.parametrize(
    "region",
    [
        (0, 0, 1),
        (0, 0, 1, 1, 2),
        (0, 0, 1.0, 1),
        "0,0,1,1",
    ],
)
def test_region_rejects_malformed_values(
    rgb_png: Path,
    region: Any,
) -> None:
    with pytest.raises(TypeError, match="four integers"):
        sample_image(rgb_png, region=region)


@pytest.mark.parametrize(
    ("size", "pixels"),
    [
        ((1, 10), [[(value * 16, 0, 0)] for value in range(10)]),
        ((10, 1), [[(value * 16, 0, 0) for value in range(10)]]),
    ],
    ids=["narrow", "wide"],
)
def test_region_tiles_cover_extreme_aspect_ratios(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    size: tuple[int, int],
    pixels: list[list[tuple[int, int, int]]],
) -> None:
    monkeypatch.setattr(sample_module, "TILE_PIXELS", 4)
    source = _make_image(tmp_path, "tiles.png", "RGB", pixels)

    result = sample_image(source, region=(0, 0, *size))

    assert result["pixel_count"] == 10
    assert result["analyzed_pixel_count"] == 10
    assert result["color_group_count"] == 10
    assert result["coverage_percentage"] == 100.0
    assert len(result["colors"]) == 10
    assert result["colors"][0]["rgb"] == [8, 8, 8]
    assert result["colors"][-1]["rgb"] == [152, 8, 8]


def test_region_tiles_cover_non_even_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sample_module, "TILE_PIXELS", 4)
    source = _gradient_grid(tmp_path, 5, 16)

    result = sample_image(source, region=(0, 0, 5, 5))

    assert result["pixel_count"] == 25
    assert result["analyzed_pixel_count"] == 25
    assert result["color_group_count"] == 25
    assert result["coverage_percentage"] == 40.0
    assert len(result["colors"]) == 10
    assert result["colors"][0]["rgb"] == [8, 8, 8]
    assert result["colors"][-1]["rgb"] == [24, 72, 8]


def test_region_counts_merge_across_tiles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sample_module, "TILE_PIXELS", 4)
    source = _make_image(
        tmp_path,
        "merged.png",
        "RGB",
        [[(10, 20, 30)] * 10],
    )

    result = sample_image(source, region=(0, 0, 10, 1))

    assert result["pixel_count"] == 10
    assert result["analyzed_pixel_count"] == 10
    assert result["color_group_count"] == 1
    assert result["coverage_percentage"] == 100.0
    assert result["colors"] == [
        {"rgb": [8, 24, 24], "count": 10, "percentage": 100.0},
    ]


def test_empty_source_is_rejected(empty_file: Path) -> None:
    with pytest.raises(ValueError, match="between 1 byte"):
        sample_image(empty_file, points=[(0, 0)])


def test_corrupted_source_is_rejected(garbage_file: Path) -> None:
    with pytest.raises(ValueError, match="Invalid or unsafe image"):
        sample_image(garbage_file, points=[(0, 0)])


def test_multi_frame_source_is_rejected(animated_gif: Path) -> None:
    with pytest.raises(ValueError, match="Multi-frame"):
        sample_image(animated_gif, points=[(0, 0)])


def test_source_file_size_limit_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
    padded_png: Path,
) -> None:
    _assert_inclusive_limit(
        monkeypatch,
        padded_png,
        "MAX_FILE_SIZE",
        padded_png.stat().st_size,
        "between 1 byte",
    )


def test_source_pixel_limit_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
    rgb_png: Path,
) -> None:
    _assert_inclusive_limit(
        monkeypatch,
        rgb_png,
        "MAX_PIXELS",
        8,
        "pixel limit",
    )


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
        sample_image(source, points=[(0, 0)])


def test_rejects_real_pixel_limit_before_decoding(tmp_path: Path) -> None:
    source = tmp_path / "too_many_pixels.png"
    source.write_bytes(_png_header(8_000, 8_000))

    with pytest.raises(ValueError, match="pixel limit"):
        sample_image(source, points=[(0, 0)])


def test_source_is_unchanged_and_no_artifacts_are_created(
    tmp_path: Path,
    rgb_png: Path,
) -> None:
    before = _digest(rgb_png)
    entries = _snapshot(tmp_path)

    sample_image(rgb_png, points=[(0, 0), (1, 1)], neighbors=True)
    sample_image(rgb_png, region=(0, 0, 4, 2))

    assert _digest(rgb_png) == before
    assert _snapshot(tmp_path) == entries


def test_missing_source_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sample_image(
            tmp_path / "absent.png",
            points=[(0, 0)],
        )


def test_directory_source_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Not a regular file"):
        sample_image(tmp_path, points=[(0, 0)])


def test_cli_points_success_uses_compact_json(
    run_cli: Callable[..., Any],
    rgb_png: Path,
) -> None:
    result = run_cli(
        "image",
        "sample",
        str(rgb_png),
        "--points",
        "1,0;0,0",
    )

    assert result.returncode == 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert [(point["x"], point["y"]) for point in payload["points"]] == [
        (1, 0),
        (0, 0),
    ]
    assert payload["points"][0]["rgba"] == [60, 30, 200, 255]
    assert "\n" not in result.stdout.strip()
    assert result.stdout.strip() == json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def test_cli_region_success_uses_compact_json(
    run_cli: Callable[..., Any],
    rgb_png: Path,
) -> None:
    result = run_cli(
        "image",
        "sample",
        str(rgb_png),
        "--region",
        "0,0,4,2",
    )

    assert result.returncode == 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["region"] == [0, 0, 4, 2]
    assert payload["pixel_count"] == 8
    assert "\n" not in result.stdout.strip()
    assert result.stdout.strip() == json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def test_cli_neighbors_success(
    run_cli: Callable[..., Any],
    tmp_path: Path,
) -> None:
    source = _gradient_grid(tmp_path, 3, 10)

    result = run_cli(
        "image",
        "sample",
        str(source),
        "--points",
        "1,1",
        "--neighbors",
    )

    assert result.returncode == 0
    point = json.loads(result.stdout)["points"][0]
    assert point["rgba"] == [10, 10, 0, 255]
    assert point["neighbors"][1][1] == "#0A0A00FF"


@pytest.mark.parametrize(
    "extra",
    [
        (),
        ("--points", "0,0", "--region", "0,0,1,1"),
    ],
    ids=["neither", "both"],
)
def test_cli_requires_exactly_one_mode(
    run_cli: Callable[..., Any],
    rgb_png: Path,
    extra: tuple[str, ...],
) -> None:
    result = run_cli("image", "sample", str(rgb_png), *extra)

    _assert_cli_error(result, code=2, error_type="CLIArgumentError")


def test_cli_rejects_neighbors_with_region(
    run_cli: Callable[..., Any],
    rgb_png: Path,
) -> None:
    result = run_cli(
        "image",
        "sample",
        str(rgb_png),
        "--region",
        "0,0,1,1",
        "--neighbors",
    )

    _assert_cli_error(
        result,
        code=2,
        error_type="CLIArgumentError",
        message="neighbors",
    )


@pytest.mark.parametrize(
    "value",
    ["", "1,2;", "1", "1,2,3", "a,b", "1,2;;3,4"],
)
def test_cli_rejects_malformed_points(
    run_cli: Callable[..., Any],
    rgb_png: Path,
    value: str,
) -> None:
    result = run_cli(
        "image",
        "sample",
        str(rgb_png),
        "--points",
        value,
    )

    _assert_cli_error(result, code=2, error_type="CLIArgumentError")


@pytest.mark.parametrize(
    "value",
    ["", "0,0,1", "0,0,1,1,2", "a,b,c,d"],
)
def test_cli_rejects_malformed_regions(
    run_cli: Callable[..., Any],
    rgb_png: Path,
    value: str,
) -> None:
    result = run_cli(
        "image",
        "sample",
        str(rgb_png),
        "--region",
        value,
    )

    _assert_cli_error(result, code=2, error_type="CLIArgumentError")


def test_cli_missing_source_is_json_error(
    run_cli: Callable[..., Any],
    tmp_path: Path,
) -> None:
    result = run_cli(
        "image",
        "sample",
        str(tmp_path / "absent.png"),
        "--points",
        "0,0",
    )

    _assert_cli_error(result, code=1, error_type="FileNotFoundError")


def test_cli_invalid_point_fails_whole_request(
    run_cli: Callable[..., Any],
    rgb_png: Path,
) -> None:
    result = run_cli(
        "image",
        "sample",
        str(rgb_png),
        "--points",
        "0,0;9,9",
    )

    _assert_cli_error(
        result,
        code=1,
        error_type="ValueError",
        message="outside image bounds",
    )


def test_cli_invalid_region_is_json_error(
    run_cli: Callable[..., Any],
    rgb_png: Path,
) -> None:
    result = run_cli(
        "image",
        "sample",
        str(rgb_png),
        "--region",
        "0,0,0,1",
    )

    _assert_cli_error(
        result,
        code=1,
        error_type="ValueError",
        message="Region requires",
    )


def test_cli_does_not_modify_source_or_create_files(
    run_cli: Callable[..., Any],
    tmp_path: Path,
    rgb_png: Path,
) -> None:
    before = _digest(rgb_png)
    entries = _snapshot(tmp_path)

    points = run_cli(
        "image",
        "sample",
        str(rgb_png),
        "--points",
        "0,0",
    )
    region = run_cli(
        "image",
        "sample",
        str(rgb_png),
        "--region",
        "0,0,4,2",
    )

    assert points.returncode == 0
    assert region.returncode == 0
    assert _digest(rgb_png) == before
    assert _snapshot(tmp_path) == entries


def test_cli_help_exposes_sample(run_cli: Callable[..., Any]) -> None:
    result = run_cli("image", "sample", "--help")

    assert result.returncode == 0
    assert result.stderr == ""
    assert "usage:" in result.stdout
    assert "--points" in result.stdout
    assert "--region" in result.stdout
    assert "--neighbors" in result.stdout

    listing = run_cli("image", "--help")
    assert listing.returncode == 0
    assert "sample" in listing.stdout
