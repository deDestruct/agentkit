from __future__ import annotations

import errno
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

from agentkit.image import diff as diff_module
from agentkit.image.diff import diff_image

BACKGROUND = (0, 0, 0)
TARGET = (255, 0, 0)


def _make(
    tmp_path: Path,
    name: str,
    mode: str,
    rows: list[list[Any]],
    *,
    transparency: int | None = None,
    palette: list[int] | None = None,
) -> Path:
    height = len(rows)
    width = len(rows[0])
    path = tmp_path / name
    with Image.new(mode, (width, height)) as image:
        if mode == "P":
            image.putpalette(
                palette
                if palette is not None
                else [255, 0, 0, 0, 255, 0, 0, 0, 255]
            )
        for y, row in enumerate(rows):
            for x, value in enumerate(row):
                image.putpixel((x, y), value)
        if transparency is not None:
            image.info["transparency"] = transparency
        image.save(path)
    return path


def _pair(
    tmp_path: Path,
    before_rows: list[list[Any]],
    after_rows: list[list[Any]],
    *,
    mode: str = "RGB",
    before_transparency: int | None = None,
    after_transparency: int | None = None,
    palette: list[int] | None = None,
) -> tuple[Path, Path]:
    return (
        _make(
            tmp_path,
            "before.png",
            mode,
            before_rows,
            transparency=before_transparency,
            palette=palette,
        ),
        _make(
            tmp_path,
            "after.png",
            mode,
            after_rows,
            transparency=after_transparency,
            palette=palette,
        ),
    )


def _diff(
    before: Path,
    after: Path,
    *,
    tolerance: int = 15,
    min_area: int = 9,
    region: tuple[int, int, int, int] | None = None,
    json_name: str = "result.json",
    image_name: str = "result.png",
) -> tuple[dict[str, Any], Path, Path, dict[str, str]]:
    output = before.parent / json_name
    image = before.parent / image_name
    metadata = diff_image(
        before,
        after,
        output,
        image,
        tolerance=tolerance,
        min_area=min_area,
        region=region,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))
    return artifact, output, image, metadata


def _pixels(path: Path, mode: str) -> list[list[tuple[int, ...]]]:
    with Image.open(path) as image, image.convert(mode) as converted:
        return [
            [
                cast(
                    tuple[int, ...],
                    converted.getpixel((x, y)),
                )
                for x in range(converted.width)
            ]
            for y in range(converted.height)
        ]


def _pixel_difference(
    first: tuple[int, ...],
    second: tuple[int, ...],
    *,
    transparency: bool,
) -> int:
    if transparency:
        first_alpha = first[3]
        second_alpha = second[3]
        return max(
            abs(first_alpha - second_alpha),
            *(
                abs(
                    first[index] * first_alpha // 255
                    - second[index] * second_alpha // 255
                )
                for index in range(3)
            ),
        )

    return max(
        abs(first[index] - second[index])
        for index in range(3)
    )


def _reference(
    before: Path,
    after: Path,
    *,
    mode: str,
    region: tuple[int, int, int, int],
    tolerance: int,
    min_area: int,
) -> dict[str, Any]:
    transparency = mode == "RGBA"
    before_pixels = _pixels(before, mode)
    after_pixels = _pixels(after, mode)
    width = len(after_pixels[0])
    x1, y1, x2, y2 = region

    differences: dict[tuple[int, int], int] = {}
    changed: set[tuple[int, int]] = set()
    total = 0
    maximum = 0
    for y in range(y1, y2):
        for x in range(x1, x2):
            difference = _pixel_difference(
                before_pixels[y][x],
                after_pixels[y][x],
                transparency=transparency,
            )
            total += difference
            maximum = max(maximum, difference)
            differences[(x, y)] = difference
            if difference > tolerance:
                changed.add((x, y))

    seen: set[tuple[int, int]] = set()
    components: list[dict[str, Any]] = []
    for y in range(y1, y2):
        for x in range(x1, x2):
            if (x, y) not in changed or (x, y) in seen:
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
                        if neighbour in changed and neighbour not in seen:
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
    for component in retained:
        del component["first"]

    region_pixels = (x2 - x1) * (y2 - y1)
    return {
        "summary": {
            "region_pixels": region_pixels,
            "changed_pixels": len(changed),
            "changed_percentage": round(
                100 * len(changed) / region_pixels,
                2,
            ),
            "component_count": len(components),
            "retained_component_count": len(retained),
            "retained_pixels": sum(
                component["pixels"] for component in retained
            ),
            "mean_difference": round(total / region_pixels, 2),
            "max_difference": maximum,
        },
        "components": retained,
        "changed": changed,
        "differences": differences,
    }


def _assert_matches_reference(
    artifact: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert artifact["summary"] == expected["summary"]
    assert artifact["components"] == expected["components"]


def _png_pixels(path: Path) -> list[list[tuple[int, int, int]]]:
    with Image.open(path) as image:
        assert image.mode == "RGB"
        return [
            [
                cast(
                    tuple[int, int, int],
                    image.getpixel((x, y)),
                )
                for x in range(image.width)
            ]
            for y in range(image.height)
        ]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


# ---------------------------------------------------------------------------
# Comparison semantics
# ---------------------------------------------------------------------------


def test_identical_images_produce_no_changes(tmp_path: Path) -> None:
    rows = [[(10, 10, 10), (40, 40, 40)] for _ in range(3)]
    before, after = _pair(tmp_path, rows, rows)

    artifact, _, image, metadata = _diff(before, after)

    assert metadata["status"] == "ok"
    assert artifact["summary"] == {
        "region_pixels": 6,
        "changed_pixels": 0,
        "changed_percentage": 0.0,
        "component_count": 0,
        "retained_component_count": 0,
        "retained_pixels": 0,
        "mean_difference": 0.0,
        "max_difference": 0,
    }
    assert artifact["components"] == []
    assert {
        pixel for row in _png_pixels(image) for pixel in row
    } == {(5, 5, 5), (20, 20, 20)}


def test_completely_different_images_form_one_component(
    tmp_path: Path,
) -> None:
    before, after = _pair(
        tmp_path,
        [[BACKGROUND] * 4 for _ in range(2)],
        [[(255, 255, 255)] * 4 for _ in range(2)],
        before_transparency=None,
    )

    artifact, _, image, _ = _diff(before, after, tolerance=0, min_area=1)

    assert artifact["summary"]["changed_pixels"] == 8
    assert artifact["summary"]["changed_percentage"] == 100.0
    assert artifact["summary"]["component_count"] == 1
    assert artifact["components"] == [
        {
            "box": [0, 0, 4, 2],
            "pixels": 8,
            "centroid": [1.5, 0.5],
        }
    ]
    assert all(
        pixel == TARGET
        for row in _png_pixels(image)
        for pixel in row
    )


def test_single_pixel_difference_and_min_area_filtering(
    tmp_path: Path,
) -> None:
    before = [[BACKGROUND] * 4 for _ in range(3)]
    after = [row[:] for row in before]
    after[1][2] = TARGET
    before_path, after_path = _pair(
        tmp_path,
        before,
        after,
        before_transparency=None,
    )

    filtered, _, filtered_image, _ = _diff(
        before_path,
        after_path,
        tolerance=0,
    )
    assert filtered["summary"]["changed_pixels"] == 1
    assert filtered["summary"]["component_count"] == 1
    assert filtered["summary"]["retained_component_count"] == 0
    assert filtered["summary"]["retained_pixels"] == 0
    assert filtered["components"] == []
    assert _png_pixels(filtered_image)[1][2] == TARGET

    retained, _, _, _ = _diff(
        before_path,
        after_path,
        tolerance=0,
        min_area=1,
        json_name="retained.json",
        image_name="retained.png",
    )
    assert retained["components"] == [
        {
            "box": [2, 1, 3, 2],
            "pixels": 1,
            "centroid": [2.0, 1.0],
        }
    ]


@pytest.mark.parametrize(
    ("difference", "changed"),
    [(0, False), (15, False), (16, True), (100, True)],
)
def test_opaque_tolerance_boundary_is_strict(
    tmp_path: Path,
    difference: int,
    changed: bool,
) -> None:
    before, after = _pair(
        tmp_path,
        [[(100, 100, 100)]],
        [[(100 + difference, 100, 100)]],
    )

    artifact, _, _, _ = _diff(before, after, tolerance=15)

    assert artifact["summary"]["changed_pixels"] == int(changed)
    assert artifact["summary"]["max_difference"] == difference


def test_tolerance_zero_and_255(tmp_path: Path) -> None:
    # Documented extremes, checked through the public function's validation
    # and a one-pixel fixture.
    rows_before = [[(1, 2, 3)]]
    rows_after = [[(2, 2, 3)]]

    before, after = _pair(tmp_path, rows_before, rows_after)
    artifact, _, _, _ = _diff(before, after, tolerance=0, min_area=1)
    assert artifact["summary"]["changed_pixels"] == 1
    artifact, _, _, _ = _diff(
        before,
        after,
        tolerance=255,
        json_name="max.json",
        image_name="max.png",
    )
    assert artifact["summary"]["changed_pixels"] == 0


@pytest.mark.parametrize(
    ("after_alpha", "changed"),
    [(100, False), (115, False), (116, True)],
)
def test_alpha_difference_uses_same_strict_tolerance(
    tmp_path: Path,
    after_alpha: int,
    changed: bool,
) -> None:
    before, after = _pair(
        tmp_path,
        [[(0, 0, 0, 100)]],
        [[(0, 0, 0, after_alpha)]],
        mode="RGBA",
    )

    artifact, _, _, _ = _diff(before, after, tolerance=15)

    assert artifact["summary"]["changed_pixels"] == int(changed)
    assert artifact["summary"]["max_difference"] == after_alpha - 100


def test_premultiplied_rgb_difference_with_equal_alpha(
    tmp_path: Path,
) -> None:
    before, after = _pair(
        tmp_path,
        [[(200, 0, 0, 128)]],
        [[(0, 0, 0, 128)]],
        mode="RGBA",
    )

    hidden, _, _, _ = _diff(before, after, tolerance=100)
    assert hidden["summary"]["changed_pixels"] == 0
    assert hidden["summary"]["max_difference"] == 100

    visible, _, _, _ = _diff(
        before,
        after,
        tolerance=99,
        json_name="visible.json",
        image_name="visible.png",
    )
    assert visible["summary"]["changed_pixels"] == 1


def test_hidden_rgb_in_fully_transparent_pixels_is_ignored(
    tmp_path: Path,
) -> None:
    before, after = _pair(
        tmp_path,
        [[(255, 0, 0, 0)] * 3],
        [[(0, 255, 0, 0)] * 3],
        mode="RGBA",
    )

    artifact, _, image, _ = _diff(before, after, tolerance=0, min_area=1)

    assert artifact["summary"]["changed_pixels"] == 0
    assert artifact["summary"]["max_difference"] == 0
    assert artifact["components"] == []
    assert {
        pixel for row in _png_pixels(image) for pixel in row
    } == {(64, 64, 64)}


def test_hidden_palette_rgb_is_ignored_when_transparent(
    tmp_path: Path,
) -> None:
    before = _make(
        tmp_path,
        "before.png",
        "P",
        [[0, 0]],
        transparency=0,
        palette=[255, 0, 0, 0, 255, 0, 0, 0, 255],
    )
    after = _make(
        tmp_path,
        "after.png",
        "P",
        [[0, 0]],
        transparency=0,
        palette=[0, 255, 0, 255, 0, 0, 0, 0, 255],
    )

    artifact, _, _, _ = _diff(before, after, tolerance=0, min_area=1)

    assert artifact["summary"]["changed_pixels"] == 0
    assert artifact["components"] == []


def test_palette_and_grayscale_are_converted(tmp_path: Path) -> None:
    before = _make(
        tmp_path,
        "before.png",
        "L",
        [[0, 0, 0]],
    )
    after = _make(
        tmp_path,
        "after.png",
        "RGB",
        [[(255, 255, 255)] * 3],
    )

    artifact, _, _, _ = _diff(before, after, tolerance=0, min_area=1)

    assert artifact["summary"]["changed_pixels"] == 3
    assert artifact["summary"]["component_count"] == 1


# ---------------------------------------------------------------------------
# Regions, connectivity and ordering
# ---------------------------------------------------------------------------


def test_region_clipping_uses_absolute_coordinates(tmp_path: Path) -> None:
    before = [[BACKGROUND] * 5 for _ in range(4)]
    after = [row[:] for row in before]
    after[0][0] = TARGET
    after[1][1] = TARGET
    after[1][2] = TARGET
    before_path, after_path = _pair(tmp_path, before, after)

    artifact, _, image, _ = _diff(
        before_path,
        after_path,
        tolerance=0,
        min_area=1,
        region=(1, 1, 3, 3),
    )

    assert artifact["summary"]["region_pixels"] == 4
    assert artifact["summary"]["changed_pixels"] == 2
    assert artifact["components"] == [
        {
            "box": [1, 1, 3, 2],
            "pixels": 2,
            "centroid": [1.5, 1.0],
        }
    ]
    pixels = _png_pixels(image)
    # The visualization keeps the whole darkened AFTER background; only raw
    # changes inside the region are highlighted, so the out-of-region change at
    # (0, 0) is background rather than red.
    assert pixels[0][0] == (38, 38, 38)
    assert pixels[0][0] != TARGET
    assert pixels[1][1] == TARGET


def test_diagonal_pixels_are_eight_connected(tmp_path: Path) -> None:
    before = [[BACKGROUND] * 4 for _ in range(4)]
    after = [row[:] for row in before]
    after[1][1] = TARGET
    after[2][2] = TARGET
    after[3][0] = (0, 255, 0)
    before_path, after_path = _pair(tmp_path, before, after)

    artifact, _, _, _ = _diff(
        before_path,
        after_path,
        tolerance=0,
        min_area=1,
    )

    assert artifact["summary"]["component_count"] == 2
    assert [component["pixels"] for component in artifact["components"]] == [
        2,
        1,
    ]


def test_component_ordering_is_deterministic(tmp_path: Path) -> None:
    before = [[BACKGROUND] * 6 for _ in range(3)]
    after = [row[:] for row in before]
    for x in (0, 1, 2):
        after[0][x] = TARGET
    for x in (0, 1):
        after[2][x] = TARGET
    after[2][2] = TARGET
    before_path, after_path = _pair(tmp_path, before, after)

    artifact, _, _, _ = _diff(
        before_path,
        after_path,
        tolerance=0,
        min_area=1,
    )

    assert [
        (component["pixels"], component["box"])
        for component in artifact["components"]
    ] == [
        (3, [0, 0, 3, 1]),
        (3, [0, 2, 3, 3]),
    ]


# ---------------------------------------------------------------------------
# Exact statistics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "seed", "tolerance", "min_area", "width", "height"),
    [
        ("RGB", 0, 0, 1, 7, 5),
        ("RGB", 1, 20, 2, 7, 5),
        ("RGB", 2, 60, 3, 7, 5),
        ("RGBA", 3, 0, 1, 7, 5),
        ("RGBA", 4, 40, 2, 7, 5),
        ("RGBA", 5, 90, 4, 7, 5),
        ("RGB", 6, 20, 2, 5, 7),
        ("RGBA", 7, 40, 3, 5, 7),
    ],
)
def test_random_fixtures_match_independent_reference(
    tmp_path: Path,
    mode: str,
    seed: int,
    tolerance: int,
    min_area: int,
    width: int,
    height: int,
) -> None:
    import random

    generator = random.Random(seed)

    def random_pixel() -> tuple[int, ...]:
        channels = tuple(
            generator.randrange(0, 256, 30)
            for _ in range(3)
        )
        if mode == "RGBA":
            return (*channels, generator.randrange(0, 256, 40))
        return channels

    before_rows = [
        [random_pixel() for _ in range(width)]
        for _ in range(height)
    ]
    after_rows = [
        [random_pixel() for _ in range(width)]
        for _ in range(height)
    ]
    before, after = _pair(tmp_path, before_rows, after_rows, mode=mode)
    region = (1, 1, width - 1, height - 1)

    artifact, _, _, _ = _diff(
        before,
        after,
        tolerance=tolerance,
        min_area=min_area,
        region=region,
    )
    expected = _reference(
        before,
        after,
        mode=mode,
        region=region,
        tolerance=tolerance,
        min_area=min_area,
    )

    _assert_matches_reference(artifact, expected)


@pytest.mark.parametrize("transpose", [False, True])
def test_band_boundaries_match_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    transpose: bool,
) -> None:
    import random

    generator = random.Random(41 if transpose else 40)
    width, height = (9, 6) if not transpose else (6, 9)
    before_rows = [
        [BACKGROUND for _ in range(width)]
        for _ in range(height)
    ]
    after_rows = [
        [
            TARGET if generator.random() < 0.55 else BACKGROUND
            for _ in range(width)
        ]
        for _ in range(height)
    ]
    before, after = _pair(tmp_path, before_rows, after_rows)
    monkeypatch.setattr(diff_module, "TILE_PIXELS", 3)

    artifact, _, _, _ = _diff(
        before,
        after,
        tolerance=0,
        min_area=1,
    )
    expected = _reference(
        before,
        after,
        mode="RGB",
        region=(0, 0, width, height),
        tolerance=0,
        min_area=1,
    )

    _assert_matches_reference(artifact, expected)


def test_mixed_rgb_and_rgba_sources_use_alpha_comparison(
    tmp_path: Path,
) -> None:
    before = _make(
        tmp_path,
        "before.png",
        "RGB",
        [[(0, 0, 0), (0, 0, 0)]],
    )
    after = _make(
        tmp_path,
        "after.png",
        "RGBA",
        [[(0, 0, 0, 255), (255, 0, 0, 255)]],
    )

    artifact, _, _, _ = _diff(before, after, tolerance=0, min_area=1)

    assert artifact["summary"]["changed_pixels"] == 1
    assert artifact["components"] == [
        {
            "box": [1, 0, 2, 1],
            "pixels": 1,
            "centroid": [1.0, 0.0],
        }
    ]


def test_exact_summary_statistics(tmp_path: Path) -> None:
    before, after = _pair(
        tmp_path,
        [[(0, 0, 0), (0, 0, 0)]],
        [[(255, 0, 0), (10, 0, 0)]],
    )

    artifact, _, _, _ = _diff(before, after, tolerance=15)

    assert artifact["summary"] == {
        "region_pixels": 2,
        "changed_pixels": 1,
        "changed_percentage": 50.0,
        "component_count": 1,
        "retained_component_count": 0,
        "retained_pixels": 0,
        "mean_difference": 132.5,
        "max_difference": 255,
    }


# ---------------------------------------------------------------------------
# PNG visualization
# ---------------------------------------------------------------------------


def test_png_highlights_raw_changes_before_filtering(
    tmp_path: Path,
) -> None:
    before = [[BACKGROUND] * 4 for _ in range(3)]
    after = [row[:] for row in before]
    after[1][1] = TARGET
    after[2][3] = TARGET
    before_path, after_path = _pair(tmp_path, before, after)

    artifact, _, image, _ = _diff(
        before_path,
        after_path,
        tolerance=0,
        min_area=99,
    )

    assert artifact["summary"]["changed_pixels"] == 2
    assert artifact["components"] == []
    pixels = _png_pixels(image)
    red = [
        (x, y)
        for y, row in enumerate(pixels)
        for x, pixel in enumerate(row)
        if pixel == TARGET
    ]
    assert red == [(1, 1), (3, 2)]


def test_png_uses_darkened_after_and_neutral_transparency(
    tmp_path: Path,
) -> None:
    before, after = _pair(
        tmp_path,
        [[(0, 0, 0, 0), (255, 255, 255, 255)]],
        [[(200, 100, 50, 0), (0, 0, 0, 255)]],
        mode="RGBA",
    )

    artifact, _, image, _ = _diff(before, after, tolerance=0, min_area=1)

    assert artifact["summary"]["changed_pixels"] == 1
    pixels = _png_pixels(image)
    assert pixels[0][0] == (64, 64, 64)
    assert pixels[0][1] == TARGET


def test_png_red_pixels_match_changed_statistics(
    tmp_path: Path,
) -> None:
    import random

    generator = random.Random(21)
    before_rows = [
        [
            (generator.randrange(256), generator.randrange(256), 0)
            for _ in range(12)
        ]
        for _ in range(9)
    ]
    after_rows = [
        [
            (generator.randrange(256), generator.randrange(256), 0)
            for _ in range(12)
        ]
        for _ in range(9)
    ]
    before, after = _pair(tmp_path, before_rows, after_rows)

    artifact, _, image, _ = _diff(
        before,
        after,
        tolerance=30,
        min_area=2,
    )

    red_count = sum(
        pixel == TARGET
        for row in _png_pixels(image)
        for pixel in row
    )
    assert red_count == artifact["summary"]["changed_pixels"]


# ---------------------------------------------------------------------------
# Artifact format
# ---------------------------------------------------------------------------


def test_artifact_has_multiline_header_and_compact_component_lines(
    tmp_path: Path,
) -> None:
    before = [[BACKGROUND] * 8]
    after = [[TARGET] * 2 + [BACKGROUND] + [TARGET] * 3 + [BACKGROUND, TARGET]]
    before_path, after_path = _pair(tmp_path, before, after)

    artifact, output, _, _ = _diff(
        before_path,
        after_path,
        tolerance=0,
        min_area=1,
    )

    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "{"
    start = lines.index('  "components": [')
    end = lines.index("  ]")
    component_lines = lines[start + 1 : end]
    assert len(component_lines) == 3

    components = [
        json.loads(line.removesuffix(","))
        for line in component_lines
    ]
    assert components == artifact["components"]
    assert [component["pixels"] for component in components] == [3, 2, 1]
    assert all(
        line.startswith('    {"box":')
        for line in component_lines
    )
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_artifact_without_matches_is_valid_json(tmp_path: Path) -> None:
    before = [[BACKGROUND] * 3]
    before_path, after_path = _pair(tmp_path, before, before)

    artifact, output, _, _ = _diff(
        before_path,
        after_path,
        tolerance=0,
    )

    lines = output.read_text(encoding="utf-8").splitlines()
    assert '  "components": []' in lines
    assert artifact["components"] == []
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


# ---------------------------------------------------------------------------
# Input validation and limits
# ---------------------------------------------------------------------------


def test_rejects_dimension_mismatch(
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    before, _ = _pair(
        tmp_path,
        [[BACKGROUND] * 4 for _ in range(2)],
        [[BACKGROUND] * 4 for _ in range(2)],
    )
    after = _make(
        tmp_path,
        "after.png",
        "RGB",
        [[BACKGROUND] * 5 for _ in range(2)],
    )

    with pytest.raises(ValueError, match="identical dimensions"):
        diff_image(
            before,
            after,
            tmp_path / "result.json",
            tmp_path / "result.png",
        )

    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / "result.png").exists()
    assert_no_leftovers(tmp_path)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"tolerance": -1}, ValueError),
        ({"tolerance": 256}, ValueError),
        ({"tolerance": "1"}, TypeError),
        ({"min_area": 0}, ValueError),
        ({"min_area": "1"}, TypeError),
    ],
)
def test_rejects_invalid_options(
    tmp_path: Path,
    kwargs: dict[str, Any],
    error: type[Exception],
) -> None:
    before, after = _pair(
        tmp_path,
        [[BACKGROUND]],
        [[TARGET]],
    )

    with pytest.raises(error):
        diff_image(
            before,
            after,
            tmp_path / "result.json",
            tmp_path / "result.png",
            **kwargs,
        )


@pytest.mark.parametrize(
    ("region", "error", "message"),
    [
        ((0, 0, 1), TypeError, "four integers"),
        ((-1, 0, 1, 1), ValueError, "negative"),
        ((1, 1, 1, 2), ValueError, "x2 > x1"),
        ((0, 0, 3, 3), ValueError, "exceeds image bounds"),
    ],
)
def test_rejects_invalid_region(
    tmp_path: Path,
    region: Any,
    error: type[Exception],
    message: str,
) -> None:
    before, after = _pair(
        tmp_path,
        [[BACKGROUND] * 2 for _ in range(2)],
        [[TARGET] * 2 for _ in range(2)],
    )

    with pytest.raises(error, match=message):
        diff_image(
            before,
            after,
            tmp_path / "result.json",
            tmp_path / "result.png",
            region=region,
        )


def test_rejects_source_file_size_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    before, after = _pair(
        tmp_path,
        [[BACKGROUND]],
        [[TARGET]],
    )
    monkeypatch.setattr(diff_module, "MAX_FILE_SIZE", 10)

    with pytest.raises(ValueError, match="between 1 byte"):
        diff_image(
            before,
            after,
            tmp_path / "result.json",
            tmp_path / "result.png",
        )


def test_rejects_source_pixel_limit(tmp_path: Path) -> None:
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    before.write_bytes(_png_header(8_000, 8_000))
    after.write_bytes(_png_header(8_000, 8_000))

    with pytest.raises(ValueError, match="pixel"):
        diff_image(
            before,
            after,
            tmp_path / "result.json",
            tmp_path / "result.png",
        )


def test_rejects_empty_and_corrupted_sources(
    empty_file: Path,
    garbage_file: Path,
    tmp_path: Path,
) -> None:
    good = _make(tmp_path, "good.png", "RGB", [[BACKGROUND]])

    with pytest.raises(ValueError, match="between 1 byte"):
        diff_image(
            empty_file,
            good,
            tmp_path / "result.json",
            tmp_path / "result.png",
        )
    with pytest.raises(ValueError, match="Invalid or unsafe image"):
        diff_image(
            garbage_file,
            good,
            tmp_path / "result.json",
            tmp_path / "result.png",
        )


def test_rejects_multi_frame_source(
    animated_gif: Path,
    tmp_path: Path,
) -> None:
    after = _make(tmp_path, "after.png", "RGB", [[BACKGROUND]])

    with pytest.raises(ValueError, match="Multi-frame"):
        diff_image(
            animated_gif,
            after,
            tmp_path / "result.json",
            tmp_path / "result.png",
        )


def test_json_output_size_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    before, after = _pair(
        tmp_path,
        [[BACKGROUND]],
        [[TARGET]],
    )
    _, measured, image, _ = _diff(before, after, tolerance=0, min_area=1)
    measured_size = measured.stat().st_size
    assert image.stat().st_size < measured_size

    monkeypatch.setattr(
        diff_module,
        "MAX_OUTPUT_SIZE",
        measured_size,
    )
    accepted, _, _, _ = _diff(
        before,
        after,
        tolerance=0,
        min_area=1,
        json_name="accepted.json",
        image_name="accepted.png",
    )
    assert accepted["status"] == "ok"

    monkeypatch.setattr(
        diff_module,
        "MAX_OUTPUT_SIZE",
        measured_size - 1,
    )
    with pytest.raises(ValueError, match="Output exceeds"):
        diff_image(
            before,
            after,
            tmp_path / "rejected.json",
            tmp_path / "rejected.png",
            tolerance=0,
            min_area=1,
        )

    assert not (tmp_path / "rejected.json").exists()
    assert not (tmp_path / "rejected.png").exists()
    assert_no_leftovers(tmp_path)


def test_png_output_size_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    import random

    generator = random.Random(31)
    before = [
        [(0, 0, 0) for _ in range(40)]
        for _ in range(40)
    ]
    after = [
        [
            (
                generator.randrange(256),
                generator.randrange(256),
                generator.randrange(256),
            )
            for _ in range(40)
        ]
        for _ in range(40)
    ]
    before_path, after_path = _pair(tmp_path, before, after)

    monkeypatch.setattr(diff_module, "MAX_OUTPUT_SIZE", 1_000)

    with pytest.raises(ValueError, match="Output exceeds"):
        diff_image(
            before_path,
            after_path,
            tmp_path / "result.json",
            tmp_path / "result.png",
            tolerance=255,
        )

    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / "result.png").exists()
    assert_no_leftovers(tmp_path)


def test_component_label_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    before = [[BACKGROUND] * 6]
    after = [[TARGET, BACKGROUND, TARGET, BACKGROUND, TARGET, BACKGROUND]]
    before_path, after_path = _pair(tmp_path, before, after)
    monkeypatch.setattr(diff_module, "MAX_COMPONENT_LABELS", 1)

    with pytest.raises(ValueError, match="Component label limit exceeded"):
        diff_image(
            before_path,
            after_path,
            tmp_path / "result.json",
            tmp_path / "result.png",
            tolerance=0,
            min_area=1,
        )

    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / "result.png").exists()
    assert_no_leftovers(tmp_path)


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------


def test_rejects_existing_outputs(
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    before, after = _pair(
        tmp_path,
        [[BACKGROUND]],
        [[TARGET]],
    )
    output = tmp_path / "result.json"
    output.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError, match="already exists"):
        diff_image(
            before,
            after,
            output,
            tmp_path / "result.png",
        )

    assert output.read_bytes() == b"sentinel"
    assert not (tmp_path / "result.png").exists()
    assert_no_leftovers(tmp_path)


def test_rejects_unsupported_output_extensions(
    tmp_path: Path,
) -> None:
    before, after = _pair(
        tmp_path,
        [[BACKGROUND]],
        [[TARGET]],
    )

    with pytest.raises(ValueError, match=r"\.json"):
        diff_image(
            before,
            after,
            tmp_path / "result.txt",
            tmp_path / "result.png",
        )
    with pytest.raises(ValueError, match=r"\.png"):
        diff_image(
            before,
            after,
            tmp_path / "result.json",
            tmp_path / "result.jpg",
        )


def test_rejects_output_matching_a_source(
    tmp_path: Path,
) -> None:
    # A source can carry any extension; the collision check must still run.
    before = tmp_path / "before.json"
    with Image.new("RGB", (1, 1), BACKGROUND) as image:
        image.save(before, format="PNG")
    after = _make(tmp_path, "after.png", "RGB", [[TARGET]])

    with pytest.raises(ValueError, match="must be different"):
        diff_image(
            before,
            after,
            before,
            tmp_path / "result.png",
        )


def test_rejects_dangling_symlink_output(
    tmp_path: Path,
    symlink_or_skip: Callable[[Path, Path], None],
) -> None:
    before, after = _pair(
        tmp_path,
        [[BACKGROUND]],
        [[TARGET]],
    )
    output = tmp_path / "dangling.json"
    symlink_or_skip(output, tmp_path / "missing.json")

    with pytest.raises(FileExistsError, match="already exists"):
        diff_image(
            before,
            after,
            output,
            tmp_path / "result.png",
        )

    assert output.is_symlink()
    assert not output.exists()


def test_rejects_missing_source_and_directory_source(
    tmp_path: Path,
) -> None:
    after = _make(tmp_path, "after.png", "RGB", [[TARGET]])

    with pytest.raises(FileNotFoundError):
        diff_image(
            tmp_path / "missing.png",
            after,
            tmp_path / "result.json",
            tmp_path / "result.png",
        )

    with pytest.raises(ValueError, match="Not a regular file"):
        diff_image(
            tmp_path,
            after,
            tmp_path / "result.json",
            tmp_path / "result.png",
        )


def test_creates_missing_output_directories(tmp_path: Path) -> None:
    before, after = _pair(
        tmp_path,
        [[BACKGROUND]],
        [[TARGET]],
    )
    output = tmp_path / "nested" / "json" / "result.json"
    image = tmp_path / "other" / "image" / "result.png"

    artifact, output, image, _ = _diff(
        before,
        after,
        tolerance=0,
        min_area=1,
        json_name=str(output.relative_to(tmp_path)),
        image_name=str(image.relative_to(tmp_path)),
    )

    assert artifact["status"] == "ok"
    assert output.is_file()
    assert image.is_file()


# ---------------------------------------------------------------------------
# Publication contract
# ---------------------------------------------------------------------------


def test_first_publication_failure_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    before, after = _pair(
        tmp_path,
        [[BACKGROUND]],
        [[TARGET]],
    )
    calls: list[Path] = []

    def failing_link(source: Any, destination: Any) -> None:
        calls.append(Path(destination))
        raise OSError(errno.EOPNOTSUPP, "Hard links are not supported")

    monkeypatch.setattr(diff_module.os, "link", failing_link)

    with pytest.raises(OSError, match="not supported"):
        diff_image(
            before,
            after,
            tmp_path / "result.json",
            tmp_path / "result.png",
        )

    assert calls == [tmp_path / "result.png"]
    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / "result.png").exists()
    assert_no_leftovers(tmp_path)


def test_second_publication_failure_preserves_first_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    before, after = _pair(
        tmp_path,
        [[BACKGROUND]],
        [[TARGET]],
    )
    json_output = tmp_path / "result.json"
    png_output = tmp_path / "result.png"
    real_link = diff_module.os.link
    calls: list[Path] = []

    def failing_second_link(source: Any, destination: Any) -> None:
        calls.append(Path(destination))
        if len(calls) == 2:
            raise OSError(errno.EIO, "second publication failed")
        real_link(source, destination)

    monkeypatch.setattr(diff_module.os, "link", failing_second_link)

    with pytest.raises(OSError, match="Partial publication") as captured:
        diff_image(before, after, json_output, png_output)

    message = str(captured.value)
    assert str(png_output) in message
    assert str(json_output) in message
    assert "second publication failed" in message
    assert calls == [png_output, json_output]
    assert not json_output.exists()
    assert png_output.is_file()
    with Image.open(png_output) as published:
        assert published.size == (1, 1)
        assert published.mode == "RGB"
    assert_no_leftovers(tmp_path)


def test_competing_png_prevents_json_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    before, after = _pair(
        tmp_path,
        [[BACKGROUND]],
        [[TARGET]],
    )
    png_output = tmp_path / "result.png"
    json_output = tmp_path / "result.json"
    real_link = diff_module.os.link
    calls = 0

    def racing_link(source: Any, destination: Any) -> None:
        nonlocal calls
        calls += 1
        if Path(destination) == png_output:
            png_output.write_bytes(b"competitor")
        real_link(source, destination)

    monkeypatch.setattr(diff_module.os, "link", racing_link)

    with pytest.raises(FileExistsError):
        diff_image(before, after, json_output, png_output)

    assert calls == 1
    assert png_output.read_bytes() == b"competitor"
    assert not json_output.exists()
    assert_no_leftovers(tmp_path)


def test_competing_json_preserves_png_and_competitor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    before, after = _pair(
        tmp_path,
        [[BACKGROUND]],
        [[TARGET]],
    )
    png_output = tmp_path / "result.png"
    json_output = tmp_path / "result.json"
    real_link = diff_module.os.link

    def racing_link(source: Any, destination: Any) -> None:
        if Path(destination) == json_output:
            json_output.write_bytes(b"competitor")
        real_link(source, destination)

    monkeypatch.setattr(diff_module.os, "link", racing_link)

    with pytest.raises(OSError, match="Partial publication"):
        diff_image(before, after, json_output, png_output)

    assert json_output.read_bytes() == b"competitor"
    assert png_output.is_file()
    with Image.open(png_output) as published:
        assert published.size == (1, 1)
    assert_no_leftovers(tmp_path)


def test_success_publishes_both_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    before, after = _pair(
        tmp_path,
        [[BACKGROUND]],
        [[TARGET]],
    )
    json_output = tmp_path / "result.json"
    png_output = tmp_path / "result.png"
    real_link = diff_module.os.link
    published: list[Path] = []

    def checking_link(source: Any, destination: Any) -> None:
        destination_path = Path(destination)
        assert not destination_path.exists()
        assert not destination_path.is_symlink()
        published.append(destination_path)
        real_link(source, destination)

    monkeypatch.setattr(diff_module.os, "link", checking_link)

    metadata = diff_image(before, after, json_output, png_output)

    assert published == [png_output, json_output]
    assert metadata == {
        "status": "ok",
        "output": str(json_output),
        "image": str(png_output),
    }
    assert json_output.is_file()
    assert png_output.is_file()
    assert_no_leftovers(tmp_path)


def test_sources_are_unchanged(
    tmp_path: Path,
    assert_no_leftovers: Callable[[Path], None],
) -> None:
    before, after = _pair(
        tmp_path,
        [[BACKGROUND] * 3 for _ in range(2)],
        [[TARGET] * 3 for _ in range(2)],
    )
    before_digest = _digest(before)
    after_digest = _digest(after)

    _diff(before, after, tolerance=0)

    assert _digest(before) == before_digest
    assert _digest(after) == after_digest
    assert_no_leftovers(tmp_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_success_prints_compact_metadata(
    run_cli: Callable[..., Any],
    tmp_path: Path,
) -> None:
    before, after = _pair(
        tmp_path,
        [[BACKGROUND] * 2],
        [[TARGET, BACKGROUND]],
    )
    output = tmp_path / "result.json"
    image = tmp_path / "result.png"

    result = run_cli(
        "image",
        "diff",
        str(before),
        str(after),
        str(output),
        str(image),
        "--tolerance",
        "0",
        "--min-area",
        "1",
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.strip() == json.dumps(
        {
            "status": "ok",
            "output": str(output),
            "image": str(image),
        },
        separators=(",", ":"),
    )
    assert json.loads(output.read_text(encoding="utf-8"))["summary"][
        "changed_pixels"
    ] == 1
    with Image.open(image) as visualization:
        assert visualization.mode == "RGB"


def test_cli_help_exposes_diff(run_cli: Callable[..., Any]) -> None:
    result = run_cli("image", "diff", "--help")

    assert result.returncode == 0
    assert result.stderr == ""
    for option in (
        "--tolerance",
        "--region",
        "--min-area",
        "before",
        "after",
    ):
        assert option in result.stdout

    listing = run_cli("image", "--help")
    assert listing.returncode == 0
    assert "diff" in listing.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ["--tolerance", "256"],
        ["--tolerance", "-1"],
        ["--min-area", "0"],
        ["--region", "1,2,3"],
    ],
)
def test_cli_runtime_option_errors_are_json_errors(
    run_cli: Callable[..., Any],
    tmp_path: Path,
    arguments: list[str],
) -> None:
    before, after = _pair(
        tmp_path,
        [[BACKGROUND]],
        [[TARGET]],
    )

    result = run_cli(
        "image",
        "diff",
        str(before),
        str(after),
        str(tmp_path / "result.json"),
        str(tmp_path / "result.png"),
        *arguments,
    )

    # argparse rejects the malformed region; the rest reach diff_image.
    if arguments == ["--region", "1,2,3"]:
        _assert_cli_error(
            result,
            code=2,
            error_type="CLIArgumentError",
        )
    else:
        assert result.returncode == 1
        assert result.stdout == ""
        payload = json.loads(result.stderr)
        assert payload["status"] == "error"
        assert "Traceback" not in result.stderr


def test_cli_existing_output_is_json_error(
    run_cli: Callable[..., Any],
    tmp_path: Path,
) -> None:
    before, after = _pair(
        tmp_path,
        [[BACKGROUND]],
        [[TARGET]],
    )
    output = tmp_path / "result.json"
    output.write_bytes(b"sentinel")

    result = run_cli(
        "image",
        "diff",
        str(before),
        str(after),
        str(output),
        str(tmp_path / "result.png"),
    )

    _assert_cli_error(result, code=1, error_type="FileExistsError")
    assert output.read_bytes() == b"sentinel"
