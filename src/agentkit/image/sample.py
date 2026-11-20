from __future__ import annotations

import math
import warnings
from collections.abc import Iterator, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_PIXELS = 50_000_000
MAX_POINTS = 10
MAX_COLOR_GROUPS = 10
QUANTIZATION_STEP = 16
QUANTIZED_COLOR_COUNT = QUANTIZATION_STEP**3
TILE_PIXELS = 4_000_000

BUCKET_LUT = [value // QUANTIZATION_STEP for value in range(256)] * 3
TRANSPARENT_MASK_LUT = [255] + [0] * 255
SENTINEL_RGB = (QUANTIZATION_STEP, 0, 0)

Point = tuple[int, int]
Region = tuple[int, int, int, int]
Bucket = tuple[int, int, int]
Pixel = tuple[int, int, int, int]


def sample_image(
    source: str | Path,
    *,
    points: Sequence[Point] | None = None,
    region: Region | None = None,
    neighbors: bool = False,
) -> dict[str, Any]:
    _validate_mode(points, region, neighbors)

    validated_points: list[Point] | None = None
    validated_region: Region | None = None
    if points is not None:
        validated_points = _validate_points(points)
    else:
        assert region is not None
        validated_region = _validate_region(region)

    source = Path(source).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Not a regular file: {source}")

    with source.open("rb") as stream:
        data = stream.read(MAX_FILE_SIZE + 1)
    if not 0 < len(data) <= MAX_FILE_SIZE:
        raise ValueError("Source file must be between 1 byte and 100 MiB")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning,
            )

            with BytesIO(data) as buffer, Image.open(buffer) as image:
                width, height = image.size
                if width < 1 or height < 1 or width * height > MAX_PIXELS:
                    raise ValueError(
                        "Invalid dimensions or source pixel limit exceeded"
                    )
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Multi-frame images are not supported")

                if validated_points is not None:
                    _validate_point_bounds(validated_points, width, height)
                else:
                    assert validated_region is not None
                    _validate_region_bounds(validated_region, width, height)

                image.load()

                if validated_points is not None:
                    return _sample_points(
                        image,
                        validated_points,
                        neighbors,
                    )

                assert validated_region is not None
                return _sample_region(image, validated_region)

    except (
        UnidentifiedImageError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError(f"Invalid or unsafe image: {source}") from exc


def _validate_mode(
    points: Sequence[Point] | None,
    region: Region | None,
    neighbors: bool,
) -> None:
    if type(neighbors) is not bool:
        raise TypeError("Neighbors must be a boolean")
    if (points is None) == (region is None):
        raise ValueError("Exactly one of points or region must be provided")
    if neighbors and points is None:
        raise ValueError("Neighbors can only be used with points")


def _validate_points(points: Sequence[Point]) -> list[Point]:
    if not isinstance(points, (tuple, list)):
        raise TypeError("Points must be a sequence of coordinate pairs")
    if not 1 <= len(points) <= MAX_POINTS:
        raise ValueError(
            f"Points must contain between 1 and {MAX_POINTS} coordinate pairs"
        )

    validated: list[Point] = []
    for pair in points:
        if (
            not isinstance(pair, (tuple, list))
            or len(pair) != 2
            or any(type(value) is not int for value in pair)
        ):
            raise TypeError("Each point must contain exactly two integers")
        validated.append((pair[0], pair[1]))
    return validated


def _validate_region(region: Region) -> Region:
    if (
        not isinstance(region, (tuple, list))
        or len(region) != 4
        or any(type(value) is not int for value in region)
    ):
        raise TypeError("Region must contain exactly four integers")

    x1, y1, x2, y2 = region
    if x1 < 0 or y1 < 0:
        raise ValueError("Region coordinates cannot be negative")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Region requires x2 > x1 and y2 > y1")
    return (x1, y1, x2, y2)


def _validate_point_bounds(
    points: Sequence[Point],
    width: int,
    height: int,
) -> None:
    for x, y in points:
        if x < 0 or y < 0:
            raise ValueError("Point coordinates cannot be negative")
        if x >= width or y >= height:
            raise ValueError(
                f"Point ({x}, {y}) is outside image bounds ({width}x{height})"
            )


def _validate_region_bounds(
    region: Region,
    width: int,
    height: int,
) -> None:
    x1, y1, x2, y2 = region
    if x2 > width or y2 > height:
        raise ValueError(
            f"Region [{x1}, {y1}, {x2}, {y2}] exceeds image bounds "
            f"({width}x{height})"
        )


def _sample_points(
    image: Image.Image,
    points: Sequence[Point],
    neighbors: bool,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for x, y in points:
        if neighbors:
            center, matrix = _neighborhood(image, x, y)
            entries.append(
                {
                    "x": x,
                    "y": y,
                    "rgba": list(center),
                    "neighbors": matrix,
                }
            )
        else:
            entries.append(
                {
                    "x": x,
                    "y": y,
                    "rgba": list(_rgba_at(image, x, y)),
                }
            )
    return {"status": "ok", "points": entries}


def _rgba_at(image: Image.Image, x: int, y: int) -> Pixel:
    with (
        image.crop((x, y, x + 1, y + 1)) as cropped,
        cropped.convert("RGBA") as rgba,
    ):
        return _pixel_tuple(rgba.getpixel((0, 0)))


def _neighborhood(
    image: Image.Image,
    x: int,
    y: int,
) -> tuple[Pixel, list[list[str | None]]]:
    left = max(0, x - 1)
    top = max(0, y - 1)
    right = min(image.width, x + 2)
    bottom = min(image.height, y + 2)

    with (
        image.crop((left, top, right, bottom)) as cropped,
        cropped.convert("RGBA") as rgba,
    ):
        center: Pixel | None = None
        matrix: list[list[str | None]] = []

        for dy in (-1, 0, 1):
            row: list[str | None] = []
            for dx in (-1, 0, 1):
                nx = x + dx
                ny = y + dy
                if 0 <= nx < image.width and 0 <= ny < image.height:
                    pixel = _pixel_tuple(
                        rgba.getpixel((nx - left, ny - top))
                    )
                    if dx == 0 and dy == 0:
                        center = pixel
                    row.append(_hex_rgba(pixel))
                else:
                    row.append(None)
            matrix.append(row)

        if center is None:
            raise ValueError(
                f"Point ({x}, {y}) is outside image bounds"
            )
        return center, matrix


def _sample_region(image: Image.Image, region: Region) -> dict[str, Any]:
    x1, y1, x2, y2 = region
    pixel_count = (x2 - x1) * (y2 - y1)
    has_transparency = (
        "A" in image.getbands() or "transparency" in image.info
    )

    counts_by_bucket: dict[Bucket, int] = {}
    analyzed = 0
    for box in _tile_boxes(region):
        for bucket, count in _tile_counts(
            image,
            box,
            has_transparency,
        ).items():
            counts_by_bucket[bucket] = (
                counts_by_bucket.get(bucket, 0) + count
            )
            analyzed += count

    groups = [
        (
            count,
            (
                bucket[0] * QUANTIZATION_STEP + QUANTIZATION_STEP // 2,
                bucket[1] * QUANTIZATION_STEP + QUANTIZATION_STEP // 2,
                bucket[2] * QUANTIZATION_STEP + QUANTIZATION_STEP // 2,
            ),
        )
        for bucket, count in counts_by_bucket.items()
    ]
    groups.sort(key=lambda item: (-item[0], item[1]))
    top_groups = groups[:MAX_COLOR_GROUPS]

    colors: list[dict[str, Any]] = []
    if analyzed:
        colors = [
            {
                "rgb": list(rgb),
                "count": count,
                "percentage": round(100 * count / analyzed, 2),
            }
            for count, rgb in top_groups
        ]

    coverage = (
        round(
            100 * sum(count for count, _ in top_groups) / analyzed,
            2,
        )
        if analyzed
        else 0.0
    )

    return {
        "status": "ok",
        "region": [x1, y1, x2, y2],
        "pixel_count": pixel_count,
        "analyzed_pixel_count": analyzed,
        "quantization_step": QUANTIZATION_STEP,
        "color_group_count": len(groups),
        "coverage_percentage": coverage,
        "colors": colors,
    }


def _tile_boxes(region: Region) -> Iterator[Region]:
    x1, y1, x2, y2 = region
    tile_width, tile_height = _tile_size(
        x2 - x1,
        y2 - y1,
        TILE_PIXELS,
    )

    for top in range(y1, y2, tile_height):
        bottom = min(top + tile_height, y2)
        for left in range(x1, x2, tile_width):
            right = min(left + tile_width, x2)
            yield (left, top, right, bottom)


def _tile_size(
    width: int,
    height: int,
    budget: int,
) -> tuple[int, int]:
    if budget < 1:
        raise ValueError("Tile budget must be positive")

    if width >= height:
        ideal_width = round(math.sqrt(budget * width / height))
        tile_width = min(width, budget, max(1, ideal_width))
        tile_height = min(height, max(1, budget // tile_width))
    else:
        ideal_height = round(math.sqrt(budget * height / width))
        tile_height = min(height, budget, max(1, ideal_height))
        tile_width = min(width, max(1, budget // tile_height))

    return tile_width, tile_height


def _tile_counts(
    image: Image.Image,
    box: Region,
    has_transparency: bool,
) -> dict[Bucket, int]:
    with image.crop(box) as cropped:
        if has_transparency:
            with cropped.convert("RGBA") as rgba:
                alpha = rgba.getchannel("A")
                try:
                    mask = alpha.point(TRANSPARENT_MASK_LUT)
                finally:
                    alpha.close()

                try:
                    with rgba.convert("RGB") as rgb:
                        bucket = rgb.point(BUCKET_LUT)
                        try:
                            bucket.paste(SENTINEL_RGB, mask=mask)
                            return _collect_counts(bucket)
                        finally:
                            bucket.close()
                finally:
                    mask.close()

        with cropped.convert("RGB") as rgb:
            bucket = rgb.point(BUCKET_LUT)
            try:
                return _collect_counts(bucket)
            finally:
                bucket.close()


def _collect_counts(bucket: Image.Image) -> dict[Bucket, int]:
    entries = bucket.getcolors(maxcolors=QUANTIZED_COLOR_COUNT + 1)
    if entries is None:
        raise ValueError(
            "Quantized histogram exceeds expected color count"
        )

    counts: dict[Bucket, int] = {}
    for count, color in entries:
        if color == SENTINEL_RGB:
            continue
        if not isinstance(color, tuple) or len(color) != 3:
            raise ValueError("Unexpected quantized color")
        counts[(int(color[0]), int(color[1]), int(color[2]))] = count
    return counts


def _pixel_tuple(pixel: object) -> Pixel:
    if not isinstance(pixel, tuple) or len(pixel) != 4:
        raise ValueError("Unexpected pixel format")
    return (
        int(pixel[0]),
        int(pixel[1]),
        int(pixel[2]),
        int(pixel[3]),
    )


def _hex_rgba(pixel: Pixel) -> str:
    red, green, blue, alpha = pixel
    return f"#{red:02X}{green:02X}{blue:02X}{alpha:02X}"
