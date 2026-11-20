"""Compare two images and report changed connected components.

Pixels are changed when their maximum channel difference exceeds the requested
tolerance. Opaque images compare R, G and B directly. Images containing
transparency compare floor-premultiplied RGB channels plus the alpha channel;
alpha differences contribute to the same maximum and use the same strict
tolerance threshold. Fully transparent pixels therefore ignore hidden RGB.

Two artifacts are produced non-transactionally: a PNG visualization and a JSON
report. See ``_publish_outputs`` for the publication order and partial-failure
contract.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from collections.abc import Iterator
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, UnidentifiedImageError

from ._components import (
    ComponentTracker,
    iter_component_artifact,
    scan_mask,
)

MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_PIXELS = 50_000_000
MAX_OUTPUT_SIZE = 100 * 1024 * 1024
MAX_COMPONENT_LABELS = 1_000_000

# Working-set target for band construction, not a hard memory bound. A band is
# a full-width or full-height strip, so an extreme aspect ratio can produce a
# band much larger than this value.
TILE_PIXELS = 4_000_000

DEFAULT_TOLERANCE = 15
DEFAULT_MIN_AREA = 9

DARKEN_LUT = [value // 2 for value in range(256)]
NEUTRAL_BACKGROUND = (128, 128, 128)
HIGHLIGHT = (255, 0, 0)

Region = tuple[int, int, int, int]


def diff_image(
    before: str | Path,
    after: str | Path,
    output: str | Path,
    image: str | Path,
    *,
    tolerance: int = DEFAULT_TOLERANCE,
    min_area: int = DEFAULT_MIN_AREA,
    region: Region | None = None,
) -> dict[str, str]:
    if type(tolerance) is not int:
        raise TypeError("Tolerance must be an integer")
    if not 0 <= tolerance <= 255:
        raise ValueError("Tolerance must be between 0 and 255")

    if type(min_area) is not int:
        raise TypeError("Minimum area must be an integer")
    if min_area < 1:
        raise ValueError("Minimum area must be a positive integer")

    validated_region = _validate_region(region) if region is not None else None

    before_source = _resolve_source(before)
    after_source = _resolve_source(after)
    json_output = _resolve_output(output, ".json")
    png_output = _resolve_output(image, ".png")

    if json_output == png_output:
        raise ValueError("Output paths must be different")
    for target in (json_output, png_output):
        if target in (before_source, after_source):
            raise ValueError("Source and output must be different")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Output already exists: {target}")

    before_image = _load(before_source)
    try:
        after_image = _load(after_source)
    except BaseException:
        before_image.close()
        raise

    try:
        if before_image.size != after_image.size:
            raise ValueError(
                "Images must have identical dimensions: "
                f"{before_image.width}x{before_image.height} and "
                f"{after_image.width}x{after_image.height}"
            )

        width, height = before_image.size
        if validated_region is None:
            resolved_region = (0, 0, width, height)
        else:
            resolved_region = _validate_region_bounds(
                validated_region,
                width,
                height,
            )

        tracker, stats, visualization = _analyze_pair(
            before_image,
            after_image,
            resolved_region,
            tolerance,
            min_area,
        )
    finally:
        before_image.close()
        after_image.close()

    try:
        tracker.sort_retained()
        header = _header(
            width,
            height,
            resolved_region,
            tolerance,
            min_area,
            tracker,
            stats,
        )
        _publish_outputs(
            header,
            tracker,
            visualization,
            json_output,
            png_output,
        )
    finally:
        visualization.close()

    return {
        "status": "ok",
        "output": str(json_output),
        "image": str(png_output),
    }


def _resolve_source(value: str | Path) -> Path:
    source = Path(value).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Not a regular file: {source}")
    return source


def _resolve_output(value: str | Path, suffix: str) -> Path:
    requested = Path(value).expanduser()
    output = requested.parent.resolve() / requested.name
    if output.suffix.lower() != suffix:
        raise ValueError(f"Output must use the {suffix} extension")
    return output


def _validate_region(region: object) -> Region:
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


def _validate_region_bounds(
    region: Region,
    width: int,
    height: int,
) -> Region:
    x1, y1, x2, y2 = region
    if x2 > width or y2 > height:
        raise ValueError(
            f"Region [{x1}, {y1}, {x2}, {y2}] exceeds image bounds "
            f"({width}x{height})"
        )
    return region


def _load(source: Path) -> Image.Image:
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

            with BytesIO(data) as buffer:
                image = Image.open(buffer)
                width, height = image.size
                if width < 1 or height < 1 or width * height > MAX_PIXELS:
                    image.close()
                    raise ValueError(
                        "Invalid dimensions or source pixel limit exceeded"
                    )
                if getattr(image, "n_frames", 1) != 1:
                    image.close()
                    raise ValueError("Multi-frame images are not supported")

                image.load()
                return image

    except (
        UnidentifiedImageError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError(f"Invalid or unsafe image: {source}") from exc


def _has_transparency(image: Image.Image) -> bool:
    return "A" in image.getbands() or "transparency" in image.info


def _analyze_pair(
    before: Image.Image,
    after: Image.Image,
    region: Region,
    tolerance: int,
    min_area: int,
) -> tuple[ComponentTracker, _DifferenceStats, Image.Image]:
    x1, y1, x2, y2 = region
    region_width = x2 - x1
    region_height = y2 - y1
    horizontal = region_width >= region_height
    has_transparency = _has_transparency(before) or _has_transparency(after)

    tracker = ComponentTracker(
        full_width=before.width,
        min_area=min_area,
        horizontal=horizontal,
        position_offset=x1 if horizontal else y1,
        max_labels=MAX_COMPONENT_LABELS,
    )
    threshold = [
        255 if value > tolerance else 0
        for value in range(256)
    ]
    stats = _DifferenceStats()
    visualization = Image.new("RGB", before.size)

    try:
        _paint_background(
            visualization,
            after,
            has_transparency=has_transparency,
        )
        for box in _band_boxes(region, horizontal):
            scan_width = (
                box[2] - box[0]
                if horizontal
                else box[3] - box[1]
            )
            scanline_offset = box[1] if horizontal else box[0]

            changed, histogram = _band_difference(
                before,
                after,
                box,
                has_transparency,
                threshold,
                horizontal,
            )
            try:
                stats.add(histogram)
                scan_mask(
                    changed,
                    tracker,
                    width=scan_width,
                    scanline_offset=scanline_offset,
                )
                _paint_highlights(
                    visualization,
                    box,
                    changed,
                    horizontal=horizontal,
                )
            finally:
                changed.close()

        tracker.finish_scanline()
        tracker.close_active()
    except BaseException:
        visualization.close()
        raise

    return tracker, stats, visualization


def _band_boxes(
    region: Region,
    horizontal: bool,
) -> Iterator[Region]:
    x1, y1, x2, y2 = region
    if horizontal:
        band_size = max(1, TILE_PIXELS // (x2 - x1))
        for top in range(y1, y2, band_size):
            yield (x1, top, x2, min(top + band_size, y2))
    else:
        band_size = max(1, TILE_PIXELS // (y2 - y1))
        for left in range(x1, x2, band_size):
            yield (left, y1, min(left + band_size, x2), y2)


def _band_difference(
    before: Image.Image,
    after: Image.Image,
    box: Region,
    has_transparency: bool,
    threshold: list[int],
    horizontal: bool,
) -> tuple[Image.Image, list[int]]:
    with before.crop(box) as before_band, after.crop(box) as after_band:
        if horizontal:
            before_scan = before_band
            after_scan = after_band
        else:
            before_scan = before_band.transpose(Image.Transpose.TRANSPOSE)
            after_scan = after_band.transpose(Image.Transpose.TRANSPOSE)

        try:
            if has_transparency:
                max_difference = _rgba_difference(
                    before_scan,
                    after_scan,
                )
            else:
                max_difference = _rgb_difference(
                    before_scan,
                    after_scan,
                )
        finally:
            if before_scan is not before_band:
                before_scan.close()
            if after_scan is not after_band:
                after_scan.close()

    try:
        histogram = max_difference.histogram()
        changed = max_difference.point(threshold)
    finally:
        max_difference.close()

    return changed, histogram


def _rgb_difference(
    first: Image.Image,
    second: Image.Image,
) -> Image.Image:
    first_rgb = first if first.mode == "RGB" else first.convert("RGB")
    second_rgb = second if second.mode == "RGB" else second.convert("RGB")
    try:
        result: Image.Image | None = None
        for index in range(3):
            first_channel = first_rgb.getchannel(index)
            second_channel = second_rgb.getchannel(index)
            try:
                difference = ImageChops.difference(
                    first_channel,
                    second_channel,
                )
            finally:
                first_channel.close()
                second_channel.close()

            if result is None:
                result = difference
            else:
                merged = ImageChops.lighter(result, difference)
                result.close()
                difference.close()
                result = merged

        if result is None:
            raise ValueError("RGB comparison produced no channels")
        return result
    finally:
        if first_rgb is not first:
            first_rgb.close()
        if second_rgb is not second:
            second_rgb.close()


def _rgba_difference(
    first: Image.Image,
    second: Image.Image,
) -> Image.Image:
    first_rgba = first if first.mode == "RGBA" else first.convert("RGBA")
    second_rgba = second if second.mode == "RGBA" else second.convert("RGBA")
    first_alpha = first_rgba.getchannel(3)
    second_alpha = second_rgba.getchannel(3)

    try:
        result = ImageChops.difference(first_alpha, second_alpha)
        try:
            for index in range(3):
                first_channel = first_rgba.getchannel(index)
                second_channel = second_rgba.getchannel(index)
                try:
                    first_premultiplied = ImageChops.multiply(
                        first_channel,
                        first_alpha,
                    )
                    second_premultiplied = ImageChops.multiply(
                        second_channel,
                        second_alpha,
                    )
                    try:
                        difference = ImageChops.difference(
                            first_premultiplied,
                            second_premultiplied,
                        )
                    finally:
                        first_premultiplied.close()
                        second_premultiplied.close()
                finally:
                    first_channel.close()
                    second_channel.close()

                merged = ImageChops.lighter(result, difference)
                result.close()
                difference.close()
                result = merged

            return result
        except BaseException:
            result.close()
            raise
    finally:
        first_alpha.close()
        second_alpha.close()
        if first_rgba is not first:
            first_rgba.close()
        if second_rgba is not second:
            second_rgba.close()


def _paint_highlights(
    visualization: Image.Image,
    box: Region,
    changed: Image.Image,
    *,
    horizontal: bool,
) -> None:
    mask = (
        changed
        if horizontal
        else changed.transpose(Image.Transpose.TRANSPOSE)
    )
    try:
        visualization.paste(HIGHLIGHT, box, mask)
    finally:
        if mask is not changed:
            mask.close()


def _paint_background(
    visualization: Image.Image,
    after: Image.Image,
    *,
    has_transparency: bool,
) -> None:
    horizontal = after.width >= after.height
    region = (0, 0, after.width, after.height)
    for box in _band_boxes(region, horizontal):
        background = _background_band(
            after,
            box,
            has_transparency=has_transparency,
        )
        try:
            visualization.paste(background, box)
        finally:
            background.close()


def _background_band(
    after: Image.Image,
    box: Region,
    *,
    has_transparency: bool,
) -> Image.Image:
    with after.crop(box) as cropped:
        if has_transparency:
            with cropped.convert("RGBA") as rgba:
                flat = Image.new("RGB", cropped.size, NEUTRAL_BACKGROUND)
                alpha = rgba.getchannel("A")
                try:
                    flat.paste(rgba, mask=alpha)
                    gray = flat.convert("L")
                finally:
                    alpha.close()
                    flat.close()
        else:
            gray = cropped.convert("L")

    try:
        dark = gray.point(DARKEN_LUT)
    finally:
        gray.close()

    try:
        return dark.convert("RGB")
    finally:
        dark.close()


class _DifferenceStats:
    def __init__(self) -> None:
        self.total = 0
        self.maximum = 0

    def add(self, histogram: list[int]) -> None:
        self.total += sum(
            value * count
            for value, count in enumerate(histogram)
        )
        for value in range(255, -1, -1):
            if histogram[value]:
                self.maximum = max(self.maximum, value)
                break


def _header(
    width: int,
    height: int,
    region: Region,
    tolerance: int,
    min_area: int,
    tracker: ComponentTracker,
    stats: _DifferenceStats,
) -> dict[str, Any]:
    x1, y1, x2, y2 = region
    region_pixels = (x2 - x1) * (y2 - y1)
    return {
        "status": "ok",
        "width": width,
        "height": height,
        "region": list(region),
        "tolerance": tolerance,
        "min_area": min_area,
        "summary": {
            "region_pixels": region_pixels,
            "changed_pixels": tracker.matched_pixels,
            "changed_percentage": round(
                100 * tracker.matched_pixels / region_pixels,
                2,
            ),
            "component_count": tracker.component_count,
            "retained_component_count": len(tracker.retained),
            "retained_pixels": tracker.retained_pixels,
            "mean_difference": round(
                stats.total / region_pixels,
                2,
            ),
            "max_difference": stats.maximum,
        },
    }


def _publish_outputs(
    header: dict[str, Any],
    tracker: ComponentTracker,
    visualization: Image.Image,
    json_output: Path,
    png_output: Path,
) -> None:
    """Write both artifacts, then publish PNG first and JSON second.

    The two hardlinks are not a transaction. If PNG publication fails, JSON is
    never attempted. If JSON publication fails after PNG succeeded, the
    published PNG is left untouched and an explicit partial-publication error
    is raised. Success is reported only after both hardlinks succeed.
    """
    json_output.parent.mkdir(parents=True, exist_ok=True)
    png_output.parent.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        json_directory = stack.enter_context(
            tempfile.TemporaryDirectory(
                dir=json_output.parent,
                prefix=".agentkit-",
                ignore_cleanup_errors=True,
            )
        )
        png_directory = stack.enter_context(
            tempfile.TemporaryDirectory(
                dir=png_output.parent,
                prefix=".agentkit-",
                ignore_cleanup_errors=True,
            )
        )

        json_temporary = Path(json_directory) / "result.json.tmp"
        png_temporary = Path(png_directory) / "result.png.tmp"

        _write_png(visualization, png_temporary)
        visualization.close()
        _write_json(header, tracker, json_temporary)

        os.link(png_temporary, png_output)
        try:
            os.link(json_temporary, json_output)
        except OSError as exc:
            raise OSError(
                "Partial publication: PNG artifact was published to "
                f"{png_output}; JSON artifact could not be published to "
                f"{json_output}: {exc}"
            ) from exc


def _write_json(
    header: dict[str, Any],
    tracker: ComponentTracker,
    path: Path,
) -> None:
    total = 0
    with path.open("wb") as stream:
        for chunk in iter_component_artifact(header, tracker):
            total = _write_bytes(stream, chunk, total)

    if not 0 < path.stat().st_size <= MAX_OUTPUT_SIZE:
        raise ValueError(
            f"Output exceeds the {MAX_OUTPUT_SIZE}-byte limit"
        )


def _write_png(
    visualization: Image.Image,
    path: Path,
) -> None:
    visualization.save(path, format="PNG")
    if not 0 < path.stat().st_size <= MAX_OUTPUT_SIZE:
        raise ValueError(
            f"Output exceeds the {MAX_OUTPUT_SIZE}-byte limit"
        )

    try:
        with Image.open(path) as check:
            if check.size != visualization.size or check.mode != "RGB":
                raise ValueError(
                    "Generated PNG has unexpected dimensions or mode"
                )
            check.verify()
    except (OSError, SyntaxError) as exc:
        raise ValueError("Generated PNG is invalid") from exc


def _write_bytes(
    stream: Any,
    text: str,
    total: int,
) -> int:
    encoded = text.encode("utf-8")
    total += len(encoded)
    if total > MAX_OUTPUT_SIZE:
        raise ValueError(
            f"Output exceeds the {MAX_OUTPUT_SIZE}-byte limit"
        )
    stream.write(encoded)
    return total
