"""Connected-component analysis for pixels matching a reference color.

Pixels are matched inside the requested rectangle with inclusive Chebyshev
channel tolerance. Fully transparent pixels never match. Components use fixed
8-connectivity and are reported with exact pixel counts, exclusive boxes and
two-decimal centroids.

Labels are provisional union-find slots. Slots for components filtered out by
``min_area`` are reused after the component can no longer grow, while
``MAX_COMPONENT_LABELS`` bounds simultaneously allocated slots rather than the
total number of components seen. Exceeding any limit fails without publishing
an artifact.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import warnings
from array import array
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, UnidentifiedImageError

MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_PIXELS = 50_000_000
MAX_OUTPUT_SIZE = 100 * 1024 * 1024
MAX_COMPONENT_LABELS = 1_000_000

# Working-set target for band construction, not a hard memory bound. A band is
# a full-width or full-height strip, so when the strip's short side is one
# pixel the band can be as large as the whole image.
TILE_PIXELS = 4_000_000

DEFAULT_TOLERANCE = 24
DEFAULT_MIN_AREA = 9

ALPHA_MASK_LUT = [0] + [255] * 255
RUN_PATTERN = re.compile(rb"[^\x00]+")
MAX_COORD = 2**31 - 1

Region = tuple[int, int, int, int]
Color = tuple[int, int, int]


def parse_color(value: str) -> Color:
    if not isinstance(value, str):
        raise TypeError("Color must be a string")

    text = value.strip()
    if text.startswith("#"):
        if len(text) != 7 or any(
            character not in "0123456789abcdefABCDEF"
            for character in text[1:]
        ):
            raise ValueError("Color must use #RRGGBB or R,G,B format")
        return (
            int(text[1:3], 16),
            int(text[3:5], 16),
            int(text[5:7], 16),
        )

    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3 or any(
        not part or not part.isascii() or not part.isdigit()
        for part in parts
    ):
        raise ValueError("Color must use #RRGGBB or R,G,B format")

    channels = (int(parts[0]), int(parts[1]), int(parts[2]))
    if any(channel > 255 for channel in channels):
        raise ValueError("Color channels must be between 0 and 255")
    return channels


def find_image(
    source: str | Path,
    output: str | Path,
    *,
    color: str,
    tolerance: int = DEFAULT_TOLERANCE,
    region: Region | None = None,
    min_area: int = DEFAULT_MIN_AREA,
) -> dict[str, str]:
    target = parse_color(color)

    if type(tolerance) is not int:
        raise TypeError("Tolerance must be an integer")
    if not 0 <= tolerance <= 255:
        raise ValueError("Tolerance must be between 0 and 255")

    if type(min_area) is not int:
        raise TypeError("Minimum area must be an integer")
    if min_area < 1:
        raise ValueError("Minimum area must be a positive integer")

    validated_region = _validate_region(region) if region is not None else None

    source = Path(source).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Not a regular file: {source}")

    requested = Path(output).expanduser()
    output = requested.parent.resolve() / requested.name
    if source == output:
        raise ValueError("Source and output must be different")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    if output.suffix.lower() != ".json":
        raise ValueError("Output must use the .json extension")

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

                if validated_region is None:
                    resolved_region = (0, 0, width, height)
                else:
                    resolved_region = _validate_region_bounds(
                        validated_region,
                        width,
                        height,
                    )

                image.load()
                data = b""

                tracker = _analyze(
                    image,
                    resolved_region,
                    target,
                    tolerance,
                    min_area,
                )

    except (
        UnidentifiedImageError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError(f"Invalid or unsafe image: {source}") from exc

    tracker.sort_retained()
    summary = _summary(tracker, resolved_region)
    header: dict[str, Any] = {
        "status": "ok",
        "width": width,
        "height": height,
        "region": list(resolved_region),
        "color": list(target),
        "tolerance": tolerance,
        "min_area": min_area,
        "summary": summary,
    }
    _publish_artifact(header, tracker, output)
    return {"status": "ok", "output": str(output)}


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


def _analyze(
    image: Image.Image,
    region: Region,
    target: Color,
    tolerance: int,
    min_area: int,
) -> _ComponentTracker:
    x1, y1, x2, y2 = region
    region_width = x2 - x1
    region_height = y2 - y1
    horizontal = region_width >= region_height
    has_transparency = (
        "A" in image.getbands() or "transparency" in image.info
    )

    tracker = _ComponentTracker(
        full_width=image.width,
        min_area=min_area,
        horizontal=horizontal,
        position_offset=x1 if horizontal else y1,
    )

    if horizontal:
        band_size = max(1, TILE_PIXELS // region_width)
        bands = (
            (x1, top, x2, min(top + band_size, y2))
            for top in range(y1, y2, band_size)
        )
    else:
        band_size = max(1, TILE_PIXELS // region_height)
        bands = (
            (left, y1, min(left + band_size, x2), y2)
            for left in range(x1, x2, band_size)
        )

    for box in bands:
        _scan_band(
            image,
            box,
            target,
            tolerance,
            has_transparency,
            tracker,
            horizontal=horizontal,
            scanline_offset=box[1] if horizontal else box[0],
        )

    tracker.finish_scanline()
    tracker.close_active()
    return tracker


def _scan_band(
    image: Image.Image,
    box: Region,
    target: Color,
    tolerance: int,
    has_transparency: bool,
    tracker: _ComponentTracker,
    *,
    horizontal: bool,
    scanline_offset: int,
) -> None:
    if horizontal:
        with image.crop(box) as cropped:
            mask = _band_mask(cropped, target, tolerance, has_transparency)
        scan_width = box[2] - box[0]
    else:
        with image.crop(box) as cropped:
            band = cropped.transpose(Image.Transpose.TRANSPOSE)
        try:
            mask = _band_mask(band, target, tolerance, has_transparency)
        finally:
            band.close()
        scan_width = box[3] - box[1]

    try:
        _scan_mask(
            mask,
            tracker,
            width=scan_width,
            scanline_offset=scanline_offset,
        )
    finally:
        mask.close()


def _band_mask(
    image: Image.Image,
    target: Color,
    tolerance: int,
    has_transparency: bool,
) -> Image.Image:
    if has_transparency:
        if image.mode == "RGBA":
            return _rgba_match_mask(image, target, tolerance)
        with image.convert("RGBA") as rgba:
            return _rgba_match_mask(rgba, target, tolerance)

    if image.mode == "RGB":
        return _rgb_match_mask(image, target, tolerance)
    with image.convert("RGB") as rgb:
        return _rgb_match_mask(rgb, target, tolerance)


def _rgba_match_mask(
    image: Image.Image,
    target: Color,
    tolerance: int,
) -> Image.Image:
    alpha = image.getchannel(3)
    try:
        alpha_mask = alpha.point(ALPHA_MASK_LUT)
    finally:
        alpha.close()

    try:
        rgb_mask = _rgb_match_mask(image, target, tolerance)
        try:
            return ImageChops.multiply(rgb_mask, alpha_mask)
        finally:
            rgb_mask.close()
    finally:
        alpha_mask.close()


def _rgb_match_mask(
    image: Image.Image,
    target: Color,
    tolerance: int,
) -> Image.Image:
    masks: list[Image.Image] = []
    try:
        for index, value in enumerate(target):
            channel = image.getchannel(index)
            try:
                masks.append(
                    channel.point(_channel_lut(value, tolerance))
                )
            finally:
                channel.close()

        red, green, blue = masks
        with ImageChops.multiply(red, green) as red_green:
            return ImageChops.multiply(red_green, blue)
    finally:
        for mask in masks:
            mask.close()


def _channel_lut(target: int, tolerance: int) -> list[int]:
    return [
        255 if abs(value - target) <= tolerance else 0
        for value in range(256)
    ]


def _scan_mask(
    mask: Image.Image,
    tracker: _ComponentTracker,
    *,
    width: int,
    scanline_offset: int,
) -> None:
    data = mask.tobytes()
    for match in RUN_PATTERN.finditer(data):
        position = match.start()
        end = match.end()
        while position < end:
            scanline = position // width
            row_end = (scanline + 1) * width
            segment_end = min(end, row_end)
            start = position - scanline * width
            tracker.add_run(
                start,
                segment_end - scanline * width,
                scanline_offset + scanline,
            )
            position = segment_end


class _ComponentTracker:
    def __init__(
        self,
        *,
        full_width: int,
        min_area: int,
        horizontal: bool,
        position_offset: int,
    ) -> None:
        self.full_width = full_width
        self.min_area = min_area
        self.horizontal = horizontal
        self.position_offset = position_offset

        self.parent = array("i")
        self.sizes = array("i")
        self.min_x = array("i")
        self.min_y = array("i")
        self.max_x = array("i")
        self.max_y = array("i")
        self.sum_x = array("q")
        self.sum_y = array("q")
        self.first_key = array("i")

        self.free_labels: list[int] = []
        self.retained: list[int] = []
        self.active: set[int] = set()

        self.component_count = 0
        self.matched_pixels = 0
        self.retained_pixels = 0

        self.current_scanline: int | None = None
        self.prev_index = 0

        self.prev_starts = array("i")
        self.prev_ends = array("i")
        self.prev_labels = array("i")
        self.cur_starts = array("i")
        self.cur_ends = array("i")
        self.cur_labels = array("i")

    def find(self, label: int) -> int:
        parent = self.parent
        root = label
        while parent[root] != root:
            root = parent[root]
        while parent[label] != label:
            next_label = parent[label]
            parent[label] = root
            label = next_label
        return root

    def union(self, first: int, second: int) -> int:
        first = self.find(first)
        second = self.find(second)
        if first == second:
            return first

        if self.sizes[first] < self.sizes[second] or (
            self.sizes[first] == self.sizes[second] and first > second
        ):
            first, second = second, first

        self.parent[second] = first
        self.sizes[first] += self.sizes[second]
        self.min_x[first] = min(self.min_x[first], self.min_x[second])
        self.min_y[first] = min(self.min_y[first], self.min_y[second])
        self.max_x[first] = max(self.max_x[first], self.max_x[second])
        self.max_y[first] = max(self.max_y[first], self.max_y[second])
        self.sum_x[first] += self.sum_x[second]
        self.sum_y[first] += self.sum_y[second]
        self.first_key[first] = min(
            self.first_key[first],
            self.first_key[second],
        )
        return first

    def add_run(self, start: int, end: int, scanline: int) -> None:
        self._start_scanline(scanline)
        self.matched_pixels += end - start

        prev_starts = self.prev_starts
        prev_ends = self.prev_ends
        prev_labels = self.prev_labels
        prev_count = len(prev_starts)

        index = self.prev_index
        while index < prev_count and prev_ends[index] < start:
            index += 1
        self.prev_index = index

        root = -1
        cursor = index
        while cursor < prev_count and prev_starts[cursor] <= end:
            label = self.find(prev_labels[cursor])
            root = label if root < 0 else self.union(root, label)
            cursor += 1

        root = self._new_label() if root < 0 else self.find(root)

        self._add_run_stats(root, start, end, scanline)
        self.cur_starts.append(start)
        self.cur_ends.append(end)
        self.cur_labels.append(root)

    def finish_scanline(self) -> None:
        for index in range(len(self.cur_labels)):
            self.cur_labels[index] = self.find(self.cur_labels[index])
        current_roots = set(self.cur_labels)

        for root in {self.find(label) for label in self.active}:
            if root not in current_roots:
                self._close(root)

        self.active = current_roots

        self.prev_starts, self.cur_starts = (
            self.cur_starts,
            self.prev_starts,
        )
        self.prev_ends, self.cur_ends = self.cur_ends, self.prev_ends
        self.prev_labels, self.cur_labels = (
            self.cur_labels,
            self.prev_labels,
        )
        del self.cur_starts[:]
        del self.cur_ends[:]
        del self.cur_labels[:]
        self.prev_index = 0

    def close_active(self) -> None:
        for root in {self.find(label) for label in self.active}:
            self._close(root)
        self.active.clear()
        del self.prev_starts[:]
        del self.prev_ends[:]
        del self.prev_labels[:]
        self.prev_index = 0

    def sort_retained(self) -> None:
        self.retained.sort(
            key=lambda root: (
                -self.sizes[root],
                self.min_y[root],
                self.min_x[root],
                self.max_y[root],
                self.max_x[root],
                self.first_key[root],
            )
        )

    def _start_scanline(self, scanline: int) -> None:
        if self.current_scanline is None:
            self.current_scanline = scanline
            return
        if scanline == self.current_scanline:
            return

        self.finish_scanline()
        if scanline > self.current_scanline + 1:
            self.close_active()

        self.current_scanline = scanline

    def _new_label(self) -> int:
        if self.free_labels:
            label = self.free_labels.pop()
            self.parent[label] = label
            self.sizes[label] = 0
            self.min_x[label] = MAX_COORD
            self.min_y[label] = MAX_COORD
            self.max_x[label] = -1
            self.max_y[label] = -1
            self.sum_x[label] = 0
            self.sum_y[label] = 0
            self.first_key[label] = MAX_COORD
            return label

        if len(self.parent) >= MAX_COMPONENT_LABELS:
            raise ValueError(
                "Component label limit exceeded "
                f"({MAX_COMPONENT_LABELS})"
            )

        label = len(self.parent)
        self.parent.append(label)
        self.sizes.append(0)
        self.min_x.append(MAX_COORD)
        self.min_y.append(MAX_COORD)
        self.max_x.append(-1)
        self.max_y.append(-1)
        self.sum_x.append(0)
        self.sum_y.append(0)
        self.first_key.append(MAX_COORD)
        return label

    def _add_run_stats(
        self,
        root: int,
        start: int,
        end: int,
        scanline: int,
    ) -> None:
        length = end - start
        position_start = self.position_offset + start
        position_end = self.position_offset + end

        self.sizes[root] += length

        if self.horizontal:
            self.min_x[root] = min(self.min_x[root], position_start)
            self.max_x[root] = max(self.max_x[root], position_end)
            self.min_y[root] = min(self.min_y[root], scanline)
            self.max_y[root] = max(self.max_y[root], scanline + 1)
            self.sum_x[root] += (
                (position_start + position_end - 1) * length // 2
            )
            self.sum_y[root] += scanline * length
            key = scanline * self.full_width + position_start
        else:
            self.min_y[root] = min(self.min_y[root], position_start)
            self.max_y[root] = max(self.max_y[root], position_end)
            self.min_x[root] = min(self.min_x[root], scanline)
            self.max_x[root] = max(self.max_x[root], scanline + 1)
            self.sum_y[root] += (
                (position_start + position_end - 1) * length // 2
            )
            self.sum_x[root] += scanline * length
            key = position_start * self.full_width + scanline

        self.first_key[root] = min(self.first_key[root], key)

    def _close(self, root: int) -> None:
        self.component_count += 1
        size = self.sizes[root]
        if size >= self.min_area:
            self.retained.append(root)
            self.retained_pixels += size
        else:
            self.free_labels.append(root)


def _summary(
    tracker: _ComponentTracker,
    region: Region,
) -> dict[str, int | float]:
    x1, y1, x2, y2 = region
    region_pixels = (x2 - x1) * (y2 - y1)
    matched_percentage = (
        round(100 * tracker.matched_pixels / region_pixels, 2)
        if region_pixels
        else 0.0
    )
    return {
        "region_pixels": region_pixels,
        "matched_pixels": tracker.matched_pixels,
        "matched_percentage": matched_percentage,
        "component_count": tracker.component_count,
        "retained_component_count": len(tracker.retained),
        "retained_pixels": tracker.retained_pixels,
    }


def _publish_artifact(
    header: dict[str, Any],
    tracker: _ComponentTracker,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    header_json = json.dumps(
        header,
        ensure_ascii=False,
        indent=2,
    )

    with tempfile.TemporaryDirectory(
        dir=output.parent,
        prefix=".agentkit-",
        ignore_cleanup_errors=True,
    ) as temporary_dir:
        temporary = Path(temporary_dir) / "result.tmp"
        total = 0
        with temporary.open("wb") as stream:
            total = _write_bytes(stream, header_json[:-2], total)
            if not tracker.retained:
                _write_bytes(
                    stream,
                    ',\n  "components": []\n}\n',
                    total,
                )
            else:
                total = _write_bytes(
                    stream,
                    ',\n  "components": [\n',
                    total,
                )

                for index, root in enumerate(tracker.retained):
                    if index:
                        total = _write_bytes(stream, ",\n", total)
                    pixels = tracker.sizes[root]
                    component = {
                        "box": [
                            tracker.min_x[root],
                            tracker.min_y[root],
                            tracker.max_x[root],
                            tracker.max_y[root],
                        ],
                        "pixels": pixels,
                        "centroid": [
                            round(tracker.sum_x[root] / pixels, 2),
                            round(tracker.sum_y[root] / pixels, 2),
                        ],
                    }
                    total = _write_bytes(
                        stream,
                        "    "
                        + json.dumps(
                            component,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        total,
                    )

                _write_bytes(stream, "\n  ]\n}\n", total)

        os.link(temporary, output)


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
