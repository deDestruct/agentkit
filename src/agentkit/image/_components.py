"""Shared connected-component engine.

The tracker consumes horizontal runs of set pixels. It is independent of how
those runs are produced, so both color search and image comparison use the same
union-find, statistics, filtering and ordering behavior.
"""

from __future__ import annotations

import json
import re
from array import array
from collections.abc import Iterator
from typing import Any

from PIL import Image

MAX_COORD = 2**31 - 1
RUN_PATTERN = re.compile(rb"[^\x00]+")


def scan_mask(
    mask: Image.Image,
    tracker: ComponentTracker,
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


class ComponentTracker:
    def __init__(
        self,
        *,
        full_width: int,
        min_area: int,
        horizontal: bool,
        position_offset: int,
        max_labels: int,
    ) -> None:
        self.full_width = full_width
        self.min_area = min_area
        self.horizontal = horizontal
        self.position_offset = position_offset
        self.max_labels = max_labels

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

        if len(self.parent) >= self.max_labels:
            raise ValueError(
                "Component label limit exceeded "
                f"({self.max_labels})"
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


def iter_component_artifact(
    header: dict[str, Any],
    tracker: ComponentTracker,
) -> Iterator[str]:
    """Yield the JSON artifact in the shared line-oriented format."""
    header_json = json.dumps(
        header,
        ensure_ascii=False,
        indent=2,
    )
    yield header_json[:-2]

    if not tracker.retained:
        yield ',\n  "components": []\n}\n'
        return

    yield ',\n  "components": [\n'

    for index, root in enumerate(tracker.retained):
        if index:
            yield ",\n"
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
        yield (
            "    "
            + json.dumps(
                component,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    yield "\n  ]\n}\n"
