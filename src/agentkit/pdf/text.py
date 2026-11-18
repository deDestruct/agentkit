import json
import math
from contextlib import closing
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypedDict

import pypdfium2 as pdfium

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MiB


class TextBox(TypedDict):
    box: tuple[float, float, float, float]
    text: str


class PageText(TypedDict):
    page: int
    text: str
    char_count: int
    has_text: bool


class PageTextWithBoxes(PageText):
    width_pt: float
    height_pt: float
    rotation: int
    page_box: tuple[float, float, float, float]
    box_count: int
    boxes: list[TextBox]


def _extract_boxes(
    textpage: pdfium.PdfTextPage,
) -> list[TextBox]:
    boxes: list[TextBox] = []

    rect_count = textpage.count_rects()

    if rect_count < 0:
        raise ValueError("Failed to count text rectangles")

    for index in range(rect_count):
        left, bottom, right, top = textpage.get_rect(index)

        box = (left, bottom, right, top)

        if not all(math.isfinite(value) for value in box):
            raise ValueError("Invalid text rectangle coordinates")

        if left > right or bottom > top:
            raise ValueError("Invalid text rectangle boundaries")

        text = textpage.get_text_bounded(
            left=left,
            bottom=bottom,
            right=right,
            top=top,
            errors="replace",
        )

        boxes.append({
            "box": box,
            "text": text,
        })

    return boxes


def _extract_page(
    document: pdfium.PdfDocument,
    page_number: int,
    boxes: bool,
) -> PageText | PageTextWithBoxes:
    with (
        closing(document[page_number - 1]) as page,
        closing(page.get_textpage()) as textpage,
    ):
        char_count = textpage.count_chars()

        if char_count < 0:
            raise ValueError(
                f"Failed to count characters on page {page_number}"
            )

        text = textpage.get_text_bounded(
            errors="replace",
        )

        if not boxes:
            return {
                "page": page_number,
                "text": text,
                "char_count": char_count,
                "has_text": bool(text.strip()),
            }

        width, height = page.get_size()
        rotation = page.get_rotation()
        page_box = page.get_bbox()

        if (
            not math.isfinite(width)
            or not math.isfinite(height)
            or width <= 0
            or height <= 0
        ):
            raise ValueError(
                f"Invalid dimensions on page {page_number}"
            )

        if rotation not in (0, 90, 180, 270):
            raise ValueError(
                f"Invalid rotation on page {page_number}"
            )

        if not all(math.isfinite(value) for value in page_box):
            raise ValueError(
                f"Invalid page boundaries on page {page_number}"
            )

        text_boxes = _extract_boxes(textpage)

        return {
            "page": page_number,
            "text": text,
            "char_count": char_count,
            "has_text": bool(text.strip()),
            "width_pt": width,
            "height_pt": height,
            "rotation": rotation,
            "page_box": page_box,
            "box_count": len(text_boxes),
            "boxes": text_boxes,
        }


def extract_pdf_text(
    source: str | Path,
    output: str | Path,
    page: int | None = None,
    boxes: bool = False,
) -> dict[str, str]:
    source = Path(source).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()

    if not source.is_file():
        raise ValueError(f"Not a regular file: {source}")

    if source == output:
        raise ValueError("Source and output must be different")

    if output.suffix.lower() != ".json":
        raise ValueError("Output must be a JSON file")

    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    if page is not None and (type(page) is not int or page < 1):
        raise ValueError("Page number must be a positive integer")

    with source.open("rb") as stream:
        data = stream.read(MAX_FILE_SIZE + 1)

    size = len(data)

    if size == 0:
        raise ValueError(f"File is empty: {source}")

    if size > MAX_FILE_SIZE:
        raise ValueError(f"File exceeds 100 MiB limit: {source}")

    try:
        with pdfium.PdfDocument(data) as document:
            page_count = len(document)

            if page_count < 1:
                raise ValueError("PDF document contains no pages")

            if page is not None and page > page_count:
                raise ValueError(
                    f"Page {page} is out of range: 1..{page_count}"
                )

            page_numbers = (
                range(page, page + 1)
                if page is not None
                else range(1, page_count + 1)
            )

            pages: list[PageText | PageTextWithBoxes] = []

            for page_number in page_numbers:
                pages.append(
                    _extract_page(
                        document=document,
                        page_number=page_number,
                        boxes=boxes,
                    )
                )

    except pdfium.PdfiumError as exc:
        raise ValueError(
            f"Failed to extract PDF text: {source}"
        ) from exc

    result = {
        "status": "ok",
        "source": str(source),
        "page_count": page_count,
        "count": len(pages),
        "boxes": boxes,
        "coordinate_system": {
            "unit": "pdf_user_space_unit",
            "x_direction": "right",
            "y_direction": "up",
            "box_format": ["left", "bottom", "right", "top"],
        } if boxes else None,
        "pages": pages,
    }

    payload = (
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ) + "\n"
    ).encode("utf-8")

    if len(payload) > MAX_FILE_SIZE:
        raise ValueError("Output JSON exceeds 100 MiB limit")

    output.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(payload)

        output.hardlink_to(temporary_path)

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return {
        "status": "ok",
        "output": str(output),
    }
