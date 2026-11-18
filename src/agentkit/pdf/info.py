import json
import math
from contextlib import closing
from pathlib import Path
from tempfile import NamedTemporaryFile

import pypdfium2 as pdfium

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MiB


def get_pdf_info(
    source: str | Path,
    output: str | Path,
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

            pages = []

            for index in range(page_count):
                with closing(document[index]) as page:
                    width, height = page.get_size()
                    rotation = page.get_rotation()

                    if (
                        not math.isfinite(width)
                        or not math.isfinite(height)
                        or width <= 0
                        or height <= 0
                    ):
                        raise ValueError(
                            f"Invalid dimensions on page {index + 1}"
                        )

                    if rotation not in (0, 90, 180, 270):
                        raise ValueError(
                            f"Invalid rotation on page {index + 1}"
                        )

                    if width > height:
                        orientation = "landscape"
                    elif height > width:
                        orientation = "portrait"
                    else:
                        orientation = "square"

                    with closing(page.get_textpage()) as textpage:
                        char_count = textpage.count_chars()

                    if char_count < 0:
                        raise ValueError(
                            f"Failed to count characters on page {index + 1}"
                        )

                    pages.append({
                        "page": index + 1,
                        "width_pt": width,
                        "height_pt": height,
                        "orientation": orientation,
                        "rotation": rotation,
                        "has_text": char_count > 0,
                        "char_count": char_count,
                    })

            result = {
                "path": str(source),
                "format": "PDF",
                "size_bytes": size,
                "page_count": page_count,
                "pdf_version": document.get_version(),
                "metadata": document.get_metadata_dict(skip_empty=True),
                "pages": pages,
            }

    except pdfium.PdfiumError as exc:
        raise ValueError(
            f"Invalid, unsupported or password-protected PDF: {source}"
        ) from exc

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
