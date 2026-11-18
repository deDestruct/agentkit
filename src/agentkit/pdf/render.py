import json
import math
from contextlib import closing
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import pypdfium2 as pdfium

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MiB
MAX_PIXELS = 50_000_000

DEFAULT_DPI = 150
MIN_DPI = 1
MAX_DPI = 600


def _write_file(output: Path, data: bytes) -> None:
    if len(data) > MAX_FILE_SIZE:
        raise ValueError(f"Output exceeds 100 MiB limit: {output}")

    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(data)

        output.hardlink_to(temporary_path)

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _render_page(
    document: pdfium.PdfDocument,
    page_number: int,
    dpi: int,
    output: Path,
) -> dict[str, str | int]:
    scale = dpi / 72

    with closing(document[page_number - 1]) as page:
        width_pt, height_pt = page.get_size()

        if (
            not math.isfinite(width_pt)
            or not math.isfinite(height_pt)
            or width_pt <= 0
            or height_pt <= 0
        ):
            raise ValueError(
                f"Invalid dimensions on page {page_number}"
            )

        width = math.ceil(width_pt * scale)
        height = math.ceil(height_pt * scale)

        if width * height > MAX_PIXELS:
            raise ValueError(
                f"Page {page_number} exceeds "
                f"{MAX_PIXELS} pixels at {dpi} DPI"
            )

        with (
            closing(
                page.render(scale=scale)  # pyright: ignore[reportArgumentType]
            ) as bitmap,
            bitmap.to_pil() as image,
            BytesIO() as buffer,
        ):
            width, height = image.size

            if width * height > MAX_PIXELS:
                raise ValueError(
                    f"Page {page_number} exceeds {MAX_PIXELS} pixels"
                )

            image.save(
                buffer,
                format="PNG",
                dpi=(dpi, dpi),
            )

            size = buffer.tell()

            if size > MAX_FILE_SIZE:
                raise ValueError(
                    f"Page {page_number} exceeds 100 MiB limit"
                )

            data = buffer.getvalue()

    _write_file(output, data)

    return {
        "page": page_number,
        "output": str(output),
        "width": width,
        "height": height,
        "dpi": dpi,
        "size_bytes": size,
    }


def render_pdf(
    source: str | Path,
    output: str | Path,
    dpi: int = DEFAULT_DPI,
    page: int | None = None,
    *,
    original_source: Path | None = None,
) -> dict[str, str]:
    source = Path(source).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()

    if not source.is_file():
        raise ValueError(f"Not a regular file: {source}")

    if source == output:
        raise ValueError("Source and output must be different")

    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    if type(dpi) is not int or not MIN_DPI <= dpi <= MAX_DPI:
        raise ValueError(
            f"DPI must be an integer between {MIN_DPI} and {MAX_DPI}"
        )

    if page is not None:
        if type(page) is not int or page < 1:
            raise ValueError("Page number must be a positive integer")

        if output.suffix.lower() != ".png":
            raise ValueError("Output must be a PNG file")

        result_path = output.with_suffix(".json")

        if result_path == source:
            raise ValueError("Source and result must be different")

        if result_path.exists():
            raise FileExistsError(
                f"Output already exists: {result_path}"
            )
    else:
        result_path = output / "result.json"

    with source.open("rb") as stream:
        data = stream.read(MAX_FILE_SIZE + 1)

    size = len(data)

    if size == 0:
        raise ValueError(f"File is empty: {source}")

    if size > MAX_FILE_SIZE:
        raise ValueError(f"File exceeds 100 MiB limit: {source}")

    created_files: list[Path] = []
    created_directory = False

    try:
        with pdfium.PdfDocument(data) as document:
            page_count = len(document)

            if page_count < 1:
                raise ValueError("PDF document contains no pages")

            if page is not None and page > page_count:
                raise ValueError(
                    f"Page {page} is out of range: 1..{page_count}"
                )

            if page is not None:
                output.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                page_numbers = range(page, page + 1)

            else:
                output.mkdir(
                    parents=True,
                    exist_ok=False,
                )

                created_directory = True
                page_numbers = range(1, page_count + 1)

            rendered_pages: list[dict[str, str | int]] = []

            for page_number in page_numbers:
                image_path = (
                    output
                    if page is not None
                    else output / f"page_{page_number:04d}.png"
                )

                rendered = _render_page(
                    document=document,
                    page_number=page_number,
                    dpi=dpi,
                    output=image_path,
                )

                created_files.append(image_path)
                rendered_pages.append(rendered)

        result = {
            "status": "ok",
            "source": str(
                original_source
                if original_source is not None
                else source
            ),
            "format": "PNG",
            "dpi": dpi,
            "page_count": page_count,
            "count": len(rendered_pages),
            "pages": rendered_pages,
        }

        payload = (
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ) + "\n"
        ).encode("utf-8")

        _write_file(result_path, payload)
        created_files.append(result_path)

    except BaseException as exc:
        for created_path in reversed(created_files):
            created_path.unlink(missing_ok=True)

        if created_directory:
            output.rmdir()

        if isinstance(exc, pdfium.PdfiumError):
            raise ValueError(
                f"Failed to process PDF: {source}"
            ) from exc

        raise

    return {
        "status": "ok",
        "output": str(output),
        "result": str(result_path),
    }
