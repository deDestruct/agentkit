from __future__ import annotations

import math
from contextlib import suppress
from pathlib import Path
from tempfile import mkdtemp
from typing import Literal

from agentkit.docx._backends._base import (
    BackendUnavailableError,
    ConversionBackend,
)
from agentkit.docx._backends.word import WordBackend
from agentkit.pdf.render import (
    DEFAULT_DPI,
    MAX_DPI,
    MIN_DPI,
    render_pdf,
)

MAX_FILE_SIZE = 100 * 1024 * 1024
DEFAULT_TIMEOUT = 120.0

type BackendName = Literal["auto", "word"]


def _select_backend(
    name: BackendName,
) -> ConversionBackend:
    if name not in ("auto", "word"):
        raise ValueError(
            f"Unsupported DOCX backend: {name}"
        )

    backend = WordBackend()

    if backend.is_available():
        return backend

    raise BackendUnavailableError(
        "No available DOCX rendering backend. "
        "Currently supported: Microsoft Word on Windows."
    )


def render_docx(
    source: str | Path,
    output: str | Path,
    dpi: int = DEFAULT_DPI,
    page: int | None = None,
    *,
    backend: BackendName = "auto",
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, str]:
    source = Path(source).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()

    if not source.is_file():
        raise ValueError(
            f"Not a regular file: {source}"
        )

    if source.suffix.lower() != ".docx":
        raise ValueError(
            "Source must be a DOCX file"
        )

    if source == output:
        raise ValueError(
            "Source and output must be different"
        )

    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}"
        )

    if type(dpi) is not int or not MIN_DPI <= dpi <= MAX_DPI:
        raise ValueError(
            f"DPI must be an integer between "
            f"{MIN_DPI} and {MAX_DPI}"
        )

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(
            "Timeout must be finite and positive"
        )

    if page is not None:
        if type(page) is not int or page < 1:
            raise ValueError(
                "Page number must be a positive integer"
            )

        if output.suffix.lower() != ".png":
            raise ValueError(
                "Output must be a PNG file"
            )

        result_path = output.with_suffix(".json")

        if result_path.exists():
            raise FileExistsError(
                f"Output already exists: {result_path}"
            )

    with source.open("rb") as stream:
        data = stream.read(MAX_FILE_SIZE + 1)

    if not data:
        raise ValueError(
            f"File is empty: {source}"
        )

    if len(data) > MAX_FILE_SIZE:
        raise ValueError(
            f"File exceeds 100 MiB limit: {source}"
        )

    selected_backend = _select_backend(backend)

    workspace = Path(
        mkdtemp(prefix="agentkit-docx-")
    )

    temporary_source = workspace / "document.docx"
    temporary_pdf = workspace / "document.pdf"

    try:
        with temporary_source.open("xb") as stream:
            stream.write(data)

        del data

        selected_backend.convert_to_pdf(
            source=temporary_source,
            output=temporary_pdf,
            timeout=timeout,
        )

        result = render_pdf(
            source=temporary_pdf,
            output=output,
            dpi=dpi,
            page=page,
            original_source=source,
        )

    finally:
        for path in (temporary_pdf, temporary_source):
            with suppress(OSError):
                path.unlink(missing_ok=True)

        with suppress(OSError):
            workspace.rmdir()

    return {
        **result,
        "backend": selected_backend.name,
    }
