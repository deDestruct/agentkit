import argparse
import json
import os
import tempfile
import warnings
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_PIXELS = 50_000_000

SHEET_WIDTH = 1920
SHEET_HEIGHT = 1080
COLUMNS = 4
ROWS = 3
GAP = 20
LABEL_HEIGHT = 30
BACKGROUND = "white"

INPUT_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif",
}


def _collect_sources(
    sources: str | Path | list[str | Path],
) -> list[Path]:
    if isinstance(sources, (str, Path)):
        sources = [sources]
    if not sources:
        raise ValueError("Provide a directory or at least one image path")

    paths = [Path(path).expanduser().resolve(strict=True) for path in sources]
    if len(paths) == 1 and paths[0].is_dir():
        paths = sorted(
            (
                path for path in paths[0].iterdir()
                if path.is_file() and path.suffix.lower() in INPUT_EXTENSIONS
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    if not paths:
        raise ValueError("Directory contains no supported images")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Expected an image file: {path}")
    return paths


def _make_thumbnail(path: Path, size: tuple[int, int]) -> Image.Image:
    with path.open("rb") as stream:
        data = stream.read(MAX_FILE_SIZE + 1)
    if not 0 < len(data) <= MAX_FILE_SIZE:
        raise ValueError(f"Source must be between 1 byte and 100 MiB: {path}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with BytesIO(data) as buffer, Image.open(buffer) as image:
                width, height = image.size
                if width < 1 or height < 1 or width * height > MAX_PIXELS:
                    raise ValueError(f"Invalid dimensions or pixel limit exceeded: {path}")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError(f"Multi-frame images are not supported: {path}")
                image.load()

                ratio = min(size[0] / width, size[1] / height)
                target = (
                    max(1, min(size[0], round(width * ratio))),
                    max(1, min(size[1], round(height * ratio))),
                )
                # RGBA preserves transparency and enables LANCZOS for palettes.
                with image.convert("RGBA") as rgba:
                    return rgba.resize(target, Image.Resampling.LANCZOS)
    except (
        UnidentifiedImageError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError(f"Invalid or unsafe image: {path}") from exc


def sheet_images(
    sources: str | Path | list[str | Path],
    output: str | Path,
) -> dict[str, object]:
    """Create numbered PNG contact sheets in the output directory.

    A directory input is scanned non-recursively in filename order.
    Explicit file paths retain their supplied order. EXIF rotation is not applied.
    Each sheet is published without overwriting existing files. Publication of
    the whole batch is not atomic: an I/O failure may leave completed sheets.
    """
    paths = _collect_sources(sources)

    if min(SHEET_WIDTH, SHEET_HEIGHT, COLUMNS, ROWS) < 1:
        raise ValueError("Sheet dimensions and grid counts must be positive")
    if SHEET_WIDTH * SHEET_HEIGHT > MAX_PIXELS:
        raise ValueError("Sheet exceeds pixel limit")
    if GAP < 0 or LABEL_HEIGHT < 1:
        raise ValueError("Invalid sheet spacing")

    cell_width = (SHEET_WIDTH - GAP * (COLUMNS + 1)) // COLUMNS
    cell_height = (SHEET_HEIGHT - GAP * (ROWS + 1)) // ROWS
    image_height = cell_height - LABEL_HEIGHT
    if cell_width < 1 or image_height < 1:
        raise ValueError("Sheet is too small for the configured grid")

    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_sheet = COLUMNS * ROWS
    sheet_count = (len(paths) + per_sheet - 1) // per_sheet
    destinations = [
        output / f"sheet_{number:04d}.png"
        for number in range(1, sheet_count + 1)
    ]
    for destination in destinations:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Output already exists: {destination}")

    font = ImageFont.load_default(size=20)
    sheets = []
    with tempfile.TemporaryDirectory(
        dir=output, prefix=".agentkit-", ignore_cleanup_errors=True,
    ) as temporary_dir:
        # Finish and validate all sheets before publishing any of them.
        for page, destination in enumerate(destinations):
            start = page * per_sheet
            entries = []
            with Image.new("RGB", (SHEET_WIDTH, SHEET_HEIGHT), BACKGROUND) as canvas:
                draw = ImageDraw.Draw(canvas)
                for slot, path in enumerate(paths[start:start + per_sheet]):
                    row, column = divmod(slot, COLUMNS)
                    left = GAP + column * (cell_width + GAP)
                    top = GAP + row * (cell_height + GAP)
                    index = start + slot + 1

                    with _make_thumbnail(path, (cell_width, image_height)) as thumb:
                        x = left + (cell_width - thumb.width) // 2
                        y = top + (image_height - thumb.height) // 2
                        canvas.paste(thumb, (x, y), thumb)
                        box = [x, y, x + thumb.width, y + thumb.height]

                    draw.text(
                        (left, top + image_height + 4),
                        str(index), font=font, fill="black",
                    )
                    entries.append({"index": index, "source": str(path), "box": box})

                temporary = Path(temporary_dir) / destination.name
                canvas.save(temporary, format="PNG")

            if not 0 < temporary.stat().st_size <= MAX_FILE_SIZE:
                raise ValueError(f"Sheet exceeds 100 MiB limit: {destination.name}")
            sheets.append({
                "output": str(destination),
                "width": SHEET_WIDTH,
                "height": SHEET_HEIGHT,
                "images": entries,
            })

        published = []
        for destination in destinations:
            try:
                os.link(Path(temporary_dir) / destination.name, destination)
            except OSError as exc:
                # Never remove destination paths: they may belong to another process.
                raise OSError(
                    f"Could not publish {destination}: {exc}. "
                    f"Already published: {published}"
                ) from exc
            published.append(str(destination))

    return {"status": "ok", "count": len(paths), "sheets": sheets}
