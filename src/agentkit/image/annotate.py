import os
import tempfile
import warnings
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, UnidentifiedImageError

MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_PIXELS = 50_000_000
STROKE_WIDTH = 4

SHAPES = ("rectangle", "line")
PALETTE = {
    "red": "#FF0000",
    "green": "#00C853",
    "blue": "#2962FF",
    "yellow": "#FFD600",
    "orange": "#FF6D00",
    "purple": "#AA00FF",
    "black": "#000000",
    "white": "#FFFFFF",
}
OUTPUT_FORMATS = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}


def annotate_image(
    source: str | Path,
    output: str | Path,
    shape: str,
    coordinates: tuple[int, int, int, int],
    color: str,
    *,
    quality: int | None = None,
) -> dict[str, str]:
    if not isinstance(shape, str) or shape not in SHAPES:
        raise ValueError("Shape must be rectangle or line")
    if not isinstance(color, str) or color not in PALETTE:
        raise ValueError(f"Color must be one of: {', '.join(PALETTE)}")
    if (
        not isinstance(coordinates, (tuple, list))
        or len(coordinates) != 4
        or any(type(value) is not int for value in coordinates)
    ):
        raise TypeError("Coordinates must contain exactly four integers")

    x1, y1, x2, y2 = coordinates
    if min(x1, y1, x2, y2) < 0:
        raise ValueError("Coordinates cannot be negative")
    if shape == "rectangle" and (x2 <= x1 or y2 <= y1):
        raise ValueError("Rectangle requires x2 > x1 and y2 > y1")
    if shape == "line" and (x1, y1) == (x2, y2):
        raise ValueError("Line endpoints must be different")
    if type(STROKE_WIDTH) is not int or STROKE_WIDTH < 1:
        raise ValueError("Stroke width must be a positive integer")

    source = Path(source).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Not a regular file: {source}")

    requested = Path(output).expanduser()
    output = requested.parent.resolve() / requested.name
    if source == output:
        raise ValueError("Source and output must be different")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    output_format = OUTPUT_FORMATS.get(output.suffix.lower())
    if output_format is None:
        raise ValueError(f"Unsupported output format: {output.suffix}")
    if quality is not None:
        if type(quality) is not int:
            raise TypeError("JPEG quality must be an integer")
        if not 1 <= quality <= 95:
            raise ValueError("JPEG quality must be between 1 and 95")
        if output_format != "JPEG":
            raise ValueError("Quality is supported only for JPEG output")
    save_options = {"quality": 90 if quality is None else quality} if output_format == "JPEG" else {}

    with source.open("rb") as stream:
        data = stream.read(MAX_FILE_SIZE + 1)
    if not 0 < len(data) <= MAX_FILE_SIZE:
        raise ValueError("Source file must be between 1 byte and 100 MiB")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with BytesIO(data) as buffer, Image.open(buffer) as image:
                width, height = image.size
                if width < 1 or height < 1 or width * height > MAX_PIXELS:
                    raise ValueError("Invalid dimensions or source pixel limit exceeded")
                # Rectangles use crop's exclusive right/bottom boundaries.
                if shape == "rectangle" and (x2 > width or y2 > height):
                    raise ValueError(f"Rectangle exceeds image bounds ({width}x{height})")
                if shape == "line" and (max(x1, x2) >= width or max(y1, y2) >= height):
                    raise ValueError(
                        f"Coordinates must be within 0..{width - 1}, 0..{height - 1}"
                    )
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Multi-frame images are not supported")
                image.load()

                has_alpha = "A" in image.getbands() or "transparency" in image.info
                if has_alpha and output_format in ("JPEG", "BMP"):
                    raise ValueError("Use PNG, WEBP or TIFF to preserve transparency")

                # Convert grayscale/palette images so every palette color is usable.
                with image.convert("RGBA" if has_alpha else "RGB") as annotated:
                    draw = ImageDraw.Draw(annotated)
                    points = (x1, y1, x2, y2)
                    if shape == "rectangle":
                        draw.rectangle(
                            (x1, y1, x2 - 1, y2 - 1), fill=None,
                            outline=PALETTE[color],
                            width=min(STROKE_WIDTH, (min(x2 - x1, y2 - y1) + 1) // 2),
                        )
                    else:
                        draw.line(points, fill=PALETTE[color], width=STROKE_WIDTH)

                    output.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(
                        dir=output.parent,
                        prefix=".agentkit-",
                        ignore_cleanup_errors=True,
                    ) as temporary_dir:
                        temporary = Path(temporary_dir) / "result.tmp"
                        annotated.save(temporary, format=output_format, **save_options)
                        if not 0 < temporary.stat().st_size <= MAX_FILE_SIZE:
                            raise ValueError("Output file must be between 1 byte and 100 MiB")
                        os.link(temporary, output)

    except (
        UnidentifiedImageError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError(f"Invalid or unsafe image: {source}") from exc

    return {"status": "ok", "output": str(output)}
