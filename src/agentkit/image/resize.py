import math
import os
import tempfile
import warnings
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_PIXELS = 50_000_000

OUTPUT_FORMATS = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}


def resize_image(
    source: str | Path,
    output: str | Path,
    scale: float,
    *,
    quality: int | None = None,
) -> dict[str, str]:
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError("Scale must be a number")
    try:
        scale = float(scale)
    except OverflowError as exc:
        raise ValueError("Scale is too large") from exc
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be finite and positive")

    source = Path(source).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Not a regular file: {source}")

    requested = Path(output).expanduser()
    # Resolve the parent only: keep the final symlink visible.
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
    save_options = {"quality": 95 if quality is None else quality} if output_format == "JPEG" else {}

    # Decode exactly the bytes whose size we checked.
    with source.open("rb") as stream:
        data = stream.read(MAX_FILE_SIZE + 1)
    if not data or len(data) > MAX_FILE_SIZE:
        raise ValueError("Source file must be between 1 byte and 100 MiB")

    try:
        with warnings.catch_warnings(), ExitStack() as stack:
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            buffer = stack.enter_context(BytesIO(data))
            image = stack.enter_context(Image.open(buffer))
            width, height = image.size
            if width < 1 or height < 1 or width * height > MAX_PIXELS:
                raise ValueError("Source image exceeds pixel limit")

            scaled = (width * scale, height * scale)
            if not all(math.isfinite(value) for value in scaled):
                raise ValueError("Resulting dimensions are too large")
            new_width, new_height = map(round, scaled)
            if new_width < 1 or new_height < 1:
                raise ValueError("Resulting dimensions are too small")
            if new_width * new_height > MAX_PIXELS:
                raise ValueError("Resulting image exceeds pixel limit")

            # This operation handles one static image, not an animation.
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Multi-frame images are not supported")
            image.load()

            # Pillow forces NEAREST for palette and bilevel images.
            if image.mode in ("P", "PA"):
                mode = (
                    "RGBA"
                    if image.mode == "PA" or "transparency" in image.info
                    else "RGB"
                )
                image = stack.enter_context(image.convert(mode))
            elif image.mode == "1":
                image = stack.enter_context(image.convert("L"))

            if output_format == "JPEG" and (
                image.mode not in ("RGB", "L")
                or "transparency" in image.info
            ):
                raise ValueError("JPEG requires RGB or L without transparency")

            resized = stack.enter_context(
                image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                dir=output.parent,
                prefix=".agentkit-",
                ignore_cleanup_errors=True,
            ) as temporary_dir:
                temporary = Path(temporary_dir) / "result.tmp"
                resized.save(temporary, format=output_format, **save_options)
                result_size = temporary.stat().st_size
                if not 0 < result_size <= MAX_FILE_SIZE:
                    raise ValueError("Output file must be between 1 byte and 100 MiB")

                # Publish the complete file without replacing any entry.
                # No copy/rename fallback: unsupported hard links are an error.
                os.link(temporary, output)

    except (
        UnidentifiedImageError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError(f"Invalid or unsafe image: {source}") from exc

    return {"status": "ok", "output": str(output)}
