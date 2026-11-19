import os
import tempfile
import warnings
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_PIXELS = 50_000_000
ANGLES = (90, 180, 270)
OUTPUT_FORMATS = {
    ".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG",
    ".webp": "WEBP", ".bmp": "BMP", ".tif": "TIFF", ".tiff": "TIFF",
}

# Pillow rotates counter-clockwise, so clockwise angles map to the inverse
# constants. Arbitrary angles and interpolation are intentionally unsupported.
TRANSPOSE = {
    90: Image.Transpose.ROTATE_270,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_90,
}


def rotate_image(
    source: str | Path,
    output: str | Path,
    angle: int,
    *,
    quality: int | None = None,
) -> dict[str, str]:
    if type(angle) is not int:
        raise TypeError("Angle must be an integer")
    if angle not in ANGLES:
        raise ValueError("Angle must be one of: 90, 180, 270")

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
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Multi-frame images are not supported")
                image.load()
                with image.transpose(TRANSPOSE[angle]) as rotated:
                    if rotated.width < 1 or rotated.height < 1:
                        raise ValueError("Invalid rotation result")
                    if output_format == "JPEG" and (
                        rotated.mode not in ("RGB", "L")
                        or "transparency" in rotated.info
                    ):
                        raise ValueError("JPEG requires RGB or grayscale without transparency")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(
                        dir=output.parent, prefix=".agentkit-", ignore_cleanup_errors=True,
                    ) as temporary_dir:
                        temporary = Path(temporary_dir) / "result.tmp"
                        rotated.save(temporary, format=output_format, **save_options)
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
