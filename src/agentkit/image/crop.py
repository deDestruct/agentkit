import warnings
from pathlib import Path

from PIL import Image, UnidentifiedImageError

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 МБ
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


def crop_image(
    source: str | Path,
    output: str | Path,
    box: tuple[int, int, int, int],
) -> dict:
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()

    # 1. Validate source
    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")

    if not source.is_file():
        raise ValueError(f"Not a regular file: {source}")

    size = source.stat().st_size

    if size == 0:
        raise ValueError(f"File is empty: {source}")

    if size > MAX_FILE_SIZE:
        raise ValueError(
            f"File exceeds 100 MB limit: {source}"
        )

    # 2. Validate output
    if source == output:
        raise ValueError("Source and output must be different")

    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}"
        )

    output_format = OUTPUT_FORMATS.get(output.suffix.lower())

    if output_format is None:
        raise ValueError(
            f"Unsupported output format: {output.suffix}"
        )

    # 3. Validate coordinates
    if len(box) != 4 or any(
        type(value) is not int for value in box
    ):
        raise TypeError("Box must contain exactly four integers")

    x1, y1, x2, y2 = box

    if x1 < 0 or y1 < 0:
        raise ValueError("Coordinates cannot be negative")

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid crop rectangle")

    # 4. Open and validate image
    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning,
            )

            with Image.open(source) as image:
                width, height = image.size

                if width * height > MAX_PIXELS:
                    raise ValueError(
                        f"Image exceeds {MAX_PIXELS} pixels"
                    )

                if x2 > width or y2 > height:
                    raise ValueError(
                        f"Crop {box} exceeds image bounds "
                        f"({width}x{height})"
                    )

                # Decode image data before writing
                image.load()

                cropped = image.crop(box)

                if (
                    output_format == "JPEG"
                    and cropped.mode not in ("RGB", "L")
                ):
                    raise ValueError(
                        "JPEG requires RGB or grayscale image"
                    )

                output.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                # Exclusive creation: do not overwrite files
                with output.open("xb") as file:
                    try:
                        cropped.save(
                            file,
                            format=output_format,
                        )
                        return  {
                            "status": "ok",
                            "output": str(output),
                                }
                    
                    except Exception:
                        file.close()
                        output.unlink(missing_ok=True)
                        raise

    except (
        UnidentifiedImageError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError(
            f"Invalid or unsafe image: {source}"
        ) from exc
