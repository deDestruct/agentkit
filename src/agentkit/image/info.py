from pathlib import Path

from PIL import Image, UnidentifiedImageError

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 МБ


def get_image_info(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"File does not exist or is not a regular file: {path}"
        )

    size = path.stat().st_size

    if size == 0:
        raise ValueError(f"File is empty: {path}")

    if size > MAX_FILE_SIZE:
        raise ValueError(
            f"File exceeds maximum size of 100 MiB: {path}"
        )

    try:
        with Image.open(path) as image:
            return {
                "path": str(path),
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode,
                "size_bytes": size,
            }

    except UnidentifiedImageError as exc:
        raise ValueError(
            f"Unsupported or invalid image: {path}"
        ) from exc
