import warnings
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 МиБ
MAX_PIXELS = 50_000_000


def get_image_info(path: str | Path) -> dict[str, str | int | None]:
    path = Path(path).expanduser().resolve(strict=True)

    if not path.is_file():
        raise ValueError(f"Not a regular file: {path}")

    # Читаем ограниченное количество байт и проверяем именно
    # те данные, из которых затем получаем метаданные.
    with path.open("rb") as stream:
        data = stream.read(MAX_FILE_SIZE + 1)

    size = len(data)

    if size == 0:
        raise ValueError(f"File is empty: {path}")

    if size > MAX_FILE_SIZE:
        raise ValueError(f"File exceeds 100 MiB limit: {path}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning,
            )

            with BytesIO(data) as buffer, Image.open(buffer) as image:
                width, height = image.size

                if width < 1 or height < 1:
                    raise ValueError("Invalid image dimensions")

                if width * height > MAX_PIXELS:
                    raise ValueError(
                        f"Image exceeds {MAX_PIXELS} pixels"
                    )

                return {
                    "path": str(path),
                    "width": width,
                    "height": height,
                    "format": image.format,
                    "mode": image.mode,
                    "size_bytes": size,
                }

    except (
        UnidentifiedImageError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError(
            f"Unsupported, invalid or unsafe image: {path}"
        ) from exc
