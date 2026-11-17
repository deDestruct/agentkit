import os
import tempfile
import warnings
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 МиБ
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
) -> dict[str, str]:
    # Координаты: (left, top, right, bottom).
    # Правая и нижняя границы не включаются.
    if (
        not isinstance(box, (tuple, list))
        or len(box) != 4
        or any(type(value) is not int for value in box)
    ):
        raise TypeError("Box must contain exactly four integers")

    x1, y1, x2, y2 = box

    if x1 < 0 or y1 < 0:
        raise ValueError("Coordinates cannot be negative")

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid crop rectangle")

    source = Path(source).expanduser().resolve(strict=True)

    if not source.is_file():
        raise ValueError(f"Not a regular file: {source}")

    requested = Path(output).expanduser()

    # не раскрываем символическую ссылку в конечном имени.
    output = requested.parent.resolve() / requested.name

    if source == output:
        raise ValueError("Source and output must be different")

    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")

    output_format = OUTPUT_FORMATS.get(output.suffix.lower())

    if output_format is None:
        raise ValueError(
            f"Unsupported output format: {output.suffix}"
        )

    # Pillow получит именно те байты, размер которых мы проверили.
    with source.open("rb") as stream:
        data = stream.read(MAX_FILE_SIZE + 1)

    if not data:
        raise ValueError(f"File is empty: {source}")

    if len(data) > MAX_FILE_SIZE:
        raise ValueError(f"Source file exceeds 100 MiB: {source}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning,
            )

            with BytesIO(data) as buffer, Image.open(buffer) as image:
                width, height = image.size

                if width < 1 or height < 1:
                    raise ValueError("Invalid source image dimensions")

                if width * height > MAX_PIXELS:
                    raise ValueError(
                        f"Source image exceeds {MAX_PIXELS} pixels"
                    )

                if x2 > width or y2 > height:
                    raise ValueError(
                        f"Crop {(x1, y1, x2, y2)} exceeds "
                        f"image bounds ({width}x{height})"
                    )

                # не допускаем молчаливую потерю остальных кадров.
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError(
                        "Multi-frame images are not supported"
                    )

                image.load()

                with image.crop((x1, y1, x2, y2)) as cropped:
                    if output_format == "JPEG" and (
                        cropped.mode not in ("RGB", "L")
                        or "transparency" in cropped.info
                    ):
                        raise ValueError(
                            "JPEG requires RGB or grayscale "
                            "without transparency"
                        )

                    output.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    with tempfile.TemporaryDirectory(
                        dir=output.parent,
                        prefix=".agentkit-",
                        ignore_cleanup_errors=True,
                    ) as temporary_dir:
                        temporary = Path(temporary_dir) / "result.tmp"

                        cropped.save(
                            temporary,
                            format=output_format,
                        )

                        result_size = temporary.stat().st_size

                        if not 0 < result_size <= MAX_FILE_SIZE:
                            raise ValueError(
                                "Output file must be between "
                                "1 byte and 100 MiB"
                            )

                        # Публикуем уже готовый файл.
                        # Если output появился во время обработки,
                        # операция завершится ошибкой без перезаписи.
                        os.link(temporary, output)

    except (
        UnidentifiedImageError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError(
            f"Invalid or unsafe image: {source}"
        ) from exc

    return {
        "status": "ok",
        "output": str(output),
    }
