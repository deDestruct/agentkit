import os
import tempfile
import warnings
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFont, UnidentifiedImageError

MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_PIXELS = 50_000_000

DEFAULT_STEP = 100
DEFAULT_OPACITY = 35

LINE_WIDTH = 1
LABEL_FONT_SIZE = 12
LABEL_MARGIN = 2
LABEL_PADDING = 2
LABEL_GAP = 2

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

Box = tuple[int, int, int, int]
Label = tuple[str, Box]
Font = ImageFont.FreeTypeFont | ImageFont.ImageFont


def _text_size(font: Font, text: str) -> tuple[int, int]:
    left, top, right, bottom = font.getbbox(text)
    return round(right - left), round(bottom - top)


def _boxes_overlap(first: Box, second: Box, gap: int) -> bool:
    return (
        first[0] < second[2] + gap
        and second[0] < first[2] + gap
        and first[1] < second[3] + gap
        and second[1] < first[3] + gap
    )


def _label_layout(
    width: int,
    height: int,
    step: int,
    font: Font,
) -> list[Label]:
    labels: list[Label] = []
    occupied: list[Box] = []

    def add(text: str, left: int, top: int) -> None:
        text_width, text_height = _text_size(font, text)
        box_width = text_width + LABEL_PADDING * 2
        box_height = text_height + LABEL_PADDING * 2
        if (
            box_width + LABEL_MARGIN * 2 > width
            or box_height + LABEL_MARGIN * 2 > height
        ):
            return

        left = min(max(left, LABEL_MARGIN), width - box_width - LABEL_MARGIN)
        top = min(max(top, LABEL_MARGIN), height - box_height - LABEL_MARGIN)
        box = (left, top, left + box_width, top + box_height)
        if any(_boxes_overlap(box, other, LABEL_GAP) for other in occupied):
            return

        occupied.append(box)
        labels.append((text, box))

    for x in range(0, width, step):
        text = str(x)
        text_width, _ = _text_size(font, text)
        box_width = text_width + LABEL_PADDING * 2
        add(text, x - box_width // 2, LABEL_MARGIN)

    # The top-left label already marks the origin, so Y labels start at step.
    for y in range(step, height, step):
        text = str(y)
        _, text_height = _text_size(font, text)
        box_height = text_height + LABEL_PADDING * 2
        add(text, LABEL_MARGIN, y - box_height // 2)

    return labels


def _draw_label(
    draw: ImageDraw.ImageDraw,
    font: Font,
    text: str,
    box: Box,
    *,
    plate_fill: tuple[int, int, int, int],
    text_fill: tuple[int, int, int, int],
) -> None:
    draw.rectangle(box, fill=plate_fill)
    left, top, _, _ = font.getbbox(text)
    draw.text(
        (box[0] + LABEL_PADDING - left, box[1] + LABEL_PADDING - top),
        text,
        font=font,
        fill=text_fill,
    )


def _grid_overlay(
    width: int,
    height: int,
    *,
    step: int,
    color: str,
    opacity: int,
    labels: bool,
) -> Image.Image:
    alpha = round(255 * opacity / 100)
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line_fill = (*ImageColor.getrgb(PALETTE[color]), alpha)

    for x in range(0, width, step):
        draw.line((x, 0, x, height - 1), fill=line_fill, width=LINE_WIDTH)
    for y in range(0, height, step):
        draw.line((0, y, width - 1, y), fill=line_fill, width=LINE_WIDTH)

    if labels:
        font = ImageFont.load_default(size=LABEL_FONT_SIZE)
        plate_fill = (0, 0, 0, alpha)
        text_fill = (255, 255, 255, alpha)
        for text, box in _label_layout(width, height, step, font):
            _draw_label(
                draw,
                font,
                text,
                box,
                plate_fill=plate_fill,
                text_fill=text_fill,
            )

    return overlay


def _save_image(
    image: Image.Image,
    output: Path,
    output_format: str,
    save_options: dict[str, int],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent,
        prefix=".agentkit-",
        ignore_cleanup_errors=True,
    ) as temporary_dir:
        temporary = Path(temporary_dir) / "result.tmp"
        image.save(temporary, format=output_format, **save_options)
        if not 0 < temporary.stat().st_size <= MAX_FILE_SIZE:
            raise ValueError("Output file must be between 1 byte and 100 MiB")
        os.link(temporary, output)


def grid_image(
    source: str | Path,
    output: str | Path,
    *,
    step: int = DEFAULT_STEP,
    color: str = "red",
    opacity: int = DEFAULT_OPACITY,
    labels: bool = True,
    quality: int | None = None,
) -> dict[str, str]:
    if type(step) is not int:
        raise TypeError("Step must be an integer")
    if step < 1:
        raise ValueError("Step must be a positive integer")
    if type(opacity) is not int:
        raise TypeError("Opacity must be an integer")
    if not 0 <= opacity <= 100:
        raise ValueError("Opacity must be between 0 and 100")
    if not isinstance(color, str) or color not in PALETTE:
        raise ValueError(f"Color must be one of: {', '.join(PALETTE)}")

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
    save_options = (
        {"quality": 90 if quality is None else quality}
        if output_format == "JPEG"
        else {}
    )

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
                    raise ValueError(
                        "Invalid dimensions or source pixel limit exceeded"
                    )
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Multi-frame images are not supported")
                image.load()

                has_alpha = (
                    "A" in image.getbands() or "transparency" in image.info
                )
                if has_alpha and output_format in ("JPEG", "BMP"):
                    raise ValueError(
                        "Use PNG, WEBP or TIFF to preserve transparency"
                    )

                if (
                    opacity == 0
                    and output_format == "PNG"
                    and image.format == "PNG"
                ):
                    # No overlay and no format conversion: keep the original mode.
                    _save_image(image, output, output_format, save_options)
                else:
                    with image.convert("RGBA" if has_alpha else "RGB") as base:
                        if opacity == 0:
                            # No overlay: preserve decoded pixels.
                            _save_image(base, output, output_format, save_options)
                        else:
                            overlay = _grid_overlay(
                                width,
                                height,
                                step=step,
                                color=color,
                                opacity=opacity,
                                labels=labels,
                            )
                            try:
                                with (
                                    base.convert("RGBA") as canvas,
                                    Image.alpha_composite(
                                        canvas, overlay
                                    ) as composed,
                                ):
                                    if has_alpha:
                                        _save_image(
                                            composed,
                                            output,
                                            output_format,
                                            save_options,
                                        )
                                    else:
                                        with composed.convert("RGB") as result:
                                            _save_image(
                                                result,
                                                output,
                                                output_format,
                                                save_options,
                                            )
                            finally:
                                overlay.close()

    except (
        UnidentifiedImageError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError(f"Invalid or unsafe image: {source}") from exc

    return {"status": "ok", "output": str(output)}
