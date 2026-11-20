import argparse
import json

from agentkit.image.annotate import PALETTE, SHAPES, annotate_image
from agentkit.image.crop import crop_image
from agentkit.image.find import (
    DEFAULT_MIN_AREA,
    DEFAULT_TOLERANCE,
    find_image,
    parse_color,
)
from agentkit.image.grid import (
    DEFAULT_OPACITY,
    DEFAULT_STEP,
    grid_image,
)
from agentkit.image.grid import (
    PALETTE as GRID_PALETTE,
)
from agentkit.image.info import get_image_info
from agentkit.image.resize import resize_image
from agentkit.image.rotate import ANGLES, rotate_image
from agentkit.image.sample import sample_image
from agentkit.image.sheet import DEFAULT_PER_SHEET, sheet_images


def _parse_points(value: str) -> list[tuple[int, int]]:
    entries = value.split(";")
    if not entries or any(not entry.strip() for entry in entries):
        raise argparse.ArgumentTypeError(
            "Points must use the 'x,y;x,y;...' format"
        )

    points: list[tuple[int, int]] = []
    for entry in entries:
        parts = entry.split(",")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"Invalid point {entry!r}: expected x,y"
            )
        try:
            x = int(parts[0].strip())
            y = int(parts[1].strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid point {entry!r}: coordinates must be integers"
            ) from exc
        points.append((x, y))
    return points


def _parse_color(value: str) -> str:
    try:
        parse_color(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def _parse_region(value: str) -> tuple[int, int, int, int]:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Region must use the 'x1,y1,x2,y2' format"
        )
    try:
        x1, y1, x2, y2 = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Region coordinates must be integers"
        ) from exc
    return x1, y1, x2, y2


def register(image_parser: argparse.ArgumentParser) -> None:
    tool = image_parser.add_subparsers(required=True)

    info = tool.add_parser("info", help="Metadata of image")
    info.add_argument("path", help="Source image path")
    info.set_defaults(func=handle_info)

    crop = tool.add_parser("crop", help="Crop image")
    crop.add_argument("path", help="Source image path")
    crop.add_argument("x1", type=int)
    crop.add_argument("y1", type=int)
    crop.add_argument("x2", type=int)
    crop.add_argument("y2", type=int)
    crop.add_argument("--out", required=True, help="Output image path")
    crop.add_argument("--quality", type=int, help="JPEG quality: 1..95 (default: 90); JPEG output only")
    crop.set_defaults(func=handle_crop)

    resize = tool.add_parser("resize", help="Scale image preserving aspect ratio")
    resize.add_argument("path")
    resize.add_argument("--scale", type=float, required=True)
    resize.add_argument("--out", required=True)
    resize.add_argument("--quality", type=int, help="JPEG quality: 1..95 (default: 90); JPEG output only")
    resize.set_defaults(func=handle_resize)

    rotate = tool.add_parser("rotate", help="Rotate image clockwise by 90, 180 or 270 degrees")
    rotate.add_argument("path", help="Source image path")
    rotate.add_argument(
        "--angle",
        type=int,
        required=True,
        choices=ANGLES,
        help="Clockwise rotation angle in degrees: 90, 180 or 270",
    )
    rotate.add_argument("--out", required=True, help="Output image path")
    rotate.add_argument("--quality", type=int, help="JPEG quality: 1..95 (default: 90); JPEG output only")
    rotate.set_defaults(func=handle_rotate)

    sheet = tool.add_parser("sheet", help="Create image contact sheets")
    sheet.add_argument("paths", nargs="+", help="One directory or multiple image paths")
    sheet.add_argument("--out", required=True, help="Output directory")
    sheet.add_argument(
        "--per-sheet",
        type=int,
        default=DEFAULT_PER_SHEET,
        help="Maximum images per sheet (default: %(default)s)",
    )
    sheet.set_defaults(func=handle_sheet)

    annotate = tool.add_parser(
        "annotate", help="Draw a rectangle outline or a line",
        description="Rectangle: x2/y2 are exclusive (same as crop). Line: both endpoints are included.",
    )
    annotate.add_argument("path", help="Source image path")
    annotate.add_argument("shape", choices=SHAPES)
    annotate.add_argument("x1", type=int)
    annotate.add_argument("y1", type=int)
    annotate.add_argument("x2", type=int)
    annotate.add_argument("y2", type=int)
    annotate.add_argument("--color", required=True, choices=tuple(PALETTE))
    annotate.add_argument("--out", required=True, help="Output image path")
    annotate.add_argument("--quality", type=int, help="JPEG quality: 1..95 (default: 90); JPEG output only")
    annotate.set_defaults(func=handle_annotate)

    grid = tool.add_parser(
        "grid",
        help="Overlay a coordinate grid",
        description="Draw grid lines and optional coordinate labels without changing image size.",
    )
    grid.add_argument("source", help="Source image path")
    grid.add_argument("output", help="Output image path")
    grid.add_argument(
        "--step",
        type=int,
        default=DEFAULT_STEP,
        help="Grid spacing in pixels (default: %(default)s)",
    )
    grid.add_argument(
        "--color",
        choices=tuple(GRID_PALETTE),
        default="red",
        help="Grid color (default: %(default)s)",
    )
    grid.add_argument(
        "--opacity",
        type=int,
        default=DEFAULT_OPACITY,
        help="Overlay opacity percent: 0..100 (default: %(default)s)",
    )
    grid.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not draw coordinate labels",
    )
    grid.add_argument("--quality", type=int, help="JPEG quality: 1..95 (default: 90); JPEG output only")
    grid.set_defaults(func=handle_grid)

    sample = tool.add_parser(
        "sample",
        help="Inspect pixels and regional colors",
        description="Read-only pixel and regional color inspection.",
    )
    sample.add_argument("source", help="Source image path")
    sample_mode = sample.add_mutually_exclusive_group(required=True)
    sample_mode.add_argument(
        "--points",
        type=_parse_points,
        metavar="POINTS",
        help='Semicolon-separated points, e.g. "114,170;282,220"',
    )
    sample_mode.add_argument(
        "--region",
        type=_parse_region,
        metavar="REGION",
        help="Region as x1,y1,x2,y2",
    )
    sample.add_argument(
        "--neighbors",
        action="store_true",
        help="Include a 3x3 neighborhood for each point",
    )
    sample.set_defaults(func=handle_sample, sample_parser=sample)

    find = tool.add_parser(
        "find",
        help="Find connected regions of a reference color",
        description=(
            "Write connected-component statistics for pixels matching a color."
        ),
    )
    find.add_argument("source", help="Source image path")
    find.add_argument("output", help="Output JSON path")
    find.add_argument(
        "--color",
        type=_parse_color,
        required=True,
        metavar="COLOR",
        help='Reference color as "#RRGGBB" or "R,G,B"',
    )
    find.add_argument(
        "--tolerance",
        type=int,
        default=DEFAULT_TOLERANCE,
        help="Per-channel tolerance: 0..255 (default: %(default)s)",
    )
    find.add_argument(
        "--region",
        type=_parse_region,
        metavar="REGION",
        help="Region as x1,y1,x2,y2",
    )
    find.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_AREA,
        help="Minimum matching pixels per component (default: %(default)s)",
    )
    find.set_defaults(func=handle_find)


def handle_info(args: argparse.Namespace) -> None:
    result = get_image_info(args.path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def handle_crop(args: argparse.Namespace) -> None:
    result = crop_image(
        source=args.path,
        output=args.out,
        box=(args.x1, args.y1, args.x2, args.y2),
        quality=args.quality,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def handle_resize(args: argparse.Namespace) -> None:
    result = resize_image(source=args.path, output=args.out, scale=args.scale, quality=args.quality)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def handle_rotate(args: argparse.Namespace) -> None:
    result = rotate_image(
        source=args.path,
        output=args.out,
        angle=args.angle,
        quality=args.quality,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def handle_sheet(args: argparse.Namespace) -> None:
    result = sheet_images(
        sources=args.paths,
        output=args.out,
        per_sheet=args.per_sheet,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def handle_annotate(args: argparse.Namespace) -> None:
    result = annotate_image(
        source=args.path,
        output=args.out,
        shape=args.shape,
        coordinates=(args.x1, args.y1, args.x2, args.y2),
        color=args.color,
        quality=args.quality,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def handle_grid(args: argparse.Namespace) -> None:
    result = grid_image(
        source=args.source,
        output=args.output,
        step=args.step,
        color=args.color,
        opacity=args.opacity,
        labels=not args.no_labels,
        quality=args.quality,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def handle_sample(args: argparse.Namespace) -> None:
    if args.neighbors and args.region is not None:
        args.sample_parser.error(
            "--neighbors can only be used with --points"
        )

    result = sample_image(
        source=args.source,
        points=args.points,
        region=args.region,
        neighbors=args.neighbors,
    )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def handle_find(args: argparse.Namespace) -> None:
    result = find_image(
        source=args.source,
        output=args.output,
        color=args.color,
        tolerance=args.tolerance,
        region=args.region,
        min_area=args.min_area,
    )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
