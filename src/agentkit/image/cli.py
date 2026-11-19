import argparse
import json

from agentkit.image.annotate import PALETTE, SHAPES, annotate_image
from agentkit.image.crop import crop_image
from agentkit.image.info import get_image_info
from agentkit.image.resize import resize_image
from agentkit.image.rotate import ANGLES, rotate_image
from agentkit.image.sheet import DEFAULT_PER_SHEET, sheet_images


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
