import argparse
import json

from agentkit.image.crop import crop_image
from agentkit.image.info import get_image_info
from agentkit.image.resize import resize_image


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
    crop.set_defaults(func=handle_crop)

    resize = tool.add_parser(
        "resize",
        help="Scale image preserving aspect ratio",
    )
    resize.add_argument("path")
    resize.add_argument("--scale", type=float, required=True)
    resize.add_argument("--out", required=True)
    resize.set_defaults(func=handle_resize)

def handle_info(args: argparse.Namespace) -> None:
    result = get_image_info(args.path)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def handle_crop(args: argparse.Namespace) -> None:
    result = crop_image(source = args.path,
                        output = args.out,
                        box = (args.x1, args.y1, args.x2, args.y2))
    print(json.dumps(result, ensure_ascii=False, indent=2))

def handle_resize(args: argparse.Namespace) -> None:
    result = resize_image(
        source=args.path,
        output=args.out,
        scale=args.scale,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
