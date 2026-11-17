import argparse
import json

from agentkit.image.crop import crop_image
from agentkit.image.info import get_image_info


def register(image_parser: argparse.ArgumentParser) -> None:
    tool = image_parser.add_subparsers(required=True)

    info = tool.add_parser("info", help="Metadata of image")
    info.add_argument("path")
    info.set_defaults(func=handle_info)

    crop = tool.add_parser("crop", help="Crop image")
    crop.add_argument("path")
    crop.add_argument("x1", type=int)
    crop.add_argument("y1", type=int)
    crop.add_argument("x2", type=int)
    crop.add_argument("y2", type=int)
    crop.add_argument("--out", required=True, help="Output image path")
    crop.set_defaults(func=handle_crop)

def handle_info(args: argparse.Namespace) -> None:
    result = get_image_info(args.path)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def handle_crop(args: argparse.Namespace) -> None:
    result = crop_image(args.path, args.out, (args.x1, args.y1, args.x2, args.y2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
