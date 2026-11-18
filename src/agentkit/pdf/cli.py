import argparse
import json

from agentkit.pdf.info import get_pdf_info
from agentkit.pdf.render import DEFAULT_DPI, render_pdf


def register(pdf_parser: argparse.ArgumentParser) -> None:
    tool = pdf_parser.add_subparsers(required=True)

    info = tool.add_parser("info", help="Metadata of PDF document")
    info.add_argument("path", help="Source PDF path")
    info.add_argument("--out", required=True, help="Output JSON path")
    info.set_defaults(func=handle_info)

    render = tool.add_parser("render", help="Render PDF pages to PNG")
    render.add_argument("path", help="Source PDF path")
    render.add_argument(
        "--page",
        type=int,
        help="Page number (1-based); omit to render all pages",
    )
    render.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Rendering DPI: 1..600 (default: %(default)s)",
    )
    render.add_argument(
        "--out",
        required=True,
        help="Output PNG path or output directory",
    )
    render.set_defaults(func=handle_render)


def handle_info(args: argparse.Namespace) -> None:
    result = get_pdf_info(
        source=args.path,
        output=args.out,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def handle_render(args: argparse.Namespace) -> None:
    result = render_pdf(
        source=args.path,
        output=args.out,
        dpi=args.dpi,
        page=args.page,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
