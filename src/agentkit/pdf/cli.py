import argparse
import json

from agentkit.pdf.info import get_pdf_info


def register(pdf_parser: argparse.ArgumentParser) -> None:
    tool = pdf_parser.add_subparsers(required=True)

    info = tool.add_parser("info", help="Metadata of PDF document")
    info.add_argument("path", help="Source PDF path")
    info.add_argument("--out", required=True, help="Output JSON path")
    info.set_defaults(func=handle_info)


def handle_info(args: argparse.Namespace) -> None:
    result = get_pdf_info(
        source=args.path,
        output=args.out,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
