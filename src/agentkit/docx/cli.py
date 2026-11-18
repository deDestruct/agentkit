import argparse
import json

from agentkit.docx.structure import extract_docx_structure


def register(
    docx_parser: argparse.ArgumentParser,
) -> None:
    tool = docx_parser.add_subparsers(
        required=True,
    )

    structure = tool.add_parser(
        "structure",
        help="Extract DOCX structure and content",
        description=(
            "Extract document structure, formatting, "
            "tables and embedded images."
        ),
    )

    structure.add_argument(
        "path",
        help="Source DOCX path",
    )

    structure.add_argument(
        "--out",
        required=True,
        help="New output directory",
    )

    structure.add_argument(
        "--format",
        choices=("dsl", "json", "text"),
        default="dsl",
        help=(
            "Output format: "
            "dsl - compact LLM-readable structure; "
            "json - machine-readable structure; "
            "text - plain text without images "
            "(default: dsl)"
        ),
    )

    structure.set_defaults(
        func=handle_structure,
    )


def handle_structure(
    args: argparse.Namespace,
) -> None:
    result = extract_docx_structure(
        source=args.path,
        output=args.out,
        output_format=args.format,
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )
