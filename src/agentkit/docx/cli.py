import argparse
import json

from agentkit.docx.render import (
    DEFAULT_TIMEOUT,
    render_docx,
)
from agentkit.docx.structure import extract_docx_structure
from agentkit.pdf.render import DEFAULT_DPI


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
            "text - plain text without image extraction "
            "(default: dsl)"
        ),
    )

    structure.set_defaults(
        func=handle_structure,
    )

    render = tool.add_parser(
        "render",
        help="Render DOCX pages as PNG images",
        description=(
            "Render DOCX pages using an available "
            "document rendering backend."
        ),
    )

    render.add_argument(
        "path",
        help="Source DOCX path",
    )

    render.add_argument(
        "--out",
        required=True,
        help=(
            "New output directory, or PNG file "
            "when rendering a single page"
        ),
    )

    render.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Rendering resolution (default: 150)",
    )

    render.add_argument(
        "--page",
        type=int,
        default=None,
        help="Render only the specified page (1-based)",
    )

    render.add_argument(
        "--backend",
        choices=("auto", "word"),
        default="auto",
        help="Rendering backend (default: auto)",
    )

    render.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="DOCX conversion timeout in seconds (default: 120)",
    )

    render.set_defaults(
        func=handle_render,
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


def handle_render(
    args: argparse.Namespace,
) -> None:
    result = render_docx(
        source=args.path,
        output=args.out,
        dpi=args.dpi,
        page=args.page,
        backend=args.backend,
        timeout=args.timeout,
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )
