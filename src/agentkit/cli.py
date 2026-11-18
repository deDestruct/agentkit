import argparse
import io
import json
import sys
from typing import NoReturn

from agentkit.docx.cli import register as register_docx
from agentkit.image.cli import register as image_register
from agentkit.pdf.cli import register as pdf_register


def print_error(exc: Exception) -> None:
    print(
        json.dumps(
            {
                "status": "error",
                "error": str(exc),
                "type": type(exc).__name__,
            },
            ensure_ascii=False,
            indent=2,
        ),
        file=sys.stderr,
    )

def configure_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(
                encoding="utf-8",
                errors="strict",
            )


class CLIArgumentError(Exception):
    pass


class JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise CLIArgumentError(message)


def main() -> None:
    configure_output()

    parser = JSONArgumentParser(
        prog="agentkit",
        description="CLI tools for agents",
    )

    categories = parser.add_subparsers(required=True)

    image_register(categories.add_parser("image"))
    pdf_register(categories.add_parser("pdf"))
    register_docx(categories.add_parser("docx"))

    try:
        args = parser.parse_args()
        args.func(args)

    except CLIArgumentError as exc:
        print_error(exc)
        sys.exit(2)

    except (OSError, ValueError, TypeError) as exc:
        print_error(exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
