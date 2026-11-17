import argparse
import json
import sys
from typing import NoReturn

from agentkit.image.cli import register as image_register


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


class CLIArgumentError(Exception):
    pass


class JSONArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise CLIArgumentError(message)


def main() -> None:
    parser = JSONArgumentParser(
        prog="agentkit",
        description="CLI tools for agents",
    )

    categories = parser.add_subparsers(required=True)
    image_register(categories.add_parser("image"))

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
