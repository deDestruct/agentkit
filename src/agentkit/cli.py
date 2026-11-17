import argparse

from agentkit.image.cli import register as image_register


def main() -> None:

    parser = argparse.ArgumentParser(
        prog='agentkit',
        description='CLI tools for agents'
    )

    categories = parser.add_subparsers(required=True)   # image/pdf/docx/xlsx
    
    image_register(categories.add_parser("image"))  # register image tools. example crop/resize/info

    print("Hello? Its cli")
