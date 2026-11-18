from __future__ import annotations

import json
from dataclasses import fields, is_dataclass

from agentkit.docx._formatting import format_properties
from agentkit.docx._models import (
    Block,
    Break,
    DocumentStructure,
    Hyperlink,
    ImageReference,
    InlineElement,
    Paragraph,
    Section,
    Table,
    TableCell,
    TableCellReference,
    TextRun,
    UnsupportedElement,
)

type JsonValue = (
    bool
    | int
    | float
    | str
    | list[JsonValue]
    | dict[str, JsonValue]
    | None
)

ELEMENT_TYPES: dict[type, str] = {
    TextRun: "text",
    Break: "break",
    UnsupportedElement: "unsupported",
    ImageReference: "image",
    Hyperlink: "hyperlink",
    Paragraph: "paragraph",
    Table: "table",
    TableCell: "cell",
    TableCellReference: "cell_reference",
}


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


class DSLSerializer:
    def serialize(
        self,
        document: DocumentStructure,
    ) -> str:
        lines: list[str] = [
            f"@AGENTKIT_DOCX_STRUCTURE {document.version}",
            f"@FORMAT_RESOLUTION {document.format_resolution}",
            "",
        ]

        for section in document.sections:
            lines.append(
                f"@SECTION {section.index} "
                f"{format_properties(section.properties)}"
            )

        lines.append("")

        for style_id, properties in (
            document.character_styles.items()
        ):
            lines.append(
                f"@CHAR {style_id} "
                f"{format_properties(properties)}"
            )

        for style_id, properties in (
            document.paragraph_styles.items()
        ):
            lines.append(
                f"@PARA {style_id} "
                f"{format_properties(properties)}"
            )

        lines.append("")

        for image_id, image in document.images.items():
            lines.append(
                f"@IMAGE {image_id} "
                f"path={_quote(image.path)} "
                f"content_type={_quote(image.content_type)} "
                f"size_bytes={image.size_bytes}"
            )

        lines.append("")

        for section in document.sections:
            self._section(section, lines)

        lines.append("")

        return "\n".join(lines)

    def _section(
        self,
        section: Section,
        lines: list[str],
    ) -> None:
        lines.append(
            f"[SECTION {section.index}]"
        )

        for block in section.blocks:
            self._block(block, lines, "")

        lines.append(
            f"[/SECTION {section.index}]"
        )

    def _block(
        self,
        block: Block,
        lines: list[str],
        indent: str,
    ) -> None:
        if isinstance(block, Paragraph):
            self._paragraph(block, lines, indent)

        elif isinstance(block, Table):
            self._table(block, lines, indent)

        else:
            raise TypeError(
                f"Unsupported block: {type(block).__name__}"
            )

    def _paragraph(
        self,
        paragraph: Paragraph,
        lines: list[str],
        indent: str,
    ) -> None:
        lines.append(
            f"{indent}[{paragraph.id}:{paragraph.style_id}]"
        )

        for element in paragraph.content:
            self._inline(element, lines, indent)

        lines.append(
            f"{indent}[/{paragraph.id}]"
        )

    def _inline(
        self,
        element: InlineElement,
        lines: list[str],
        indent: str,
    ) -> None:
        if isinstance(element, TextRun):
            lines.append(
                f"{indent}[{element.style_id}] "
                f"{_quote(element.text)}"
            )

        elif isinstance(element, Break):
            lines.append(
                f"{indent}[{element.kind.upper()}_BREAK]"
            )

        elif isinstance(element, UnsupportedElement):
            lines.append(
                f"{indent}[{element.kind.upper()}]"
            )

        elif isinstance(element, ImageReference):
            metadata = format_properties(
                element.properties
            )

            lines.append(
                f"{indent}[IMAGE {element.image_id} "
                f"{metadata}]"
            )

        elif isinstance(element, Hyperlink):
            lines.append(
                f"{indent}[LINK {_quote(element.target)}]"
            )

            for child in element.content:
                self._inline(
                    child,
                    lines,
                    indent,
                )

            lines.append(
                f"{indent}[/LINK]"
            )

        else:
            raise TypeError(
                f"Unsupported inline element: "
                f"{type(element).__name__}"
            )

    def _table(
        self,
        table: Table,
        lines: list[str],
        indent: str,
    ) -> None:
        properties = {
            "rows": table.row_count,
            "columns": table.column_count,
            **table.properties,
        }

        lines.append(
            f"{indent}[TABLE {table.id} "
            f"{format_properties(properties)}]"
        )

        for column in table.columns:
            if column.width_mm is not None:
                lines.append(
                    f"{indent}  [COLUMN {column.index} "
                    f"width_mm={column.width_mm}]"
                )

        for row in table.rows:
            row_properties = {
                "index": row.index,
                **row.properties,
            }

            lines.append(
                f"{indent}  [ROW "
                f"{format_properties(row_properties)}]"
            )

            for cell in row.cells:
                if isinstance(
                    cell,
                    TableCellReference,
                ):
                    lines.append(
                        f"{indent}    "
                        f"[CELL {cell.id} "
                        f"ref={cell.target_id}]"
                    )
                    continue

                lines.append(
                    f"{indent}    [CELL {cell.id} "
                    f"{format_properties(cell.properties)}]"
                )

                for block in cell.blocks:
                    self._block(
                        block,
                        lines,
                        f"{indent}      ",
                    )

                lines.append(
                    f"{indent}    [/CELL]"
                )

            lines.append(
                f"{indent}  [/ROW]"
            )

        lines.append(
            f"{indent}[/TABLE]"
        )


class JSONSerializer:
    def serialize(
        self,
        document: DocumentStructure,
    ) -> str:
        data = self._convert(document)

        return json.dumps(
            data,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n"

    def _convert(
        self,
        value: object,
    ) -> JsonValue:
        if value is None or isinstance(
            value,
            (bool, int, float, str),
        ):
            return value

        if isinstance(value, list):
            return [
                self._convert(item)
                for item in value
            ]

        if isinstance(value, dict):
            result: dict[str, JsonValue] = {}

            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(
                        "JSON dictionary keys must be strings"
                    )

                result[key] = self._convert(item)

            return result

        if (
            not isinstance(value, type)
            and is_dataclass(value)
        ):
            result: dict[str, JsonValue] = {}

            element_type = ELEMENT_TYPES.get(
                type(value)
            )

            if element_type is not None:
                result["type"] = element_type

            for item in fields(value):
                result[item.name] = self._convert(
                    getattr(value, item.name)
                )

            return result

        raise TypeError(
            f"Unsupported JSON value: {type(value).__name__}"
        )


class TextSerializer:
    def serialize(
        self,
        document: DocumentStructure,
    ) -> str:
        sections: list[str] = []

        multiple_sections = len(document.sections) > 1

        for section in document.sections:
            lines: list[str] = []

            if multiple_sections:
                lines.append(
                    f"[SECTION {section.index}]"
                )

            for block in section.blocks:
                lines.append(
                    self._block(block)
                )

            if multiple_sections:
                lines.append(
                    f"[/SECTION {section.index}]"
                )

            sections.append(
                "\n".join(lines)
            )

        return "\n\n".join(sections) + "\n"

    def _block(
        self,
        block: Block,
    ) -> str:
        if isinstance(block, Paragraph):
            return "".join(
                self._inline(element)
                for element in block.content
            )

        if isinstance(block, Table):
            return self._table(block)

        raise TypeError(
            f"Unsupported block: {type(block).__name__}"
        )

    def _inline(
        self,
        element: InlineElement,
    ) -> str:
        if isinstance(element, TextRun):
            return element.text

        if isinstance(element, Hyperlink):
            return "".join(
                self._inline(child)
                for child in element.content
            )

        if isinstance(element, Break):
            return (
                f"\n[{element.kind.upper()}_BREAK]\n"
            )

        if isinstance(element, ImageReference):
            return f"[IMAGE {element.image_id}]"

        if isinstance(element, UnsupportedElement):
            return f"[{element.kind.upper()}]"

        raise TypeError(
            f"Unsupported inline element: "
            f"{type(element).__name__}"
        )

    @staticmethod
    def _column_index(cell_id: str) -> int:
        _, separator, column = cell_id.rpartition("C")

        if not separator or not column.isdecimal():
            raise ValueError(
                f"Invalid table cell identifier: {cell_id}"
            )

        index = int(column)

        if index < 1:
            raise ValueError(
                f"Invalid table column index: {index}"
            )

        return index - 1

    def _table(
        self,
        table: Table,
    ) -> str:
        lines: list[str] = [
            f"[TABLE {table.id} "
            f"{table.row_count}x{table.column_count}]"
        ]

        for row in table.rows:
            cells = [""] * table.column_count

            for cell in row.cells:
                index = self._column_index(cell.id)

                if index >= table.column_count:
                    raise ValueError(
                        f"Cell outside table grid: {cell.id}"
                    )

                if isinstance(
                    cell,
                    TableCellReference,
                ):
                    continue

                cells[index] = self._cell(cell)

            lines.append(
                "\t".join(cells)
            )

        lines.append("[/TABLE]")

        return "\n".join(lines)

    def _cell(
        self,
        cell: TableCell,
    ) -> str:
        text = "\n".join(
            self._block(block)
            for block in cell.blocks
        )

        return (
            text
            .replace("\\", "\\\\")
            .replace("\t", "\\t")
            .replace("\r", "\\r")
            .replace("\n", "\\n")
        )
