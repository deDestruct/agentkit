from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from agentkit.docx._formatting import Properties


@dataclass(slots=True)
class TextRun:
    style_id: str
    text: str


@dataclass(slots=True)
class Break:
    kind: Literal["page", "column"]


@dataclass(slots=True)
class UnsupportedElement:
    kind: Literal[
        "external_image",
        "unsupported_image",
        "unsupported_drawing",
        "unsupported_vml",
        "embedded_object",
    ]


@dataclass(slots=True)
class ImageResource:
    id: str
    path: str
    content_type: str
    size_bytes: int


@dataclass(slots=True)
class ImageReference:
    image_id: str
    properties: Properties = field(
        default_factory=dict
    )


@dataclass(slots=True)
class Hyperlink:
    target: str
    content: list[InlineElement] = field(
        default_factory=list
    )


type InlineElement = (
    TextRun
    | Break
    | UnsupportedElement
    | ImageReference
    | Hyperlink
)


@dataclass(slots=True)
class Paragraph:
    id: str
    style_id: str
    content: list[InlineElement] = field(
        default_factory=list
    )


@dataclass(slots=True)
class TableColumn:
    index: int
    width_mm: float | None = None


@dataclass(slots=True)
class TableCell:
    id: str
    properties: Properties = field(
        default_factory=dict
    )
    blocks: list[Block] = field(
        default_factory=list
    )


@dataclass(slots=True)
class TableCellReference:
    id: str
    target_id: str


type Cell = TableCell | TableCellReference


@dataclass(slots=True)
class TableRow:
    index: int
    properties: Properties = field(
        default_factory=dict
    )
    cells: list[Cell] = field(
        default_factory=list
    )


@dataclass(slots=True)
class Table:
    id: str
    row_count: int
    column_count: int
    properties: Properties = field(
        default_factory=dict
    )
    columns: list[TableColumn] = field(
        default_factory=list
    )
    rows: list[TableRow] = field(
        default_factory=list
    )


type Block = Paragraph | Table


@dataclass(slots=True)
class Section:
    index: int
    properties: Properties = field(
        default_factory=dict
    )
    blocks: list[Block] = field(
        default_factory=list
    )


@dataclass(slots=True)
class DocumentStructure:
    sections: list[Section] = field(
        default_factory=list
    )

    character_styles: dict[str, Properties] = field(
        default_factory=dict
    )

    paragraph_styles: dict[str, Properties] = field(
        default_factory=dict
    )

    images: dict[str, ImageResource] = field(
        default_factory=dict
    )

    paragraph_count: int = 0
    table_count: int = 0

    version: int = 2
    format_resolution: Literal["partial"] = "partial"

    @property
    def section_count(self) -> int:
        return len(self.sections)

    @property
    def image_count(self) -> int:
        return len(self.images)

    @property
    def character_style_count(self) -> int:
        return len(self.character_styles)

    @property
    def paragraph_style_count(self) -> int:
        return len(self.paragraph_styles)


@dataclass(slots=True)
class ParsedDocument:
    structure: DocumentStructure

    image_data: dict[str, bytes] = field(
        default_factory=dict,
        repr=False,
    )
