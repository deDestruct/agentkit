from __future__ import annotations

import json
from contextlib import suppress
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal
from zipfile import BadZipFile, ZipFile

from docx import Document
from docx.document import Document as DocumentType
from docx.oxml.ns import qn
from docx.table import Table as WordTable
from docx.text.hyperlink import Hyperlink as WordHyperlink
from docx.text.paragraph import Paragraph as WordParagraph
from docx.text.run import Run
from lxml import etree  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[missing-module-member]

from agentkit.docx._formatting import (
    FormattingResolver,
    Properties,
)
from agentkit.docx._models import (
    Block,
    Break,
    DocumentStructure,
    Hyperlink,
    ImageReference,
    ImageResource,
    InlineElement,
    Paragraph,
    ParsedDocument,
    Section,
    Table,
    TableCell,
    TableCellReference,
    TableColumn,
    TableRow,
    TextRun,
    UnsupportedElement,
)
from agentkit.docx._serializers import (
    DSLSerializer,
    JSONSerializer,
    TextSerializer,
)

MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_UNPACKED_SIZE = 256 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 10_000
MAX_TABLE_DEPTH = 32

type OutputFormat = Literal["dsl", "json", "text"]

IMAGE_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpeg",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
    "image/x-emf": ".emf",
    "image/x-wmf": ".wmf",
    "image/emf": ".emf",
    "image/wmf": ".wmf",
}


def _read_source(source: Path) -> bytes:
    if not source.is_file():
        raise ValueError(f"Not a regular file: {source}")

    with source.open("rb") as stream:
        data = stream.read(MAX_FILE_SIZE + 1)

    if not data:
        raise ValueError(f"File is empty: {source}")

    if len(data) > MAX_FILE_SIZE:
        raise ValueError(
            f"File exceeds 100 MiB limit: {source}"
        )

    return data


def _validate_archive(data: bytes) -> None:
    try:
        with ZipFile(BytesIO(data)) as archive:
            entries = archive.infolist()

            if len(entries) > MAX_ARCHIVE_ENTRIES:
                raise ValueError("Too many DOCX archive entries")

            total_size = 0
            names: set[str] = set()

            for entry in entries:
                name = entry.filename.replace("\\", "/")
                parts = name.split("/")

                if (
                    name.startswith("/")
                    or ".." in parts
                    or ":" in name
                ):
                    raise ValueError(
                        f"Unsafe archive entry: {name}"
                    )

                if name in names:
                    raise ValueError(
                        f"Duplicate archive entry: {name}"
                    )

                names.add(name)

                if entry.flag_bits & 1:
                    raise ValueError(
                        "Encrypted archive entries are unsupported"
                    )

                if entry.file_size > MAX_FILE_SIZE:
                    raise ValueError(
                        f"Archive entry exceeds 100 MiB: {name}"
                    )

                total_size += entry.file_size

                if total_size > MAX_UNPACKED_SIZE:
                    raise ValueError(
                        "Unpacked DOCX exceeds 256 MiB limit"
                    )

            if "word/document.xml" not in names:
                raise ValueError(
                    "Invalid DOCX: missing document.xml"
                )

            actual_size = 0

            for entry in entries:
                if entry.is_dir():
                    continue

                entry_size = 0

                with archive.open(entry) as stream:
                    while chunk := stream.read(1024 * 1024):
                        entry_size += len(chunk)
                        actual_size += len(chunk)

                        if entry_size > MAX_FILE_SIZE:
                            raise ValueError(
                                "Unpacked entry exceeds 100 MiB"
                            )

                        if actual_size > MAX_UNPACKED_SIZE:
                            raise ValueError(
                                "Unpacked DOCX exceeds 256 MiB limit"
                            )

                if entry_size != entry.file_size:
                    raise ValueError(
                        f"Invalid archive entry size: "
                        f"{entry.filename}"
                    )

    except BadZipFile as exc:
        raise ValueError("Invalid DOCX archive") from exc


def _write_file(output: Path, data: bytes) -> None:
    if len(data) > MAX_FILE_SIZE:
        raise ValueError(
            f"Output exceeds 100 MiB limit: {output}"
        )

    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(data)

        output.hardlink_to(temporary_path)

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class StructureParser:
    def __init__(self, document: DocumentType) -> None:
        self.document = document
        self.formatting = FormattingResolver(document)

        self.structure = DocumentStructure()

        self.image_lookup: dict[str, str] = {}
        self.image_data: dict[str, bytes] = {}

    def _register_image(
        self,
        paragraph: WordParagraph,
        relationship_id: str,
    ) -> str | None:
        relationship = paragraph.part.rels.get(
            relationship_id
        )

        if relationship is None or relationship.is_external:
            return None

        part = relationship.target_part

        extension = IMAGE_EXTENSIONS.get(
            part.content_type
        )

        if extension is None:
            return None

        key = str(part.partname)

        if key in self.image_lookup:
            return self.image_lookup[key]

        data = part.blob

        if not data:
            raise ValueError(
                f"Embedded image is empty: {key}"
            )

        if len(data) > MAX_FILE_SIZE:
            raise ValueError(
                f"Embedded image exceeds 100 MiB limit: {key}"
            )

        image_number = len(self.structure.images) + 1
        image_id = f"img{image_number}"

        filename = (
            f"image_{image_number:04d}{extension}"
        )

        resource = ImageResource(
            id=image_id,
            path=f"images/{filename}",
            content_type=part.content_type,
            size_bytes=len(data),
        )

        self.image_lookup[key] = image_id
        self.structure.images[image_id] = resource
        self.image_data[image_id] = data

        return image_id

    def _parse_drawing(
        self,
        paragraph: WordParagraph,
        element: etree._Element,
    ) -> list[InlineElement]:
        result: list[InlineElement] = []

        for shape in element.iter():
            if shape.tag not in (
                qn("wp:inline"),
                qn("wp:anchor"),
            ):
                continue

            extent = shape.find(qn("wp:extent"))

            properties: Properties = {
                "placement": (
                    "inline"
                    if shape.tag == qn("wp:inline")
                    else "floating"
                ),
            }

            if extent is not None:
                width = extent.get("cx")
                height = extent.get("cy")

                if width is not None:
                    properties["width_mm"] = round(
                        int(width) / 36_000,
                        3,
                    )

                if height is not None:
                    properties["height_mm"] = round(
                        int(height) / 36_000,
                        3,
                    )

            description = shape.find(qn("wp:docPr"))

            if description is not None:
                title = description.get("title")
                alt = description.get("descr")

                if title:
                    properties["title"] = title

                if alt:
                    properties["alt"] = alt

            for blip in shape.iter(qn("a:blip")):
                relationship_id = blip.get(
                    qn("r:embed")
                )

                if relationship_id is None:
                    if blip.get(qn("r:link")) is not None:
                        result.append(
                            UnsupportedElement(
                                kind="external_image"
                            )
                        )
                    else:
                        result.append(
                            UnsupportedElement(
                                kind="unsupported_image"
                            )
                        )

                    continue

                image_id = self._register_image(
                    paragraph,
                    relationship_id,
                )

                if image_id is None:
                    result.append(
                        UnsupportedElement(
                            kind="unsupported_image"
                        )
                    )

                    continue

                result.append(
                    ImageReference(
                        image_id=image_id,
                        properties=properties.copy(),
                    )
                )

        if not result:
            result.append(
                UnsupportedElement(
                    kind="unsupported_drawing"
                )
            )

        return result

    def _parse_run(
        self,
        paragraph: WordParagraph,
        run: Run,
    ) -> list[InlineElement]:
        result: list[InlineElement] = []

        style_id = self.formatting.run_style(
            paragraph,
            run,
        )

        for element in run._r:
            tag = element.tag

            if tag == qn("w:t"):
                text = element.text or ""

                if text:
                    result.append(
                        TextRun(
                            style_id=style_id,
                            text=text,
                        )
                    )

            elif tag == qn("w:tab"):
                result.append(
                    TextRun(
                        style_id=style_id,
                        text="\t",
                    )
                )

            elif tag in (
                qn("w:br"),
                qn("w:cr"),
            ):
                break_type = element.get(
                    qn("w:type")
                )

                if break_type == "page":
                    result.append(
                        Break(kind="page")
                    )

                elif break_type == "column":
                    result.append(
                        Break(kind="column")
                    )

                else:
                    result.append(
                        TextRun(
                            style_id=style_id,
                            text="\n",
                        )
                    )

            elif tag == qn("w:noBreakHyphen"):
                result.append(
                    TextRun(
                        style_id=style_id,
                        text="\u2011",
                    )
                )

            elif tag == qn("w:softHyphen"):
                result.append(
                    TextRun(
                        style_id=style_id,
                        text="\u00ad",
                    )
                )

            elif tag == qn("w:drawing"):
                result.extend(
                    self._parse_drawing(
                        paragraph,
                        element,
                    )
                )

            elif tag == qn("w:pict"):
                result.append(
                    UnsupportedElement(
                        kind="unsupported_vml"
                    )
                )

            elif tag == qn("w:object"):
                result.append(
                    UnsupportedElement(
                        kind="embedded_object"
                    )
                )

        return result

    @staticmethod
    def _merge_text_runs(
        elements: list[InlineElement],
    ) -> list[InlineElement]:
        result: list[InlineElement] = []

        pending_style: str | None = None
        pending_text: list[str] = []

        def flush() -> None:
            nonlocal pending_style

            if pending_text and pending_style is not None:
                result.append(
                    TextRun(
                        style_id=pending_style,
                        text="".join(pending_text),
                    )
                )

            pending_text.clear()
            pending_style = None

        for element in elements:
            if isinstance(element, TextRun):
                if not element.text:
                    continue

                if pending_style != element.style_id:
                    flush()
                    pending_style = element.style_id

                pending_text.append(element.text)

            else:
                flush()
                result.append(element)

        flush()

        return result

    def _parse_hyperlink(
        self,
        paragraph: WordParagraph,
        hyperlink: WordHyperlink,
    ) -> Hyperlink:
        target = hyperlink.url

        if not target and hyperlink.fragment:
            target = f"#{hyperlink.fragment}"

        content: list[InlineElement] = []

        for run in hyperlink.runs:
            content.extend(
                self._parse_run(
                    paragraph,
                    run,
                )
            )

        return Hyperlink(
            target=target,
            content=self._merge_text_runs(content),
        )

    def _parse_paragraph(
        self,
        paragraph: WordParagraph,
    ) -> Paragraph:
        self.structure.paragraph_count += 1

        paragraph_id = (
            f"P{self.structure.paragraph_count}"
        )

        style_id = self.formatting.paragraph_style(
            paragraph
        )

        content: list[InlineElement] = []

        for item in paragraph.iter_inner_content():
            if isinstance(item, WordHyperlink):
                content.append(
                    self._parse_hyperlink(
                        paragraph,
                        item,
                    )
                )

            else:
                content.extend(
                    self._parse_run(
                        paragraph,
                        item,
                    )
                )

        return Paragraph(
            id=paragraph_id,
            style_id=style_id,
            content=self._merge_text_runs(content),
        )

    def _parse_table(
        self,
        table: WordTable,
        depth: int,
    ) -> Table:
        if depth > MAX_TABLE_DEPTH:
            raise ValueError(
                "Maximum nested table depth exceeded"
            )

        self.structure.table_count += 1

        table_id = (
            f"T{self.structure.table_count}"
        )

        properties: Properties = {}

        if table.style is not None:
            properties["style"] = table.style.name

        if table.alignment is not None:
            properties["alignment"] = (
                table.alignment.name.lower()
            )

        properties["autofit"] = table.autofit

        result = Table(
            id=table_id,
            row_count=len(table.rows),
            column_count=len(table.columns),
            properties=properties,
        )

        for column_index, column in enumerate(
            table.columns,
            1,
        ):
            width = (
                round(column.width.mm, 3)
                if column.width is not None
                else None
            )

            result.columns.append(
                TableColumn(
                    index=column_index,
                    width_mm=width,
                )
            )

        visited_cells: dict[object, str] = {}

        for row_index, row in enumerate(
            table.rows,
            1,
        ):
            row_properties: Properties = {}

            if row.height is not None:
                row_properties["height_mm"] = round(
                    row.height.mm,
                    3,
                )

            if row.height_rule is not None:
                row_properties["height_rule"] = (
                    row.height_rule.name.lower()
                )

            parsed_row = TableRow(
                index=row_index,
                properties=row_properties,
            )

            column_index = row.grid_cols_before + 1

            for cell in row.cells:
                cell_id = (
                    f"{table_id}R{row_index}C{column_index}"
                )

                original_id = visited_cells.get(
                    cell._tc
                )

                if original_id is not None:
                    parsed_row.cells.append(
                        TableCellReference(
                            id=cell_id,
                            target_id=original_id,
                        )
                    )

                    column_index += 1
                    continue

                visited_cells[cell._tc] = cell_id

                cell_properties: Properties = {
                    "colspan": cell.grid_span,
                }

                if cell.width is not None:
                    cell_properties["width_mm"] = round(
                        cell.width.mm,
                        3,
                    )

                if cell.vertical_alignment is not None:
                    cell_properties["vertical_alignment"] = (
                        cell.vertical_alignment.name.lower()
                    )

                tcpr = cell._tc.find(qn("w:tcPr"))

                if tcpr is not None:
                    shading = tcpr.find(qn("w:shd"))

                    if shading is not None:
                        fill = shading.get(qn("w:fill"))

                        if fill is not None:
                            cell_properties["fill"] = fill

                parsed_cell = TableCell(
                    id=cell_id,
                    properties=cell_properties,
                )

                for block in cell.iter_inner_content():
                    parsed_cell.blocks.append(
                        self._parse_block(
                            block,
                            depth + 1,
                        )
                    )

                parsed_row.cells.append(parsed_cell)

                column_index += 1

            result.rows.append(parsed_row)

        return result

    def _parse_block(
        self,
        block: WordParagraph | WordTable,
        depth: int = 0,
    ) -> Block:
        if isinstance(block, WordParagraph):
            return self._parse_paragraph(block)

        if isinstance(block, WordTable):
            return self._parse_table(block, depth)

        raise TypeError(
            f"Unsupported block: {type(block).__name__}"
        )

    def _parse_sections(self) -> list[Section]:
        result: list[Section] = []

        for index, section in enumerate(
            self.document.sections,
            1,
        ):
            properties: Properties = {}

            if section.orientation is not None:
                properties["orientation"] = (
                    section.orientation.name.lower()
                )

            if section.start_type is not None:
                properties["start"] = (
                    section.start_type.name.lower()
                )

            lengths = {
                "width_mm": section.page_width,
                "height_mm": section.page_height,
                "top_mm": section.top_margin,
                "bottom_mm": section.bottom_margin,
                "left_mm": section.left_margin,
                "right_mm": section.right_margin,
                "header_distance_mm": section.header_distance,
                "footer_distance_mm": section.footer_distance,
                "gutter_mm": section.gutter,
            }

            for key, value in lengths.items():
                if value is not None:
                    properties[key] = round(
                        value.mm,
                        3,
                    )

            result.append(
                Section(
                    index=index,
                    properties=properties,
                )
            )

        return result

    def parse(self) -> ParsedDocument:
        self.structure = DocumentStructure()

        self.formatting = FormattingResolver(
            self.document
        )

        self.image_lookup.clear()
        self.image_data.clear()

        sections = self._parse_sections()

        if not sections:
            raise ValueError(
                "DOCX contains no document sections"
            )

        section_index = 0

        for block in self.document.iter_inner_content():
            if section_index >= len(sections):
                raise ValueError(
                    "Invalid DOCX section structure"
                )

            sections[section_index].blocks.append(
                self._parse_block(block)
            )

            if isinstance(block, WordParagraph):
                ppr = block._p.find(qn("w:pPr"))

                if (
                    ppr is not None
                    and ppr.find(qn("w:sectPr")) is not None
                ):
                    section_index += 1

        if section_index != len(sections) - 1:
            raise ValueError(
                "DOCX section count mismatch"
            )

        self.structure.sections = sections

        self.structure.character_styles = {
            style_id: properties.copy()
            for style_id, properties in (
                self.formatting.character_styles.styles.items()
            )
        }

        self.structure.paragraph_styles = {
            style_id: properties.copy()
            for style_id, properties in (
                self.formatting.paragraph_styles.styles.items()
            )
        }

        return ParsedDocument(
            structure=self.structure,
            image_data=self.image_data.copy(),
        )


def parse_docx(
    source: str | Path,
) -> ParsedDocument:
    source = Path(source).expanduser().resolve(strict=True)

    data = _read_source(source)

    _validate_archive(data)

    document = Document(BytesIO(data))

    return StructureParser(document).parse()


def _serialize_document(
    document: DocumentStructure,
    output_format: OutputFormat,
) -> tuple[str, bytes]:
    match output_format:
        case "dsl":
            filename = "structure.txt"
            content = DSLSerializer().serialize(document)

        case "json":
            filename = "structure.json"
            content = JSONSerializer().serialize(document)

        case "text":
            filename = "text.txt"
            content = TextSerializer().serialize(document)

        case _:
            raise ValueError(
                f"Unsupported output format: {output_format}"
            )

    data = content.encode("utf-8")

    if len(data) > MAX_FILE_SIZE:
        raise ValueError(
            "Structure output exceeds 100 MiB limit"
        )

    return filename, data


def extract_docx_structure(
    source: str | Path,
    output: str | Path,
    output_format: OutputFormat = "dsl",
) -> dict[str, str | int]:
    source = Path(source).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()

    if source == output:
        raise ValueError(
            "Source and output must be different"
        )

    if output.exists():
        raise FileExistsError(
            f"Output already exists: {output}"
        )

    parsed = parse_docx(source)
    document = parsed.structure

    filename, structure = _serialize_document(
        document,
        output_format,
    )

    structure_path = output / filename
    result_path = output / "result.json"
    images_directory = output / "images"

    images = {
        image_id: {
            "path": str(output / image.path),
            "content_type": image.content_type,
            "size_bytes": image.size_bytes,
        }
        for image_id, image in document.images.items()
        if output_format != "text"
    }

    result = {
        "status": "ok",
        "source": str(source),
        "output": str(output),
        "structure": str(structure_path),
        "output_format": output_format,
        "structure_version": document.version,
        "paragraph_count": document.paragraph_count,
        "table_count": document.table_count,
        "image_count": document.image_count,
        "extracted_image_count": len(images),
        "section_count": document.section_count,
        "character_style_count": (
            document.character_style_count
        ),
        "paragraph_style_count": (
            document.paragraph_style_count
        ),
        "format_resolution": document.format_resolution,
        "images": images,
    }

    payload = (
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ) + "\n"
    ).encode("utf-8")

    if len(payload) > MAX_FILE_SIZE:
        raise ValueError(
            "Result JSON exceeds 100 MiB limit"
        )

    created_files: list[Path] = []
    created_directories: list[Path] = []

    try:
        output.mkdir(
            parents=True,
            exist_ok=False,
        )

        created_directories.append(output)

        if images:
            images_directory.mkdir(
                exist_ok=False,
            )

            created_directories.append(
                images_directory
            )

            for image_id, image in document.images.items():
                image_path = output / image.path

                _write_file(
                    image_path,
                    parsed.image_data[image_id],
                )

                created_files.append(image_path)

        _write_file(
            structure_path,
            structure,
        )

        created_files.append(structure_path)

        _write_file(
            result_path,
            payload,
        )

        created_files.append(result_path)

    except BaseException:
        for path in reversed(created_files):
            with suppress(OSError):
                path.unlink(missing_ok=True)

        for path in reversed(created_directories):
            with suppress(OSError):
                path.rmdir()

        raise

    return {
        "status": "ok",
        "output": str(output),
        "format": output_format,
        "structure": str(structure_path),
        "result": str(result_path),
        "image_count": document.image_count,
        "extracted_image_count": len(images),
    }
