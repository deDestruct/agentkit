import json

from docx.document import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.text.run import Run
from lxml import etree  # ty: ignore[missing-module-member] # pyright: ignore[reportAttributeAccessIssue]

type PropertyValue = str | int | float | bool | None
type Properties = dict[str, PropertyValue]

TOGGLE_PROPERTIES = {
    "b": "bold",
    "i": "italic",
    "strike": "strike",
    "dstrike": "double_strike",
    "caps": "caps",
    "smallCaps": "small_caps",
    "vanish": "hidden",
}

FALSE_VALUES = frozenset({"0", "false", "off"})


def _child(
    element: etree._Element | None,
    name: str,
) -> etree._Element | None:
    if element is None:
        return None

    return element.find(qn(f"w:{name}"))


def _attribute(
    element: etree._Element | None,
    name: str,
) -> str | None:
    if element is None:
        return None

    return element.get(qn(f"w:{name}"))


def _value(
    element: etree._Element | None,
    name: str,
) -> str | None:
    return _attribute(_child(element, name), "val")


def _boolean(element: etree._Element) -> bool:
    value = _attribute(element, "val")

    return value not in FALSE_VALUES


def _number(value: str | None) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return None


def _twips_to_mm(value: int) -> float:
    return round(value * 25.4 / 1440, 3)


def _half_points(value: int) -> float:
    return value / 2


def format_properties(properties: Properties) -> str:
    return " ".join(
        f"{key}={json.dumps(value, ensure_ascii=False)}"
        for key, value in properties.items()
    )


class StyleRegistry:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.styles: dict[str, Properties] = {}
        self._lookup: dict[str, str] = {}

    def register(self, properties: Properties) -> str:
        key = json.dumps(
            properties,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )

        if key not in self._lookup:
            style_id = f"{self.prefix}{len(self.styles) + 1}"

            self._lookup[key] = style_id
            self.styles[style_id] = properties.copy()

        return self._lookup[key]

    def definitions(self, directive: str) -> list[str]:
        return [
            f"@{directive} {style_id} "
            f"{format_properties(properties)}"
            for style_id, properties in self.styles.items()
        ]


class ThemeFonts:
    def __init__(self, document: Document) -> None:
        self.fonts: dict[str, str] = {}

        for relationship in document.part.rels.values():
            if (
                relationship.reltype != RT.THEME
                or relationship.is_external
            ):
                continue

            parser = etree.XMLParser(
                resolve_entities=False,
                no_network=True,
                load_dtd=False,
                huge_tree=False,
            )

            root = etree.fromstring(
                relationship.target_part.blob,
                parser=parser,
            )

            self._parse(root)
            break

    def _parse(self, root: etree._Element) -> None:
        scheme = root.find(
            f"{qn('a:themeElements')}/"
            f"{qn('a:fontScheme')}"
        )

        if scheme is None:
            return

        for prefix in ("major", "minor"):
            group = scheme.find(
                qn(f"a:{prefix}Font")
            )

            if group is None:
                continue

            latin = group.find(qn("a:latin"))
            east_asia = group.find(qn("a:ea"))
            complex_script = group.find(qn("a:cs"))

            if latin is not None:
                name = latin.get("typeface")

                if name:
                    self.fonts[f"{prefix}Ascii"] = name
                    self.fonts[f"{prefix}HAnsi"] = name

            if east_asia is not None:
                name = east_asia.get("typeface")

                if name:
                    self.fonts[f"{prefix}EastAsia"] = name

            if complex_script is not None:
                name = complex_script.get("typeface")

                if name:
                    self.fonts[f"{prefix}Bidi"] = name

            for font in group.findall(qn("a:font")):
                script = font.get("script")
                name = font.get("typeface")

                if script and name:
                    self.fonts[f"{prefix}:{script}"] = name

    def resolve(self, name: str) -> str | None:
        return self.fonts.get(name)


class FormattingResolver:
    def __init__(self, document: Document) -> None:
        self.theme = ThemeFonts(document)

        self.character_styles = StyleRegistry("c")
        self.paragraph_styles = StyleRegistry("p")

        root = document.styles.element

        self._styles: dict[str, etree._Element] = {}
        self._defaults: dict[str, str] = {}

        for style in root.findall(qn("w:style")):
            style_id = _attribute(style, "styleId")
            style_type = _attribute(style, "type")

            if style_id is None:
                continue

            self._styles[style_id] = style

            if (
                style_type is not None
                and _attribute(style, "default")
                in {"1", "true", "on"}
            ):
                self._defaults[style_type] = style_id

        defaults = _child(root, "docDefaults")

        self._default_rpr = _child(
            _child(defaults, "rPrDefault"),
            "rPr",
        )

        self._default_ppr = _child(
            _child(defaults, "pPrDefault"),
            "pPr",
        )

    def _style_chain(
        self,
        style_id: str | None,
    ) -> list[etree._Element]:
        chain: list[etree._Element] = []
        visited: set[str] = set()

        while style_id is not None:
            if style_id in visited:
                break

            visited.add(style_id)

            style = self._styles.get(style_id)

            if style is None:
                break

            chain.append(style)
            style_id = _value(style, "basedOn")

        chain.reverse()

        return chain

    def _paragraph_chain(
        self,
        paragraph: Paragraph,
    ) -> list[etree._Element]:
        ppr = _child(paragraph._p, "pPr")

        style_id = (
            _value(ppr, "pStyle")
            or self._defaults.get("paragraph")
        )

        return self._style_chain(style_id)

    def _apply_fonts(
        self,
        properties: Properties,
        rpr: etree._Element,
    ) -> None:
        fonts = _child(rpr, "rFonts")

        if fonts is None:
            return

        font_slots = (
            ("ascii", "asciiTheme", "font_ascii"),
            ("hAnsi", "hAnsiTheme", "font_hansi"),
            ("eastAsia", "eastAsiaTheme", "font_east_asia"),
            ("cs", "cstheme", "font_complex_script"),
        )

        for direct, theme, key in font_slots:
            direct_name = _attribute(fonts, direct)
            theme_name = _attribute(fonts, theme)

            if theme_name is not None:
                properties[key] = self.theme.resolve(
                    theme_name
                )
                properties[f"{key}_theme"] = theme_name

            elif direct_name is not None:
                properties[key] = direct_name
                properties.pop(f"{key}_theme", None)

    def _apply_run_properties(
        self,
        properties: Properties,
        rpr: etree._Element | None,
        *,
        style_layer: bool,
    ) -> None:
        if rpr is None:
            return

        self._apply_fonts(properties, rpr)

        size = _number(_value(rpr, "sz"))

        if size is not None:
            properties["size_pt"] = _half_points(size)

        size_complex = _number(_value(rpr, "szCs"))

        if size_complex is not None:
            properties["size_complex_pt"] = (
                _half_points(size_complex)
            )

        for xml_name, property_name in (
            TOGGLE_PROPERTIES.items()
        ):
            element = _child(rpr, xml_name)

            if element is None:
                continue

            enabled = _boolean(element)

            if style_layer:
                if enabled:
                    properties[property_name] = not bool(
                        properties.get(property_name, False)
                    )
            else:
                properties[property_name] = enabled

        underline = _value(rpr, "u")

        if underline is not None:
            properties["underline"] = (
                False if underline == "none" else underline
            )

        color = _child(rpr, "color")

        if color is not None:
            rgb = _attribute(color, "val")
            theme = _attribute(color, "themeColor")

            for key in (
                "color",
                "color_theme",
                "color_theme_tint",
                "color_theme_shade",
            ):
                properties.pop(key, None)

            if rgb is not None:
                properties["color"] = (
                    "auto" if rgb == "auto" else f"#{rgb}"
                )

            if theme is not None:
                properties["color_theme"] = theme

                tint = _attribute(color, "themeTint")
                shade = _attribute(color, "themeShade")

                if tint is not None:
                    properties["color_theme_tint"] = tint

                if shade is not None:
                    properties["color_theme_shade"] = shade

        highlight = _value(rpr, "highlight")

        if highlight is not None:
            properties["highlight"] = highlight

        vertical = _value(rpr, "vertAlign")

        if vertical is not None:
            properties["vertical_alignment"] = vertical

        spacing = _number(_value(rpr, "spacing"))

        if spacing is not None:
            properties["character_spacing_pt"] = (
                spacing / 20
            )

        position = _number(_value(rpr, "position"))

        if position is not None:
            properties["position_pt"] = position / 2

        language = _child(rpr, "lang")

        if language is not None:
            for slot, key in (
                ("val", "language"),
                ("eastAsia", "language_east_asia"),
                ("bidi", "language_bidi"),
            ):
                value = _attribute(language, slot)

                if value is not None:
                    properties[key] = value

    def _apply_paragraph_properties(
        self,
        properties: Properties,
        ppr: etree._Element | None,
    ) -> None:
        if ppr is None:
            return

        alignment = _value(ppr, "jc")

        if alignment is not None:
            properties["alignment"] = {
                "both": "justify",
            }.get(alignment, alignment)

        indentation = _child(ppr, "ind")

        if indentation is not None:
            for xml_name, key in (
                ("left", "left_indent_mm"),
                ("right", "right_indent_mm"),
                ("start", "start_indent_mm"),
                ("end", "end_indent_mm"),
            ):
                value = _number(
                    _attribute(indentation, xml_name)
                )

                if value is not None:
                    properties[key] = _twips_to_mm(value)

            hanging = _number(
                _attribute(indentation, "hanging")
            )

            first_line = _number(
                _attribute(indentation, "firstLine")
            )

            if hanging is not None:
                properties["first_line_indent_mm"] = (
                    -_twips_to_mm(hanging)
                )

            elif first_line is not None:
                properties["first_line_indent_mm"] = (
                    _twips_to_mm(first_line)
                )

        spacing = _child(ppr, "spacing")

        if spacing is not None:
            for name in ("before", "after"):
                value = _number(
                    _attribute(spacing, name)
                )

                if value is not None:
                    properties[f"space_{name}_pt"] = value / 20
                    properties.pop(
                        f"space_{name}_lines",
                        None,
                    )

                lines = _number(
                    _attribute(spacing, f"{name}Lines")
                )

                if lines is not None:
                    properties[f"space_{name}_lines"] = (
                        lines / 100
                    )
                    properties.pop(
                        f"space_{name}_pt",
                        None,
                    )

                automatic = _attribute(
                    spacing,
                    f"{name}Autospacing",
                )

                if automatic is not None:
                    properties[f"space_{name}_auto"] = (
                        automatic not in FALSE_VALUES
                    )

            line = _number(
                _attribute(spacing, "line")
            )

            if line is not None:
                properties["line_spacing_raw"] = line

            rule = _attribute(spacing, "lineRule")

            if rule is not None:
                properties["line_spacing_rule"] = rule

        for xml_name, key in (
            ("keepNext", "keep_with_next"),
            ("keepLines", "keep_lines"),
            ("pageBreakBefore", "page_break_before"),
            ("widowControl", "widow_control"),
            ("contextualSpacing", "contextual_spacing"),
            ("bidi", "right_to_left"),
        ):
            element = _child(ppr, xml_name)

            if element is not None:
                properties[key] = _boolean(element)

        outline = _number(_value(ppr, "outlineLvl"))

        if outline is not None:
            properties["outline_level"] = outline

        numbering = _child(ppr, "numPr")

        if numbering is not None:
            num_id = _number(
                _value(numbering, "numId")
            )
            level = _number(
                _value(numbering, "ilvl")
            )

            if num_id is not None:
                properties["numbering_id"] = num_id

            if level is not None:
                properties["numbering_level"] = level

    @staticmethod
    def _normalize_fonts(
        properties: Properties,
    ) -> None:
        ascii_font = properties.get("font_ascii")
        hansi_font = properties.get("font_hansi")

        if ascii_font == hansi_font and ascii_font is not None:
            properties["font"] = ascii_font

            properties.pop("font_ascii", None)
            properties.pop("font_hansi", None)

    @staticmethod
    def _normalize_spacing(
        properties: Properties,
    ) -> None:
        raw = properties.pop(
            "line_spacing_raw",
            None,
        )

        if not isinstance(raw, int):
            return

        rule = properties.get(
            "line_spacing_rule",
            "auto",
        )

        if rule == "auto":
            properties["line_spacing"] = raw / 240
            properties["line_spacing_unit"] = "multiple"

        elif rule in ("exact", "atLeast"):
            properties["line_spacing"] = raw / 20
            properties["line_spacing_unit"] = "pt"

        else:
            properties["line_spacing_raw"] = raw

    def resolve_run(
        self,
        paragraph: Paragraph,
        run: Run,
    ) -> Properties:
        properties: Properties = {}

        self._apply_run_properties(
            properties,
            self._default_rpr,
            style_layer=False,
        )

        for style in self._paragraph_chain(paragraph):
            self._apply_run_properties(
                properties,
                _child(style, "rPr"),
                style_layer=True,
            )

        rpr = _child(run._r, "rPr")

        character_style_id = (
            _value(rpr, "rStyle")
            or self._defaults.get("character")
        )

        for style in self._style_chain(
            character_style_id
        ):
            self._apply_run_properties(
                properties,
                _child(style, "rPr"),
                style_layer=True,
            )

        self._apply_run_properties(
            properties,
            rpr,
            style_layer=False,
        )

        self._normalize_fonts(properties)

        return properties

    def resolve_paragraph(
        self,
        paragraph: Paragraph,
    ) -> Properties:
        properties: Properties = {}

        self._apply_paragraph_properties(
            properties,
            self._default_ppr,
        )

        for style in self._paragraph_chain(paragraph):
            self._apply_paragraph_properties(
                properties,
                _child(style, "pPr"),
            )

        self._apply_paragraph_properties(
            properties,
            _child(paragraph._p, "pPr"),
        )

        self._normalize_spacing(properties)

        return properties

    def run_style(
        self,
        paragraph: Paragraph,
        run: Run,
    ) -> str:
        properties = self.resolve_run(paragraph, run)

        return self.character_styles.register(
            properties
        )

    def paragraph_style(
        self,
        paragraph: Paragraph,
    ) -> str:
        properties = self.resolve_paragraph(paragraph)

        return self.paragraph_styles.register(
            properties
        )

    def definitions(self) -> list[str]:
        return [
            *self.character_styles.definitions("CHAR"),
            *self.paragraph_styles.definitions("PARA"),
        ]
