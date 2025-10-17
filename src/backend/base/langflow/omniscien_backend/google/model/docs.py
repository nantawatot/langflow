from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SuggestionsViewMode(Enum):
    DEFAULT_FOR_CURRENT_ACCESS = "DEFAULT_FOR_CURRENT_ACCESS"
    SUGGESTIONS_INLINE = "SUGGESTIONS_INLINE"
    PREVIEW_SUGGESTIONS_ACCEPTED = "PREVIEW_SUGGESTIONS_ACCEPTED"
    PREVIEW_WITHOUT_SUGGESTIONS = "PREVIEW_WITHOUT_SUGGESTIONS"


class TabProperties(BaseModel):
    tabId: str | None = Field(None, description="Output only. The ID of the tab. This field can't be changed.")
    title: str | None = Field(None, description="The user-visible name of the tab.")
    parentTabId: str | None = Field(
        None,
        description="Optional. The ID of the parent tab. Empty when the current tab is a root-level tab, which means it doesn't have any parents.",
    )
    index: int | None = Field(None, description="The zero-based index of the tab within the parent.")
    nestingLevel: int | None = Field(
        None, description="Output only. The depth of the tab within the document. Root-level tabs start at 0."
    )


class BaselineOffset(Enum):
    BASELINE_OFFSET_UNSPECIFIED = "BASELINE_OFFSET_UNSPECIFIED"
    NONE = "NONE"
    SUPERSCRIPT = "SUPERSCRIPT"
    SUBSCRIPT = "SUBSCRIPT"


class RgbColor(BaseModel):
    red: float | None = Field(None, description="The red component of the color, from 0.0 to 1.0.")
    green: float | None = Field(None, description="The green component of the color, from 0.0 to 1.0.")
    blue: float | None = Field(None, description="The blue component of the color, from 0.0 to 1.0.")


class Unit(Enum):
    UNIT_UNSPECIFIED = "UNIT_UNSPECIFIED"
    PT = "PT"


class Dimension(BaseModel):
    class Config:
        use_enum_values = True

    magnitude: float | None = Field(None, description="The magnitude.")
    unit: Unit | None = Field(None, description="The units for magnitude.")


class WeightedFontFamily(BaseModel):
    fontFamily: str | None = Field(
        None,
        description="The font family of the text. The font family can be any font from the Font menu in Docs or from [Google Fonts] (https://fonts.google.com/). If the font name is unrecognized, the text is rendered in `Arial`.",
    )
    weight: int | None = Field(
        None,
        description="The weight of the font. This field can have any value that's a multiple of `100` between `100` and `900`, inclusive. This range corresponds to the numerical values described in the CSS 2.1 Specification, [section 15.6](https://www.w3.org/TR/CSS21/fonts.html#font-boldness), with non-numerical values disallowed. The default value is `400` (\"normal\"). The font weight makes up just one component of the rendered font weight. A combination of the `weight` and the text style's resolved `bold` value determine the rendered weight, after accounting for inheritance: * If the text is bold and the weight is less than `400`, the rendered weight is 400. * If the text is bold and the weight is greater than or equal to `400` but is less than `700`, the rendered weight is `700`. * If the weight is greater than or equal to `700`, the rendered weight is equal to the weight. * If the text is not bold, the rendered weight is equal to the weight.",
    )


class BookmarkLink(BaseModel):
    id: str | None = Field(None, description="The ID of a bookmark in this document.")
    tabId: str | None = Field(None, description="The ID of the tab containing this bookmark.")


class HeadingLink(BaseModel):
    id: str | None = Field(None, description="The ID of a heading in this document.")
    tabId: str | None = Field(None, description="The ID of the tab containing this heading.")


class TextStyleSuggestionState(BaseModel):
    boldSuggested: bool | None = Field(None, description="Indicates if there was a suggested change to bold.")
    italicSuggested: bool | None = Field(None, description="Indicates if there was a suggested change to italic.")
    underlineSuggested: bool | None = Field(None, description="Indicates if there was a suggested change to underline.")
    strikethroughSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to strikethrough."
    )
    smallCapsSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to small_caps."
    )
    backgroundColorSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to background_color."
    )
    foregroundColorSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to foreground_color."
    )
    fontSizeSuggested: bool | None = Field(None, description="Indicates if there was a suggested change to font_size.")
    weightedFontFamilySuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to weighted_font_family."
    )
    baselineOffsetSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to baseline_offset."
    )
    linkSuggested: bool | None = Field(None, description="Indicates if there was a suggested change to link.")


class Type(Enum):
    TYPE_UNSPECIFIED = "TYPE_UNSPECIFIED"
    PAGE_NUMBER = "PAGE_NUMBER"
    PAGE_COUNT = "PAGE_COUNT"


class Equation(BaseModel):
    suggestedInsertionIds: list[str] | None = Field(
        None,
        description="The suggested insertion IDs. An Equation may have multiple insertion IDs if it's a nested suggested change. If empty, then this is not a suggested insertion.",
    )
    suggestedDeletionIds: list[str] | None = Field(
        None, description="The suggested deletion IDs. If empty, then there are no suggested deletions of this content."
    )


class PersonProperties(BaseModel):
    name: str | None = Field(
        None,
        description="The name of the person if it's displayed in the link text instead of the person's email address.",
    )
    email: str | None = Field(
        None, description="The email address linked to this Person. This field is always present."
    )


class RichLinkProperties(BaseModel):
    title: str | None = Field(
        None,
        description="Output only. The title of the RichLink as displayed in the link. This title matches the title of the linked resource at the time of the insertion or last update of the link. This field is always present.",
    )
    uri: str | None = Field(None, description="Output only. The URI to the RichLink. This is always present.")
    mimeType: str | None = Field(
        None,
        description="Output only. The [MIME type](https://developers.google.com/drive/api/v3/mime-types) of the RichLink, if there's one (for example, when it's a file in Drive).",
    )


class NamedStyleType(Enum):
    NAMED_STYLE_TYPE_UNSPECIFIED = "NAMED_STYLE_TYPE_UNSPECIFIED"
    NORMAL_TEXT = "NORMAL_TEXT"
    TITLE = "TITLE"
    SUBTITLE = "SUBTITLE"
    HEADING_1 = "HEADING_1"
    HEADING_2 = "HEADING_2"
    HEADING_3 = "HEADING_3"
    HEADING_4 = "HEADING_4"
    HEADING_5 = "HEADING_5"
    HEADING_6 = "HEADING_6"


class Alignment(Enum):
    ALIGNMENT_UNSPECIFIED = "ALIGNMENT_UNSPECIFIED"
    START = "START"
    CENTER = "CENTER"
    END = "END"
    JUSTIFIED = "JUSTIFIED"


class Direction(Enum):
    CONTENT_DIRECTION_UNSPECIFIED = "CONTENT_DIRECTION_UNSPECIFIED"
    LEFT_TO_RIGHT = "LEFT_TO_RIGHT"
    RIGHT_TO_LEFT = "RIGHT_TO_LEFT"


class SpacingMode(Enum):
    SPACING_MODE_UNSPECIFIED = "SPACING_MODE_UNSPECIFIED"
    NEVER_COLLAPSE = "NEVER_COLLAPSE"
    COLLAPSE_LISTS = "COLLAPSE_LISTS"


class DashStyle(Enum):
    DASH_STYLE_UNSPECIFIED = "DASH_STYLE_UNSPECIFIED"
    SOLID = "SOLID"
    DOT = "DOT"
    DASH = "DASH"


class Alignment1(Enum):
    TAB_STOP_ALIGNMENT_UNSPECIFIED = "TAB_STOP_ALIGNMENT_UNSPECIFIED"
    START = "START"
    CENTER = "CENTER"
    END = "END"


class TabStop(BaseModel):
    class Config:
        use_enum_values = True

    offset: Dimension | None = Field(None, description="The offset between this tab stop and the start margin.")
    alignment: Alignment1 | None = Field(
        None, description="The alignment of this tab stop. If unset, the value defaults to START."
    )


class ShadingSuggestionState(BaseModel):
    backgroundColorSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to the Shading."
    )


class BulletSuggestionState(BaseModel):
    listIdSuggested: bool | None = Field(None, description="Indicates if there was a suggested change to the list_id.")
    nestingLevelSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to the nesting_level."
    )
    textStyleSuggestionState: TextStyleSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields in text style have been changed in this suggestion.",
    )


class ObjectReferences(BaseModel):
    objectIds: list[str] | None = Field(None, description="The object IDs.")


class ColumnSeparatorStyle(Enum):
    COLUMN_SEPARATOR_STYLE_UNSPECIFIED = "COLUMN_SEPARATOR_STYLE_UNSPECIFIED"
    NONE = "NONE"
    BETWEEN_EACH_COLUMN = "BETWEEN_EACH_COLUMN"


class ContentDirection(Enum):
    CONTENT_DIRECTION_UNSPECIFIED = "CONTENT_DIRECTION_UNSPECIFIED"
    LEFT_TO_RIGHT = "LEFT_TO_RIGHT"
    RIGHT_TO_LEFT = "RIGHT_TO_LEFT"


class SectionType(Enum):
    SECTION_TYPE_UNSPECIFIED = "SECTION_TYPE_UNSPECIFIED"
    CONTINUOUS = "CONTINUOUS"
    NEXT_PAGE = "NEXT_PAGE"


class SectionColumnProperties(BaseModel):
    width: Dimension | None = Field(None, description="Output only. The width of the column.")
    paddingEnd: Dimension | None = Field(None, description="The padding at the end of the column.")


class ContentAlignment(Enum):
    CONTENT_ALIGNMENT_UNSPECIFIED = "CONTENT_ALIGNMENT_UNSPECIFIED"
    CONTENT_ALIGNMENT_UNSUPPORTED = "CONTENT_ALIGNMENT_UNSUPPORTED"
    TOP = "TOP"
    MIDDLE = "MIDDLE"
    BOTTOM = "BOTTOM"


class TableCellStyleSuggestionState(BaseModel):
    rowSpanSuggested: bool | None = Field(None, description="Indicates if there was a suggested change to row_span.")
    columnSpanSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to column_span."
    )
    backgroundColorSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to background_color."
    )
    borderLeftSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to border_left."
    )
    borderRightSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to border_right."
    )
    borderTopSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to border_top."
    )
    borderBottomSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to border_bottom."
    )
    paddingLeftSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to padding_left."
    )
    paddingRightSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to padding_right."
    )
    paddingTopSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to padding_top."
    )
    paddingBottomSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to padding_bottom."
    )
    contentAlignmentSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to content_alignment."
    )


class TableRowStyle(BaseModel):
    minRowHeight: Dimension | None = Field(
        None,
        description="The minimum height of the row. The row will be rendered in the Docs editor at a height equal to or greater than this value in order to show all the content in the row's cells.",
    )
    tableHeader: bool | None = Field(None, description="Whether the row is a table header.")
    preventOverflow: bool | None = Field(
        None, description="Whether the row cannot overflow across page or column boundaries."
    )


class TableRowStyleSuggestionState(BaseModel):
    minRowHeightSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to min_row_height."
    )


class WidthType(Enum):
    WIDTH_TYPE_UNSPECIFIED = "WIDTH_TYPE_UNSPECIFIED"
    EVENLY_DISTRIBUTED = "EVENLY_DISTRIBUTED"
    FIXED_WIDTH = "FIXED_WIDTH"


class TableColumnProperties(BaseModel):
    class Config:
        use_enum_values = True

    widthType: WidthType | None = Field(None, description="The width type of the column.")
    width: Dimension | None = Field(
        None, description="The width of the column. Set when the column's `width_type` is FIXED_WIDTH."
    )


class Size(BaseModel):
    height: Dimension | None = Field(None, description="The height of the object.")
    width: Dimension | None = Field(None, description="The width of the object.")


class BackgroundSuggestionState(BaseModel):
    backgroundColorSuggested: bool | None = Field(
        None, description="Indicates whether the current background color has been modified in this suggestion."
    )


class SizeSuggestionState(BaseModel):
    heightSuggested: bool | None = Field(None, description="Indicates if there was a suggested change to height.")
    widthSuggested: bool | None = Field(None, description="Indicates if there was a suggested change to width.")


class BulletAlignment(Enum):
    BULLET_ALIGNMENT_UNSPECIFIED = "BULLET_ALIGNMENT_UNSPECIFIED"
    START = "START"
    CENTER = "CENTER"
    END = "END"


class GlyphType(Enum):
    GLYPH_TYPE_UNSPECIFIED = "GLYPH_TYPE_UNSPECIFIED"
    NONE = "NONE"
    DECIMAL = "DECIMAL"
    ZERO_DECIMAL = "ZERO_DECIMAL"
    UPPER_ALPHA = "UPPER_ALPHA"
    ALPHA = "ALPHA"
    UPPER_ROMAN = "UPPER_ROMAN"
    ROMAN = "ROMAN"


class NestingLevelSuggestionState(BaseModel):
    bulletAlignmentSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to bullet_alignment."
    )
    glyphTypeSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to glyph_type."
    )
    glyphFormatSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to glyph_format."
    )
    glyphSymbolSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to glyph_symbol."
    )
    indentFirstLineSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to indent_first_line."
    )
    indentStartSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to indent_start."
    )
    textStyleSuggestionState: TextStyleSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields in text style have been changed in this suggestion.",
    )
    startNumberSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to start_number."
    )


class Range(BaseModel):
    segmentId: str | None = Field(
        None,
        description="The ID of the header, footer, or footnote that this range is contained in. An empty segment ID signifies the document's body.",
    )
    startIndex: int | None = Field(
        None,
        description="The zero-based start index of this range, in UTF-16 code units. In all current uses, a start index must be provided. This field is an Int32Value in order to accommodate future use cases with open-ended ranges.",
    )
    endIndex: int | None = Field(
        None,
        description="The zero-based end index of this range, exclusive, in UTF-16 code units. In all current uses, an end index must be provided. This field is an Int32Value in order to accommodate future use cases with open-ended ranges.",
    )
    tabId: str | None = Field(
        None,
        description="The tab that contains this range. When omitted, the request applies to the first tab. In a document containing a single tab: - If provided, must match the singular tab's ID. - If omitted, the request applies to the singular tab. In a document containing multiple tabs: - If provided, the request applies to the specified tab. - If omitted, the request applies to the first tab in the document.",
    )


class EmbeddedDrawingProperties(BaseModel):
    pass


class CropProperties(BaseModel):
    offsetLeft: float | None = Field(
        None,
        description="The offset specifies how far inwards the left edge of the crop rectangle is from the left edge of the original content as a fraction of the original content's width.",
    )
    offsetRight: float | None = Field(
        None,
        description="The offset specifies how far inwards the right edge of the crop rectangle is from the right edge of the original content as a fraction of the original content's width.",
    )
    offsetTop: float | None = Field(
        None,
        description="The offset specifies how far inwards the top edge of the crop rectangle is from the top edge of the original content as a fraction of the original content's height.",
    )
    offsetBottom: float | None = Field(
        None,
        description="The offset specifies how far inwards the bottom edge of the crop rectangle is from the bottom edge of the original content as a fraction of the original content's height.",
    )
    angle: float | None = Field(
        None,
        description="The clockwise rotation angle of the crop rectangle around its center, in radians. Rotation is applied after the offsets.",
    )


class PropertyState(Enum):
    RENDERED = "RENDERED"
    NOT_RENDERED = "NOT_RENDERED"


class SheetsChartReference(BaseModel):
    spreadsheetId: str | None = Field(
        None, description="The ID of the Google Sheets spreadsheet that contains the source chart."
    )
    chartId: int | None = Field(
        None, description="The ID of the specific chart in the Google Sheets spreadsheet that's embedded."
    )


class EmbeddedDrawingPropertiesSuggestionState(BaseModel):
    pass


class CropPropertiesSuggestionState(BaseModel):
    offsetLeftSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to offset_left."
    )
    offsetRightSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to offset_right."
    )
    offsetTopSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to offset_top."
    )
    offsetBottomSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to offset_bottom."
    )
    angleSuggested: bool | None = Field(None, description="Indicates if there was a suggested change to angle.")


class EmbeddedObjectBorderSuggestionState(BaseModel):
    colorSuggested: bool | None = Field(None, description="Indicates if there was a suggested change to color.")
    widthSuggested: bool | None = Field(None, description="Indicates if there was a suggested change to width.")
    dashStyleSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to dash_style."
    )
    propertyStateSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to property_state."
    )


class SheetsChartReferenceSuggestionState(BaseModel):
    spreadsheetIdSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to spreadsheet_id."
    )
    chartIdSuggested: bool | None = Field(None, description="Indicates if there was a suggested change to chart_id.")


class Layout(Enum):
    POSITIONED_OBJECT_LAYOUT_UNSPECIFIED = "POSITIONED_OBJECT_LAYOUT_UNSPECIFIED"
    WRAP_TEXT = "WRAP_TEXT"
    BREAK_LEFT = "BREAK_LEFT"
    BREAK_RIGHT = "BREAK_RIGHT"
    BREAK_LEFT_RIGHT = "BREAK_LEFT_RIGHT"
    IN_FRONT_OF_TEXT = "IN_FRONT_OF_TEXT"
    BEHIND_TEXT = "BEHIND_TEXT"


class PositionedObjectPositioning(BaseModel):
    class Config:
        use_enum_values = True

    layout: Layout | None = Field(None, description="The layout of this positioned object.")
    leftOffset: Dimension | None = Field(
        None,
        description="The offset of the left edge of the positioned object relative to the beginning of the Paragraph it's tethered to. The exact positioning of the object can depend on other content in the document and the document's styling.",
    )
    topOffset: Dimension | None = Field(
        None,
        description="The offset of the top edge of the positioned object relative to the beginning of the Paragraph it's tethered to. The exact positioning of the object can depend on other content in the document and the document's styling.",
    )


class PositionedObjectPositioningSuggestionState(BaseModel):
    layoutSuggested: bool | None = Field(None, description="Indicates if there was a suggested change to layout.")
    leftOffsetSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to left_offset."
    )
    topOffsetSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to top_offset."
    )


class SubstringMatchCriteria(BaseModel):
    text: str | None = Field(None, description="The text to search for in the document.")
    matchCase: bool | None = Field(
        None,
        description="Indicates whether the search should respect case: - `True`: the search is case sensitive. - `False`: the search is case insensitive.",
    )
    searchByRegex: bool | None = Field(
        None,
        description="Optional. True if the find value should be treated as a regular expression. Any backslashes in the pattern should be escaped. - `True`: the search text is treated as a regular expressions. - `False`: the search text is treated as a substring for matching.",
    )


class TabsCriteria(BaseModel):
    tabIds: list[str] | None = Field(None, description="The list of tab IDs in which the request executes.")


class Location(BaseModel):
    segmentId: str | None = Field(
        None,
        description="The ID of the header, footer or footnote the location is in. An empty segment ID signifies the document's body.",
    )
    index: int | None = Field(
        None,
        description="The zero-based index, in UTF-16 code units. The index is relative to the beginning of the segment specified by segment_id.",
    )
    tabId: str | None = Field(
        None,
        description="The tab that the location is in. When omitted, the request is applied to the first tab. In a document containing a single tab: - If provided, must match the singular tab's ID. - If omitted, the request applies to the singular tab. In a document containing multiple tabs: - If provided, the request applies to the specified tab. - If omitted, the request applies to the first tab in the document.",
    )


class EndOfSegmentLocation(BaseModel):
    segmentId: str | None = Field(
        None,
        description="The ID of the header, footer or footnote the location is in. An empty segment ID signifies the document's body.",
    )
    tabId: str | None = Field(
        None,
        description="The tab that the location is in. When omitted, the request is applied to the first tab. In a document containing a single tab: - If provided, must match the singular tab's ID. - If omitted, the request applies to the singular tab. In a document containing multiple tabs: - If provided, the request applies to the specified tab. - If omitted, the request applies to the first tab in the document.",
    )


class BulletPreset(Enum):
    BULLET_GLYPH_PRESET_UNSPECIFIED = "BULLET_GLYPH_PRESET_UNSPECIFIED"
    BULLET_DISC_CIRCLE_SQUARE = "BULLET_DISC_CIRCLE_SQUARE"
    BULLET_DIAMONDX_ARROW3D_SQUARE = "BULLET_DIAMONDX_ARROW3D_SQUARE"
    BULLET_CHECKBOX = "BULLET_CHECKBOX"
    BULLET_ARROW_DIAMOND_DISC = "BULLET_ARROW_DIAMOND_DISC"
    BULLET_STAR_CIRCLE_SQUARE = "BULLET_STAR_CIRCLE_SQUARE"
    BULLET_ARROW3D_CIRCLE_SQUARE = "BULLET_ARROW3D_CIRCLE_SQUARE"
    BULLET_LEFTTRIANGLE_DIAMOND_DISC = "BULLET_LEFTTRIANGLE_DIAMOND_DISC"
    BULLET_DIAMONDX_HOLLOWDIAMOND_SQUARE = "BULLET_DIAMONDX_HOLLOWDIAMOND_SQUARE"
    BULLET_DIAMOND_CIRCLE_SQUARE = "BULLET_DIAMOND_CIRCLE_SQUARE"
    NUMBERED_DECIMAL_ALPHA_ROMAN = "NUMBERED_DECIMAL_ALPHA_ROMAN"
    NUMBERED_DECIMAL_ALPHA_ROMAN_PARENS = "NUMBERED_DECIMAL_ALPHA_ROMAN_PARENS"
    NUMBERED_DECIMAL_NESTED = "NUMBERED_DECIMAL_NESTED"
    NUMBERED_UPPERALPHA_ALPHA_ROMAN = "NUMBERED_UPPERALPHA_ALPHA_ROMAN"
    NUMBERED_UPPERROMAN_UPPERALPHA_DECIMAL = "NUMBERED_UPPERROMAN_UPPERALPHA_DECIMAL"
    NUMBERED_ZERODECIMAL_ALPHA_ROMAN = "NUMBERED_ZERODECIMAL_ALPHA_ROMAN"


class CreateParagraphBulletsRequest(BaseModel):
    class Config:
        use_enum_values = True

    range: Range | None = Field(None, description="The range to apply the bullet preset to.")
    bulletPreset: BulletPreset | None = Field(None, description="The kinds of bullet glyphs to be used.")


class DeleteParagraphBulletsRequest(BaseModel):
    range: Range | None = Field(None, description="The range to delete bullets from.")


class CreateNamedRangeRequest(BaseModel):
    name: str | None = Field(
        None,
        description="The name of the NamedRange. Names do not need to be unique. Names must be at least 1 character and no more than 256 characters, measured in UTF-16 code units.",
    )
    range: Range | None = Field(None, description="The range to apply the name to.")


class DeleteNamedRangeRequest(BaseModel):
    namedRangeId: str | None = Field(None, description="The ID of the named range to delete.")
    name: str | None = Field(
        None, description="The name of the range(s) to delete. All named ranges with the given name will be deleted."
    )
    tabsCriteria: TabsCriteria | None = Field(
        None,
        description="Optional. The criteria used to specify which tab(s) the range deletion should occur in. When omitted, the range deletion is applied to all tabs. In a document containing a single tab: - If provided, must match the singular tab's ID. - If omitted, the range deletion applies to the singular tab. In a document containing multiple tabs: - If provided, the range deletion applies to the specified tabs. - If not provided, the range deletion applies to all tabs.",
    )


class DeleteContentRangeRequest(BaseModel):
    range: Range | None = Field(
        None,
        description="The range of content to delete. Deleting text that crosses a paragraph boundary may result in changes to paragraph styles, lists, positioned objects and bookmarks as the two paragraphs are merged. Attempting to delete certain ranges can result in an invalid document structure in which case a 400 bad request error is returned. Some examples of invalid delete requests include: * Deleting one code unit of a surrogate pair. * Deleting the last newline character of a Body, Header, Footer, Footnote, TableCell or TableOfContents. * Deleting the start or end of a Table, TableOfContents or Equation without deleting the entire element. * Deleting the newline character before a Table, TableOfContents or SectionBreak without deleting the element. * Deleting individual rows or cells of a table. Deleting the content within a table cell is allowed.",
    )


class InsertInlineImageRequest(BaseModel):
    location: Location | None = Field(
        None,
        description="Inserts the image at a specific index in the document. The image must be inserted inside the bounds of an existing Paragraph. For instance, it cannot be inserted at a table's start index (i.e. between the table and its preceding paragraph). Inline images cannot be inserted inside a footnote or equation.",
    )
    endOfSegmentLocation: EndOfSegmentLocation | None = Field(
        None,
        description="Inserts the text at the end of a header, footer or the document body. Inline images cannot be inserted inside a footnote.",
    )
    uri: str | None = Field(
        None,
        description="The image URI. The image is fetched once at insertion time and a copy is stored for display inside the document. Images must be less than 50MB in size, cannot exceed 25 megapixels, and must be in one of PNG, JPEG, or GIF format. The provided URI must be publicly accessible and at most 2 kB in length. The URI itself is saved with the image, and exposed via the ImageProperties.content_uri field.",
    )
    objectSize: Size | None = Field(
        None,
        description="The size that the image should appear as in the document. This property is optional and the final size of the image in the document is determined by the following rules: * If neither width nor height is specified, then a default size of the image is calculated based on its resolution. * If one dimension is specified then the other dimension is calculated to preserve the aspect ratio of the image. * If both width and height are specified, the image is scaled to fit within the provided dimensions while maintaining its aspect ratio.",
    )


class InsertTableRequest(BaseModel):
    location: Location | None = Field(
        None,
        description="Inserts the table at a specific model index. A newline character will be inserted before the inserted table, therefore the table start index will be at the specified location index + 1. The table must be inserted inside the bounds of an existing Paragraph. For instance, it cannot be inserted at a table's start index (i.e. between an existing table and its preceding paragraph). Tables cannot be inserted inside a footnote or equation.",
    )
    endOfSegmentLocation: EndOfSegmentLocation | None = Field(
        None,
        description="Inserts the table at the end of the given header, footer or document body. A newline character will be inserted before the inserted table. Tables cannot be inserted inside a footnote.",
    )
    rows: int | None = Field(None, description="The number of rows in the table.")
    columns: int | None = Field(None, description="The number of columns in the table.")


class TableCellLocation(BaseModel):
    tableStartLocation: Location | None = Field(
        None, description="The location where the table starts in the document."
    )
    rowIndex: int | None = Field(
        None, description="The zero-based row index. For example, the second row in the table has a row index of 1."
    )
    columnIndex: int | None = Field(
        None,
        description="The zero-based column index. For example, the second column in the table has a column index of 1.",
    )


class InsertTableColumnRequest(BaseModel):
    tableCellLocation: TableCellLocation | None = Field(
        None,
        description="The reference table cell location from which columns will be inserted. A new column will be inserted to the left (or right) of the column where the reference cell is. If the reference cell is a merged cell, a new column will be inserted to the left (or right) of the merged cell.",
    )
    insertRight: bool | None = Field(
        None,
        description="Whether to insert new column to the right of the reference cell location. - `True`: insert to the right. - `False`: insert to the left.",
    )


class DeleteTableRowRequest(BaseModel):
    tableCellLocation: TableCellLocation | None = Field(
        None,
        description="The reference table cell location from which the row will be deleted. The row this cell spans will be deleted. If this is a merged cell that spans multiple rows, all rows that the cell spans will be deleted. If no rows remain in the table after this deletion, the whole table is deleted.",
    )


class DeleteTableColumnRequest(BaseModel):
    tableCellLocation: TableCellLocation | None = Field(
        None,
        description="The reference table cell location from which the column will be deleted. The column this cell spans will be deleted. If this is a merged cell that spans multiple columns, all columns that the cell spans will be deleted. If no columns remain in the table after this deletion, the whole table is deleted.",
    )


class InsertPageBreakRequest(BaseModel):
    location: Location | None = Field(
        None,
        description="Inserts the page break at a specific index in the document. The page break must be inserted inside the bounds of an existing Paragraph. For instance, it cannot be inserted at a table's start index (i.e. between the table and its preceding paragraph). Page breaks cannot be inserted inside a table, equation, footnote, header or footer. Since page breaks can only be inserted inside the body, the segment ID field must be empty.",
    )
    endOfSegmentLocation: EndOfSegmentLocation | None = Field(
        None,
        description="Inserts the page break at the end of the document body. Page breaks cannot be inserted inside a footnote, header or footer. Since page breaks can only be inserted inside the body, the segment ID field must be empty.",
    )


class DeletePositionedObjectRequest(BaseModel):
    objectId: str | None = Field(None, description="The ID of the positioned object to delete.")
    tabId: str | None = Field(
        None,
        description="The tab that the positioned object to delete is in. When omitted, the request is applied to the first tab. In a document containing a single tab: - If provided, must match the singular tab's ID. - If omitted, the request applies to the singular tab. In a document containing multiple tabs: - If provided, the request applies to the specified tab. - If omitted, the request applies to the first tab in the document.",
    )


class UpdateTableColumnPropertiesRequest(BaseModel):
    tableStartLocation: Location | None = Field(
        None, description="The location where the table starts in the document."
    )
    columnIndices: list[int] | None = Field(
        None,
        description="The list of zero-based column indices whose property should be updated. If no indices are specified, all columns will be updated.",
    )
    tableColumnProperties: TableColumnProperties | None = Field(
        None,
        description="The table column properties to update. If the value of `table_column_properties#width` is less than 5 points (5/72 inch), a 400 bad request error is returned.",
    )
    fields: str | None = Field(
        None,
        description='The fields that should be updated. At least one field must be specified. The root `tableColumnProperties` is implied and should not be specified. A single `"*"` can be used as short-hand for listing every field. For example to update the column width, set `fields` to `"width"`.',
    )


class TableRange(BaseModel):
    tableCellLocation: TableCellLocation | None = Field(
        None, description="The cell location where the table range starts."
    )
    rowSpan: int | None = Field(None, description="The row span of the table range.")
    columnSpan: int | None = Field(None, description="The column span of the table range.")


class UpdateTableRowStyleRequest(BaseModel):
    tableStartLocation: Location | None = Field(
        None, description="The location where the table starts in the document."
    )
    rowIndices: list[int] | None = Field(
        None,
        description="The list of zero-based row indices whose style should be updated. If no indices are specified, all rows will be updated.",
    )
    tableRowStyle: TableRowStyle | None = Field(None, description="The styles to be set on the rows.")
    fields: str | None = Field(
        None,
        description='The fields that should be updated. At least one field must be specified. The root `tableRowStyle` is implied and should not be specified. A single `"*"` can be used as short-hand for listing every field. For example to update the minimum row height, set `fields` to `"min_row_height"`.',
    )


class ImageReplaceMethod(Enum):
    IMAGE_REPLACE_METHOD_UNSPECIFIED = "IMAGE_REPLACE_METHOD_UNSPECIFIED"
    CENTER_CROP = "CENTER_CROP"


class ReplaceImageRequest(BaseModel):
    class Config:
        use_enum_values = True

    imageObjectId: str | None = Field(
        None,
        description="The ID of the existing image that will be replaced. The ID can be retrieved from the response of a get request.",
    )
    uri: str | None = Field(
        None,
        description="The URI of the new image. The image is fetched once at insertion time and a copy is stored for display inside the document. Images must be less than 50MB, cannot exceed 25 megapixels, and must be in PNG, JPEG, or GIF format. The provided URI can't surpass 2 KB in length. The URI is saved with the image, and exposed through the ImageProperties.source_uri field.",
    )
    imageReplaceMethod: ImageReplaceMethod | None = Field(None, description="The replacement method.")
    tabId: str | None = Field(
        None,
        description="The tab that the image to be replaced is in. When omitted, the request is applied to the first tab. In a document containing a single tab: - If provided, must match the singular tab's ID. - If omitted, the request applies to the singular tab. In a document containing multiple tabs: - If provided, the request applies to the specified tab. - If omitted, the request applies to the first tab in the document.",
    )


class MergeTableCellsRequest(BaseModel):
    tableRange: TableRange | None = Field(
        None,
        description='The table range specifying which cells of the table to merge. Any text in the cells being merged will be concatenated and stored in the "head" cell of the range. This is the upper-left cell of the range when the content direction is left to right, and the upper-right cell of the range otherwise. If the range is non-rectangular (which can occur in some cases where the range covers cells that are already merged or where the table is non-rectangular), a 400 bad request error is returned.',
    )


class UnmergeTableCellsRequest(BaseModel):
    tableRange: TableRange | None = Field(
        None,
        description='The table range specifying which cells of the table to unmerge. All merged cells in this range will be unmerged, and cells that are already unmerged will not be affected. If the range has no merged cells, the request will do nothing. If there is text in any of the merged cells, the text will remain in the "head" cell of the resulting block of unmerged cells. The "head" cell is the upper-left cell when the content direction is from left to right, and the upper-right otherwise.',
    )


class Type1(Enum):
    HEADER_FOOTER_TYPE_UNSPECIFIED = "HEADER_FOOTER_TYPE_UNSPECIFIED"
    DEFAULT = "DEFAULT"


class CreateHeaderRequest(BaseModel):
    class Config:
        use_enum_values = True

    type: Type1 | None = Field(None, description="The type of header to create.")
    sectionBreakLocation: Location | None = Field(
        None,
        description="The location of the SectionBreak which begins the section this header should belong to. If `section_break_location` is unset or if it refers to the first section break in the document body, the header applies to the DocumentStyle",
    )


class CreateFooterRequest(BaseModel):
    class Config:
        use_enum_values = True

    type: Type1 | None = Field(None, description="The type of footer to create.")
    sectionBreakLocation: Location | None = Field(
        None,
        description="The location of the SectionBreak immediately preceding the section whose SectionStyle this footer should belong to. If this is unset or refers to the first section break in the document, the footer applies to the document style.",
    )


class CreateFootnoteRequest(BaseModel):
    location: Location | None = Field(
        None,
        description="Inserts the footnote reference at a specific index in the document. The footnote reference must be inserted inside the bounds of an existing Paragraph. For instance, it cannot be inserted at a table's start index (i.e. between the table and its preceding paragraph). Footnote references cannot be inserted inside an equation, header, footer or footnote. Since footnote references can only be inserted in the body, the segment ID field must be empty.",
    )
    endOfSegmentLocation: EndOfSegmentLocation | None = Field(
        None,
        description="Inserts the footnote reference at the end of the document body. Footnote references cannot be inserted inside a header, footer or footnote. Since footnote references can only be inserted in the body, the segment ID field must be empty.",
    )


class ReplaceNamedRangeContentRequest(BaseModel):
    text: str | None = Field(
        None, description="Replaces the content of the specified named range(s) with the given text."
    )
    namedRangeId: str | None = Field(
        None,
        description="The ID of the named range whose content will be replaced. If there is no named range with the given ID a 400 bad request error is returned.",
    )
    namedRangeName: str | None = Field(
        None,
        description="The name of the NamedRanges whose content will be replaced. If there are multiple named ranges with the given name, then the content of each one will be replaced. If there are no named ranges with the given name, then the request will be a no-op.",
    )
    tabsCriteria: TabsCriteria | None = Field(
        None,
        description="Optional. The criteria used to specify in which tabs the replacement occurs. When omitted, the replacement applies to all tabs. In a document containing a single tab: - If provided, must match the singular tab's ID. - If omitted, the replacement applies to the singular tab. In a document containing multiple tabs: - If provided, the replacement applies to the specified tabs. - If omitted, the replacement applies to all tabs.",
    )


class InsertSectionBreakRequest(BaseModel):
    class Config:
        use_enum_values = True

    location: Location | None = Field(
        None,
        description="Inserts a newline and a section break at a specific index in the document. The section break must be inserted inside the bounds of an existing Paragraph. For instance, it cannot be inserted at a table's start index (i.e. between the table and its preceding paragraph). Section breaks cannot be inserted inside a table, equation, footnote, header, or footer. Since section breaks can only be inserted inside the body, the segment ID field must be empty.",
    )
    endOfSegmentLocation: EndOfSegmentLocation | None = Field(
        None,
        description="Inserts a newline and a section break at the end of the document body. Section breaks cannot be inserted inside a footnote, header or footer. Because section breaks can only be inserted inside the body, the segment ID field must be empty.",
    )
    sectionType: SectionType | None = Field(None, description="The type of section to insert.")


class DeleteHeaderRequest(BaseModel):
    headerId: str | None = Field(
        None,
        description="The id of the header to delete. If this header is defined on DocumentStyle, the reference to this header is removed, resulting in no header of that type for the first section of the document. If this header is defined on a SectionStyle, the reference to this header is removed and the header of that type is now continued from the previous section.",
    )
    tabId: str | None = Field(
        None,
        description="The tab containing the header to delete. When omitted, the request is applied to the first tab. In a document containing a single tab: - If provided, must match the singular tab's ID. - If omitted, the request applies to the singular tab. In a document containing multiple tabs: - If provided, the request applies to the specified tab. - If omitted, the request applies to the first tab in the document.",
    )


class DeleteFooterRequest(BaseModel):
    footerId: str | None = Field(
        None,
        description="The id of the footer to delete. If this footer is defined on DocumentStyle, the reference to this footer is removed, resulting in no footer of that type for the first section of the document. If this footer is defined on a SectionStyle, the reference to this footer is removed and the footer of that type is now continued from the previous section.",
    )
    tabId: str | None = Field(
        None,
        description="The tab that contains the footer to delete. When omitted, the request is applied to the first tab. In a document containing a single tab: - If provided, must match the singular tab's ID. - If omitted, the request applies to the singular tab. In a document containing multiple tabs: - If provided, the request applies to the specified tab. - If omitted, the request applies to the first tab in the document.",
    )


class PinTableHeaderRowsRequest(BaseModel):
    tableStartLocation: Location | None = Field(
        None, description="The location where the table starts in the document."
    )
    pinnedHeaderRowsCount: int | None = Field(
        None, description="The number of table rows to pin, where 0 implies that all rows are unpinned."
    )


class InsertPersonRequest(BaseModel):
    location: Location | None = Field(
        None,
        description="Inserts the person at a specific index in the document. The person mention must be inserted inside the bounds of an existing Paragraph. For instance, it cannot be inserted at a table's start index (i.e. between the table and its preceding paragraph). People cannot be inserted inside an equation.",
    )
    endOfSegmentLocation: EndOfSegmentLocation | None = Field(
        None, description="Inserts the person at the end of a header, footer, footnote or the document body."
    )
    personProperties: PersonProperties | None = Field(
        None, description="The properties of the person mention to insert."
    )


class WriteControl(BaseModel):
    requiredRevisionId: str | None = Field(
        None,
        description="The optional revision ID of the document the write request is applied to. If this is not the latest revision of the document, the request is not processed and returns a 400 bad request error. When a required revision ID is returned in a response, it indicates the revision ID of the document after the request was applied.",
    )
    targetRevisionId: str | None = Field(
        None,
        description="The optional target revision ID of the document the write request is applied to. If collaborator changes have occurred after the document was read using the API, the changes produced by this write request are applied against the collaborator changes. This results in a new revision of the document that incorporates both the collaborator changes and the changes in the request, with the Docs server resolving conflicting changes. When using target revision ID, the API client can be thought of as another collaborator of the document. The target revision ID can only be used to write to recent versions of a document. If the target revision is too far behind the latest revision, the request is not processed and returns a 400 bad request error. The request should be tried again after retrieving the latest version of the document. Usually a revision ID remains valid for use as a target revision for several minutes after it's read, but for frequently edited documents this window might be shorter.",
    )


class ReplaceAllTextResponse(BaseModel):
    occurrencesChanged: int | None = Field(None, description="The number of occurrences changed by replacing all text.")


class CreateNamedRangeResponse(BaseModel):
    namedRangeId: str | None = Field(None, description="The ID of the created named range.")


class InsertInlineImageResponse(BaseModel):
    objectId: str | None = Field(None, description="The ID of the created InlineObject.")


class InsertInlineSheetsChartResponse(BaseModel):
    objectId: str | None = Field(None, description="The object ID of the inserted chart.")


class CreateHeaderResponse(BaseModel):
    headerId: str | None = Field(None, description="The ID of the created header.")


class CreateFooterResponse(BaseModel):
    footerId: str | None = Field(None, description="The ID of the created footer.")


class CreateFootnoteResponse(BaseModel):
    footnoteId: str | None = Field(None, description="The ID of the created footnote.")


class Color(BaseModel):
    rgbColor: RgbColor | None = Field(None, description="The RGB color value.")


class Link(BaseModel):
    url: str | None = Field(None, description="An external URL.")
    tabId: str | None = Field(None, description="The ID of a tab in this document.")
    bookmark: BookmarkLink | None = Field(
        None,
        description="A bookmark in this document. In documents containing a single tab, links to bookmarks within the singular tab continue to return Link.bookmarkId when the includeTabsContent parameter is set to `false` or unset. Otherwise, this field is returned.",
    )
    heading: HeadingLink | None = Field(
        None,
        description="A heading in this document. In documents containing a single tab, links to headings within the singular tab continue to return Link.headingId when the includeTabsContent parameter is set to `false` or unset. Otherwise, this field is returned.",
    )
    bookmarkId: str | None = Field(
        None,
        description="The ID of a bookmark in this document. Legacy field: Instead, set includeTabsContent to `true` and use Link.bookmark for read and write operations. This field is only returned when includeTabsContent is set to `false` in documents containing a single tab and links to a bookmark within the singular tab. Otherwise, Link.bookmark is returned. If this field is used in a write request, the bookmark is considered to be from the tab ID specified in the request. If a tab ID is not specified in the request, it is considered to be from the first tab in the document.",
    )
    headingId: str | None = Field(
        None,
        description="The ID of a heading in this document. Legacy field: Instead, set includeTabsContent to `true` and use Link.heading for read and write operations. This field is only returned when includeTabsContent is set to `false` in documents containing a single tab and links to a heading within the singular tab. Otherwise, Link.heading is returned. If this field is used in a write request, the heading is considered to be from the tab ID specified in the request. If a tab ID is not specified in the request, it is considered to be from the first tab in the document.",
    )


class ParagraphStyleSuggestionState(BaseModel):
    headingIdSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to heading_id."
    )
    namedStyleTypeSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to named_style_type."
    )
    alignmentSuggested: bool | None = Field(None, description="Indicates if there was a suggested change to alignment.")
    lineSpacingSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to line_spacing."
    )
    directionSuggested: bool | None = Field(None, description="Indicates if there was a suggested change to direction.")
    spacingModeSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to spacing_mode."
    )
    spaceAboveSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to space_above."
    )
    spaceBelowSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to space_below."
    )
    borderBetweenSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to border_between."
    )
    borderTopSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to border_top."
    )
    borderBottomSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to border_bottom."
    )
    borderLeftSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to border_left."
    )
    borderRightSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to border_right."
    )
    indentFirstLineSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to indent_first_line."
    )
    indentStartSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to indent_start."
    )
    indentEndSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to indent_end."
    )
    keepLinesTogetherSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to keep_lines_together."
    )
    keepWithNextSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to keep_with_next."
    )
    avoidWidowAndOrphanSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to avoid_widow_and_orphan."
    )
    shadingSuggestionState: ShadingSuggestionState | None = Field(
        None, description="A mask that indicates which of the fields in shading have been changed in this suggestion."
    )
    pageBreakBeforeSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to page_break_before."
    )


class SectionStyle(BaseModel):
    class Config:
        use_enum_values = True

    columnProperties: list[SectionColumnProperties] | None = Field(
        None,
        description="The section's columns properties. If empty, the section contains one column with the default properties in the Docs editor. A section can be updated to have no more than 3 columns. When updating this property, setting a concrete value is required. Unsetting this property will result in a 400 bad request error.",
    )
    columnSeparatorStyle: ColumnSeparatorStyle | None = Field(
        None,
        description="The style of column separators. This style can be set even when there's one column in the section. When updating this property, setting a concrete value is required. Unsetting this property results in a 400 bad request error.",
    )
    contentDirection: ContentDirection | None = Field(
        None,
        description="The content direction of this section. If unset, the value defaults to LEFT_TO_RIGHT. When updating this property, setting a concrete value is required. Unsetting this property results in a 400 bad request error.",
    )
    marginTop: Dimension | None = Field(
        None,
        description="The top page margin of the section. If unset, the value defaults to margin_top from DocumentStyle. When updating this property, setting a concrete value is required. Unsetting this property results in a 400 bad request error.",
    )
    marginBottom: Dimension | None = Field(
        None,
        description="The bottom page margin of the section. If unset, the value defaults to margin_bottom from DocumentStyle. When updating this property, setting a concrete value is required. Unsetting this property results in a 400 bad request error.",
    )
    marginRight: Dimension | None = Field(
        None,
        description="The right page margin of the section. If unset, the value defaults to margin_right from DocumentStyle. Updating the right margin causes columns in this section to resize. Since the margin affects column width, it's applied before column properties. When updating this property, setting a concrete value is required. Unsetting this property results in a 400 bad request error.",
    )
    marginLeft: Dimension | None = Field(
        None,
        description="The left page margin of the section. If unset, the value defaults to margin_left from DocumentStyle. Updating the left margin causes columns in this section to resize. Since the margin affects column width, it's applied before column properties. When updating this property, setting a concrete value is required. Unsetting this property results in a 400 bad request error.",
    )
    marginHeader: Dimension | None = Field(
        None,
        description="The header margin of the section. If unset, the value defaults to margin_header from DocumentStyle. If updated, use_custom_header_footer_margins is set to true on DocumentStyle. The value of use_custom_header_footer_margins on DocumentStyle indicates if a header margin is being respected for this section. When updating this property, setting a concrete value is required. Unsetting this property results in a 400 bad request error.",
    )
    marginFooter: Dimension | None = Field(
        None,
        description="The footer margin of the section. If unset, the value defaults to margin_footer from DocumentStyle. If updated, use_custom_header_footer_margins is set to true on DocumentStyle. The value of use_custom_header_footer_margins on DocumentStyle indicates if a footer margin is being respected for this section When updating this property, setting a concrete value is required. Unsetting this property results in a 400 bad request error.",
    )
    sectionType: SectionType | None = Field(None, description="Output only. The type of section.")
    defaultHeaderId: str | None = Field(
        None,
        description="The ID of the default header. If unset, the value inherits from the previous SectionBreak's SectionStyle. If the value is unset in the first SectionBreak, it inherits from DocumentStyle's default_header_id. This property is read-only.",
    )
    defaultFooterId: str | None = Field(
        None,
        description="The ID of the default footer. If unset, the value inherits from the previous SectionBreak's SectionStyle. If the value is unset in the first SectionBreak, it inherits from DocumentStyle's default_footer_id. This property is read-only.",
    )
    firstPageHeaderId: str | None = Field(
        None,
        description="The ID of the header used only for the first page of the section. If use_first_page_header_footer is true, this value is used for the header on the first page of the section. If it's false, the header on the first page of the section uses the default_header_id. If unset, the value inherits from the previous SectionBreak's SectionStyle. If the value is unset in the first SectionBreak, it inherits from DocumentStyle's first_page_header_id. This property is read-only.",
    )
    firstPageFooterId: str | None = Field(
        None,
        description="The ID of the footer used only for the first page of the section. If use_first_page_header_footer is true, this value is used for the footer on the first page of the section. If it's false, the footer on the first page of the section uses the default_footer_id. If unset, the value inherits from the previous SectionBreak's SectionStyle. If the value is unset in the first SectionBreak, it inherits from DocumentStyle's first_page_footer_id. This property is read-only.",
    )
    evenPageHeaderId: str | None = Field(
        None,
        description="The ID of the header used only for even pages. If the value of DocumentStyle's use_even_page_header_footer is true, this value is used for the headers on even pages in the section. If it is false, the headers on even pages use the default_header_id. If unset, the value inherits from the previous SectionBreak's SectionStyle. If the value is unset in the first SectionBreak, it inherits from DocumentStyle's even_page_header_id. This property is read-only.",
    )
    evenPageFooterId: str | None = Field(
        None,
        description="The ID of the footer used only for even pages. If the value of DocumentStyle's use_even_page_header_footer is true, this value is used for the footers on even pages in the section. If it is false, the footers on even pages use the default_footer_id. If unset, the value inherits from the previous SectionBreak's SectionStyle. If the value is unset in the first SectionBreak, it inherits from DocumentStyle's even_page_footer_id. This property is read-only.",
    )
    useFirstPageHeaderFooter: bool | None = Field(
        None,
        description="Indicates whether to use the first page header / footer IDs for the first page of the section. If unset, it inherits from DocumentStyle's use_first_page_header_footer for the first section. If the value is unset for subsequent sectors, it should be interpreted as false. When updating this property, setting a concrete value is required. Unsetting this property results in a 400 bad request error.",
    )
    pageNumberStart: int | None = Field(
        None,
        description="The page number from which to start counting the number of pages for this section. If unset, page numbering continues from the previous section. If the value is unset in the first SectionBreak, refer to DocumentStyle's page_number_start. When updating this property, setting a concrete value is required. Unsetting this property results in a 400 bad request error.",
    )
    flipPageOrientation: bool | None = Field(
        None,
        description="Optional. Indicates whether to flip the dimensions of DocumentStyle's page_size for this section, which allows changing the page orientation between portrait and landscape. If unset, the value inherits from DocumentStyle's flip_page_orientation. When updating this property, setting a concrete value is required. Unsetting this property results in a 400 bad request error.",
    )


class SuggestedTableRowStyle(BaseModel):
    tableRowStyle: TableRowStyle | None = Field(
        None,
        description="A TableRowStyle that only includes the changes made in this suggestion. This can be used along with the table_row_style_suggestion_state to see which fields have changed and their new values.",
    )
    tableRowStyleSuggestionState: TableRowStyleSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields on the base TableRowStyle have been changed in this suggestion.",
    )


class TableStyle(BaseModel):
    tableColumnProperties: list[TableColumnProperties] | None = Field(
        None,
        description="The properties of each column. Note that in Docs, tables contain rows and rows contain cells, similar to HTML. So the properties for a row can be found on the row's table_row_style.",
    )


class DocumentStyleSuggestionState(BaseModel):
    backgroundSuggestionState: BackgroundSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields in background have been changed in this suggestion.",
    )
    defaultHeaderIdSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to default_header_id."
    )
    defaultFooterIdSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to default_footer_id."
    )
    evenPageHeaderIdSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to even_page_header_id."
    )
    evenPageFooterIdSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to even_page_footer_id."
    )
    firstPageHeaderIdSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to first_page_header_id."
    )
    firstPageFooterIdSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to first_page_footer_id."
    )
    useFirstPageHeaderFooterSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to use_first_page_header_footer."
    )
    useEvenPageHeaderFooterSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to use_even_page_header_footer."
    )
    pageNumberStartSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to page_number_start."
    )
    marginTopSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to margin_top."
    )
    marginBottomSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to margin_bottom."
    )
    marginRightSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to margin_right."
    )
    marginLeftSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to margin_left."
    )
    pageSizeSuggestionState: SizeSuggestionState | None = Field(
        None, description="A mask that indicates which of the fields in size have been changed in this suggestion."
    )
    marginHeaderSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to margin_header."
    )
    marginFooterSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to margin_footer."
    )
    useCustomHeaderFooterMarginsSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to use_custom_header_footer_margins."
    )
    flipPageOrientationSuggested: bool | None = Field(
        None, description="Optional. Indicates if there was a suggested change to flip_page_orientation."
    )


class NamedStyleSuggestionState(BaseModel):
    class Config:
        use_enum_values = True

    namedStyleType: NamedStyleType | None = Field(
        None,
        description="The named style type that this suggestion state corresponds to. This field is provided as a convenience for matching the NamedStyleSuggestionState with its corresponding NamedStyle.",
    )
    textStyleSuggestionState: TextStyleSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields in text style have been changed in this suggestion.",
    )
    paragraphStyleSuggestionState: ParagraphStyleSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields in paragraph style have been changed in this suggestion.",
    )


class ListPropertiesSuggestionState(BaseModel):
    nestingLevelsSuggestionStates: list[NestingLevelSuggestionState] | None = Field(
        None,
        description="A mask that indicates which of the fields on the corresponding NestingLevel in nesting_levels have been changed in this suggestion. The nesting level suggestion states are returned in ascending order of the nesting level with the least nested returned first.",
    )


class NamedRange(BaseModel):
    namedRangeId: str | None = Field(None, description="The ID of the named range.")
    name: str | None = Field(None, description="The name of the named range.")
    ranges: list[Range] | None = Field(None, description="The ranges that belong to this named range.")


class ImageProperties(BaseModel):
    contentUri: str | None = Field(
        None,
        description="A URI to the image with a default lifetime of 30 minutes. This URI is tagged with the account of the requester. Anyone with the URI effectively accesses the image as the original requester. Access to the image may be lost if the document's sharing settings change.",
    )
    sourceUri: str | None = Field(
        None, description="The source URI is the URI used to insert the image. The source URI can be empty."
    )
    brightness: float | None = Field(
        None,
        description="The brightness effect of the image. The value should be in the interval [-1.0, 1.0], where 0 means no effect.",
    )
    contrast: float | None = Field(
        None,
        description="The contrast effect of the image. The value should be in the interval [-1.0, 1.0], where 0 means no effect.",
    )
    transparency: float | None = Field(
        None,
        description="The transparency effect of the image. The value should be in the interval [0.0, 1.0], where 0 means no effect and 1 means transparent.",
    )
    cropProperties: CropProperties | None = Field(None, description="The crop properties of the image.")
    angle: float | None = Field(None, description="The clockwise rotation angle of the image, in radians.")


class LinkedContentReference(BaseModel):
    sheetsChartReference: SheetsChartReference | None = Field(None, description="A reference to the linked chart.")


class ImagePropertiesSuggestionState(BaseModel):
    contentUriSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to content_uri."
    )
    sourceUriSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to source_uri."
    )
    brightnessSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to brightness."
    )
    contrastSuggested: bool | None = Field(None, description="Indicates if there was a suggested change to contrast.")
    transparencySuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to transparency."
    )
    cropPropertiesSuggestionState: CropPropertiesSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields in crop_properties have been changed in this suggestion.",
    )
    angleSuggested: bool | None = Field(None, description="Indicates if there was a suggested change to angle.")


class LinkedContentReferenceSuggestionState(BaseModel):
    sheetsChartReferenceSuggestionState: SheetsChartReferenceSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields in sheets_chart_reference have been changed in this suggestion.",
    )


class ReplaceAllTextRequest(BaseModel):
    replaceText: str | None = Field(None, description="The text that will replace the matched text.")
    containsText: SubstringMatchCriteria | None = Field(
        None, description="Finds text in the document matching this substring."
    )
    tabsCriteria: TabsCriteria | None = Field(
        None,
        description="Optional. The criteria used to specify in which tabs the replacement occurs. When omitted, the replacement applies to all tabs. In a document containing a single tab: - If provided, must match the singular tab's ID. - If omitted, the replacement applies to the singular tab. In a document containing multiple tabs: - If provided, the replacement applies to the specified tabs. - If omitted, the replacement applies to all tabs.",
    )


class InsertTextRequest(BaseModel):
    location: Location | None = Field(
        None,
        description="Inserts the text at a specific index in the document. Text must be inserted inside the bounds of an existing Paragraph. For instance, text cannot be inserted at a table's start index (i.e. between the table and its preceding paragraph). The text must be inserted in the preceding paragraph.",
    )
    endOfSegmentLocation: EndOfSegmentLocation | None = Field(
        None, description="Inserts the text at the end of a header, footer, footnote or the document body."
    )
    text: str | None = Field(
        None,
        description="The text to be inserted. Inserting a newline character will implicitly create a new Paragraph at that index. The paragraph style of the new paragraph will be copied from the paragraph at the current insertion index, including lists and bullets. Text styles for inserted text will be determined automatically, generally preserving the styling of neighboring text. In most cases, the text style for the inserted text will match the text immediately before the insertion index. Some control characters (U+0000-U+0008, U+000C-U+001F) and characters from the Unicode Basic Multilingual Plane Private Use Area (U+E000-U+F8FF) will be stripped out of the inserted text.",
    )


class InsertTableRowRequest(BaseModel):
    tableCellLocation: TableCellLocation | None = Field(
        None,
        description="The reference table cell location from which rows will be inserted. A new row will be inserted above (or below) the row where the reference cell is. If the reference cell is a merged cell, a new row will be inserted above (or below) the merged cell.",
    )
    insertBelow: bool | None = Field(
        None,
        description="Whether to insert new row below the reference cell location. - `True`: insert below the cell. - `False`: insert above the cell.",
    )


class UpdateSectionStyleRequest(BaseModel):
    range: Range | None = Field(
        None,
        description="The range overlapping the sections to style. Because section breaks can only be inserted inside the body, the segment ID field must be empty.",
    )
    sectionStyle: SectionStyle | None = Field(
        None,
        description="The styles to be set on the section. Certain section style changes may cause other changes in order to mirror the behavior of the Docs editor. See the documentation of SectionStyle for more information.",
    )
    fields: str | None = Field(
        None,
        description='The fields that should be updated. At least one field must be specified. The root `section_style` is implied and must not be specified. A single `"*"` can be used as short-hand for listing every field. For example to update the left margin, set `fields` to `"margin_left"`.',
    )


class Response(BaseModel):
    replaceAllText: ReplaceAllTextResponse | None = Field(None, description="The result of replacing text.")
    createNamedRange: CreateNamedRangeResponse | None = Field(None, description="The result of creating a named range.")
    insertInlineImage: InsertInlineImageResponse | None = Field(
        None, description="The result of inserting an inline image."
    )
    insertInlineSheetsChart: InsertInlineSheetsChartResponse | None = Field(
        None, description="The result of inserting an inline Google Sheets chart."
    )
    createHeader: CreateHeaderResponse | None = Field(None, description="The result of creating a header.")
    createFooter: CreateFooterResponse | None = Field(None, description="The result of creating a footer.")
    createFootnote: CreateFootnoteResponse | None = Field(None, description="The result of creating a footnote.")


class OptionalColor(BaseModel):
    color: Color | None = Field(
        None, description="If set, this will be used as an opaque color. If unset, this represents a transparent color."
    )


class ParagraphBorder(BaseModel):
    class Config:
        use_enum_values = True

    color: OptionalColor | None = Field(None, description="The color of the border.")
    width: Dimension | None = Field(None, description="The width of the border.")
    padding: Dimension | None = Field(None, description="The padding of the border.")
    dashStyle: DashStyle | None = Field(None, description="The dash style of the border.")


class Shading(BaseModel):
    backgroundColor: OptionalColor | None = Field(None, description="The background color of this paragraph shading.")


class SectionBreak(BaseModel):
    suggestedInsertionIds: list[str] | None = Field(
        None,
        description="The suggested insertion IDs. A SectionBreak may have multiple insertion IDs if it's a nested suggested change. If empty, then this is not a suggested insertion.",
    )
    suggestedDeletionIds: list[str] | None = Field(
        None, description="The suggested deletion IDs. If empty, then there are no suggested deletions of this content."
    )
    sectionStyle: SectionStyle | None = Field(None, description="The style of the section after this section break.")


class TableCellBorder(BaseModel):
    class Config:
        use_enum_values = True

    color: OptionalColor | None = Field(None, description="The color of the border. This color cannot be transparent.")
    width: Dimension | None = Field(None, description="The width of the border.")
    dashStyle: DashStyle | None = Field(None, description="The dash style of the border.")


class Background(BaseModel):
    color: OptionalColor | None = Field(None, description="The background color.")


class NamedStylesSuggestionState(BaseModel):
    stylesSuggestionStates: list[NamedStyleSuggestionState] | None = Field(
        None,
        description="A mask that indicates which of the fields on the corresponding NamedStyle in styles have been changed in this suggestion. The order of these named style suggestion states matches the order of the corresponding named style within the named styles suggestion.",
    )


class NamedRanges(BaseModel):
    name: str | None = Field(None, description="The name that all the named ranges share.")
    namedRanges: list[NamedRange] | None = Field(None, description="The NamedRanges that share the same name.")


class EmbeddedObjectBorder(BaseModel):
    class Config:
        use_enum_values = True

    color: OptionalColor | None = Field(None, description="The color of the border.")
    width: Dimension | None = Field(None, description="The width of the border.")
    dashStyle: DashStyle | None = Field(None, description="The dash style of the border.")
    propertyState: PropertyState | None = Field(None, description="The property state of the border property.")


class EmbeddedObjectSuggestionState(BaseModel):
    embeddedDrawingPropertiesSuggestionState: EmbeddedDrawingPropertiesSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields in embedded_drawing_properties have been changed in this suggestion.",
    )
    imagePropertiesSuggestionState: ImagePropertiesSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields in image_properties have been changed in this suggestion.",
    )
    titleSuggested: bool | None = Field(None, description="Indicates if there was a suggested change to title.")
    descriptionSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to description."
    )
    embeddedObjectBorderSuggestionState: EmbeddedObjectBorderSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields in embedded_object_border have been changed in this suggestion.",
    )
    sizeSuggestionState: SizeSuggestionState | None = Field(
        None, description="A mask that indicates which of the fields in size have been changed in this suggestion."
    )
    marginLeftSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to margin_left."
    )
    marginRightSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to margin_right."
    )
    marginTopSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to margin_top."
    )
    marginBottomSuggested: bool | None = Field(
        None, description="Indicates if there was a suggested change to margin_bottom."
    )
    linkedContentReferenceSuggestionState: LinkedContentReferenceSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields in linked_content_reference have been changed in this suggestion.",
    )


class PositionedObjectPropertiesSuggestionState(BaseModel):
    positioningSuggestionState: PositionedObjectPositioningSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields in positioning have been changed in this suggestion.",
    )
    embeddedObjectSuggestionState: EmbeddedObjectSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields in embedded_object have been changed in this suggestion.",
    )


class BatchUpdateDocumentResponse(BaseModel):
    documentId: str | None = Field(None, description="The ID of the document to which the updates were applied to.")
    replies: list[Response] | None = Field(
        None,
        description="The reply of the updates. This maps 1:1 with the updates, although replies to some requests may be empty.",
    )
    writeControl: WriteControl | None = Field(None, description="The updated write control after applying the request.")


class TextStyle(BaseModel):
    class Config:
        use_enum_values = True

    bold: bool | None = Field(None, description="Whether or not the text is rendered as bold.")
    italic: bool | None = Field(None, description="Whether or not the text is italicized.")
    underline: bool | None = Field(None, description="Whether or not the text is underlined.")
    strikethrough: bool | None = Field(None, description="Whether or not the text is struck through.")
    smallCaps: bool | None = Field(None, description="Whether or not the text is in small capital letters.")
    backgroundColor: OptionalColor | None = Field(
        None,
        description="The background color of the text. If set, the color is either an RGB color or transparent, depending on the `color` field.",
    )
    foregroundColor: OptionalColor | None = Field(
        None,
        description="The foreground color of the text. If set, the color is either an RGB color or transparent, depending on the `color` field.",
    )
    fontSize: Dimension | None = Field(None, description="The size of the text's font.")
    weightedFontFamily: WeightedFontFamily | None = Field(
        None,
        description="The font family and rendered weight of the text. If an update request specifies values for both `weighted_font_family` and `bold`, the `weighted_font_family` is applied first, then `bold`. If `weighted_font_family#weight` is not set, it defaults to `400`. If `weighted_font_family` is set, then `weighted_font_family#font_family` must also be set with a non-empty value. Otherwise, a 400 bad request error is returned.",
    )
    baselineOffset: BaselineOffset | None = Field(
        None,
        description="The text's vertical offset from its normal position. Text with `SUPERSCRIPT` or `SUBSCRIPT` baseline offsets is automatically rendered in a smaller font size, computed based on the `font_size` field. Changes in this field don't affect the `font_size`.",
    )
    link: Link | None = Field(
        None,
        description='The hyperlink destination of the text. If unset, there\'s no link. Links are not inherited from parent text. Changing the link in an update request causes some other changes to the text style of the range: * When setting a link, the text foreground color will be updated to the default link color and the text will be underlined. If these fields are modified in the same request, those values will be used instead of the link defaults. * Setting a link on a text range that overlaps with an existing link will also update the existing link to point to the new URL. * Links are not settable on newline characters. As a result, setting a link on a text range that crosses a paragraph boundary, such as `"ABC\\n123"`, will separate the newline character(s) into their own text runs. The link will be applied separately to the runs before and after the newline. * Removing a link will update the text style of the range to match the style of the preceding text (or the default text styles if the preceding text is another link) unless different styles are being set in the same request.',
    )


class SuggestedTextStyle(BaseModel):
    textStyle: TextStyle | None = Field(
        None,
        description="A TextStyle that only includes the changes made in this suggestion. This can be used along with the text_style_suggestion_state to see which fields have changed and their new values.",
    )
    textStyleSuggestionState: TextStyleSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields on the base TextStyle have been changed in this suggestion.",
    )


class AutoText(BaseModel):
    class Config:
        use_enum_values = True

    type: Type | None = Field(None, description="The type of this auto text.")
    suggestedInsertionIds: list[str] | None = Field(
        None,
        description="The suggested insertion IDs. An AutoText may have multiple insertion IDs if it's a nested suggested change. If empty, then this is not a suggested insertion.",
    )
    suggestedDeletionIds: list[str] | None = Field(
        None, description="The suggested deletion IDs. If empty, then there are no suggested deletions of this content."
    )
    textStyle: TextStyle | None = Field(None, description="The text style of this AutoText.")
    suggestedTextStyleChanges: dict[str, SuggestedTextStyle] | None = Field(
        None, description="The suggested text style changes to this AutoText, keyed by suggestion ID."
    )


class PageBreak(BaseModel):
    suggestedInsertionIds: list[str] | None = Field(
        None,
        description="The suggested insertion IDs. A PageBreak may have multiple insertion IDs if it's a nested suggested change. If empty, then this is not a suggested insertion.",
    )
    suggestedDeletionIds: list[str] | None = Field(
        None, description="The suggested deletion IDs. If empty, then there are no suggested deletions of this content."
    )
    textStyle: TextStyle | None = Field(
        None,
        description="The text style of this PageBreak. Similar to text content, like text runs and footnote references, the text style of a page break can affect content layout as well as the styling of text inserted next to it.",
    )
    suggestedTextStyleChanges: dict[str, SuggestedTextStyle] | None = Field(
        None, description="The suggested text style changes to this PageBreak, keyed by suggestion ID."
    )


class ColumnBreak(BaseModel):
    suggestedInsertionIds: list[str] | None = Field(
        None,
        description="The suggested insertion IDs. A ColumnBreak may have multiple insertion IDs if it's a nested suggested change. If empty, then this is not a suggested insertion.",
    )
    suggestedDeletionIds: list[str] | None = Field(
        None, description="The suggested deletion IDs. If empty, then there are no suggested deletions of this content."
    )
    textStyle: TextStyle | None = Field(
        None,
        description="The text style of this ColumnBreak. Similar to text content, like text runs and footnote references, the text style of a column break can affect content layout as well as the styling of text inserted next to it.",
    )
    suggestedTextStyleChanges: dict[str, SuggestedTextStyle] | None = Field(
        None, description="The suggested text style changes to this ColumnBreak, keyed by suggestion ID."
    )


class FootnoteReference(BaseModel):
    footnoteId: str | None = Field(
        None, description="The ID of the footnote that contains the content of this footnote reference."
    )
    footnoteNumber: str | None = Field(None, description="The rendered number of this footnote.")
    suggestedInsertionIds: list[str] | None = Field(
        None,
        description="The suggested insertion IDs. A FootnoteReference may have multiple insertion IDs if it's a nested suggested change. If empty, then this is not a suggested insertion.",
    )
    suggestedDeletionIds: list[str] | None = Field(
        None, description="The suggested deletion IDs. If empty, then there are no suggested deletions of this content."
    )
    textStyle: TextStyle | None = Field(None, description="The text style of this FootnoteReference.")
    suggestedTextStyleChanges: dict[str, SuggestedTextStyle] | None = Field(
        None, description="The suggested text style changes to this FootnoteReference, keyed by suggestion ID."
    )


class HorizontalRule(BaseModel):
    suggestedInsertionIds: list[str] | None = Field(
        None,
        description="The suggested insertion IDs. A HorizontalRule may have multiple insertion IDs if it is a nested suggested change. If empty, then this is not a suggested insertion.",
    )
    suggestedDeletionIds: list[str] | None = Field(
        None, description="The suggested deletion IDs. If empty, then there are no suggested deletions of this content."
    )
    textStyle: TextStyle | None = Field(
        None,
        description="The text style of this HorizontalRule. Similar to text content, like text runs and footnote references, the text style of a horizontal rule can affect content layout as well as the styling of text inserted next to it.",
    )
    suggestedTextStyleChanges: dict[str, SuggestedTextStyle] | None = Field(
        None, description="The suggested text style changes to this HorizontalRule, keyed by suggestion ID."
    )


class InlineObjectElement(BaseModel):
    inlineObjectId: str | None = Field(None, description="The ID of the InlineObject this element contains.")
    suggestedInsertionIds: list[str] | None = Field(
        None,
        description="The suggested insertion IDs. An InlineObjectElement may have multiple insertion IDs if it's a nested suggested change. If empty, then this is not a suggested insertion.",
    )
    suggestedDeletionIds: list[str] | None = Field(
        None, description="The suggested deletion IDs. If empty, then there are no suggested deletions of this content."
    )
    textStyle: TextStyle | None = Field(
        None,
        description="The text style of this InlineObjectElement. Similar to text content, like text runs and footnote references, the text style of an inline object element can affect content layout as well as the styling of text inserted next to it.",
    )
    suggestedTextStyleChanges: dict[str, SuggestedTextStyle] | None = Field(
        None, description="The suggested text style changes to this InlineObject, keyed by suggestion ID."
    )


class Person(BaseModel):
    personId: str | None = Field(None, description="Output only. The unique ID of this link.")
    suggestedInsertionIds: list[str] | None = Field(
        None,
        description="IDs for suggestions that insert this person link into the document. A Person might have multiple insertion IDs if it's a nested suggested change (a suggestion within a suggestion made by a different user, for example). If empty, then this person link isn't a suggested insertion.",
    )
    suggestedDeletionIds: list[str] | None = Field(
        None,
        description="IDs for suggestions that remove this person link from the document. A Person might have multiple deletion IDs if, for example, multiple users suggest deleting it. If empty, then this person link isn't suggested for deletion.",
    )
    textStyle: TextStyle | None = Field(None, description="The text style of this Person.")
    suggestedTextStyleChanges: dict[str, SuggestedTextStyle] | None = Field(
        None, description="The suggested text style changes to this Person, keyed by suggestion ID."
    )
    personProperties: PersonProperties | None = Field(
        None, description="Output only. The properties of this Person. This field is always present."
    )


class RichLink(BaseModel):
    richLinkId: str | None = Field(None, description="Output only. The ID of this link.")
    suggestedInsertionIds: list[str] | None = Field(
        None,
        description="IDs for suggestions that insert this link into the document. A RichLink might have multiple insertion IDs if it's a nested suggested change (a suggestion within a suggestion made by a different user, for example). If empty, then this person link isn't a suggested insertion.",
    )
    suggestedDeletionIds: list[str] | None = Field(
        None,
        description="IDs for suggestions that remove this link from the document. A RichLink might have multiple deletion IDs if, for example, multiple users suggest deleting it. If empty, then this person link isn't suggested for deletion.",
    )
    textStyle: TextStyle | None = Field(None, description="The text style of this RichLink.")
    suggestedTextStyleChanges: dict[str, SuggestedTextStyle] | None = Field(
        None, description="The suggested text style changes to this RichLink, keyed by suggestion ID."
    )
    richLinkProperties: RichLinkProperties | None = Field(
        None, description="Output only. The properties of this RichLink. This field is always present."
    )


class ParagraphStyle(BaseModel):
    class Config:
        use_enum_values = True

    headingId: str | None = Field(
        None,
        description="The heading ID of the paragraph. If empty, then this paragraph is not a heading. This property is read-only.",
    )
    namedStyleType: NamedStyleType | None = Field(
        None,
        description="The named style type of the paragraph. Since updating the named style type affects other properties within ParagraphStyle, the named style type is applied before the other properties are updated.",
    )
    alignment: Alignment | None = Field(None, description="The text alignment for this paragraph.")
    lineSpacing: float | None = Field(
        None,
        description="The amount of space between lines, as a percentage of normal, where normal is represented as 100.0. If unset, the value is inherited from the parent.",
    )
    direction: Direction | None = Field(
        None,
        description="The text direction of this paragraph. If unset, the value defaults to LEFT_TO_RIGHT since paragraph direction is not inherited.",
    )
    spacingMode: SpacingMode | None = Field(None, description="The spacing mode for the paragraph.")
    spaceAbove: Dimension | None = Field(
        None,
        description="The amount of extra space above the paragraph. If unset, the value is inherited from the parent.",
    )
    spaceBelow: Dimension | None = Field(
        None,
        description="The amount of extra space below the paragraph. If unset, the value is inherited from the parent.",
    )
    borderBetween: ParagraphBorder | None = Field(
        None,
        description="The border between this paragraph and the next and previous paragraphs. If unset, the value is inherited from the parent. The between border is rendered when the adjacent paragraph has the same border and indent properties. Paragraph borders cannot be partially updated. When changing a paragraph border, the new border must be specified in its entirety.",
    )
    borderTop: ParagraphBorder | None = Field(
        None,
        description="The border at the top of this paragraph. If unset, the value is inherited from the parent. The top border is rendered when the paragraph above has different border and indent properties. Paragraph borders cannot be partially updated. When changing a paragraph border, the new border must be specified in its entirety.",
    )
    borderBottom: ParagraphBorder | None = Field(
        None,
        description="The border at the bottom of this paragraph. If unset, the value is inherited from the parent. The bottom border is rendered when the paragraph below has different border and indent properties. Paragraph borders cannot be partially updated. When changing a paragraph border, the new border must be specified in its entirety.",
    )
    borderLeft: ParagraphBorder | None = Field(
        None,
        description="The border to the left of this paragraph. If unset, the value is inherited from the parent. Paragraph borders cannot be partially updated. When changing a paragraph border, the new border must be specified in its entirety.",
    )
    borderRight: ParagraphBorder | None = Field(
        None,
        description="The border to the right of this paragraph. If unset, the value is inherited from the parent. Paragraph borders cannot be partially updated. When changing a paragraph border, the new border must be specified in its entirety.",
    )
    indentFirstLine: Dimension | None = Field(
        None,
        description="The amount of indentation for the first line of the paragraph. If unset, the value is inherited from the parent.",
    )
    indentStart: Dimension | None = Field(
        None,
        description="The amount of indentation for the paragraph on the side that corresponds to the start of the text, based on the current paragraph direction. If unset, the value is inherited from the parent.",
    )
    indentEnd: Dimension | None = Field(
        None,
        description="The amount of indentation for the paragraph on the side that corresponds to the end of the text, based on the current paragraph direction. If unset, the value is inherited from the parent.",
    )
    tabStops: list[TabStop] | None = Field(
        None,
        description="A list of the tab stops for this paragraph. The list of tab stops is not inherited. This property is read-only.",
    )
    keepLinesTogether: bool | None = Field(
        None,
        description="Whether all lines of the paragraph should be laid out on the same page or column if possible. If unset, the value is inherited from the parent.",
    )
    keepWithNext: bool | None = Field(
        None,
        description="Whether at least a part of this paragraph should be laid out on the same page or column as the next paragraph if possible. If unset, the value is inherited from the parent.",
    )
    avoidWidowAndOrphan: bool | None = Field(
        None,
        description="Whether to avoid widows and orphans for the paragraph. If unset, the value is inherited from the parent.",
    )
    shading: Shading | None = Field(
        None, description="The shading of the paragraph. If unset, the value is inherited from the parent."
    )
    pageBreakBefore: bool | None = Field(
        None,
        description="Whether the current paragraph should always start at the beginning of a page. If unset, the value is inherited from the parent. Attempting to update page_break_before for paragraphs in unsupported regions, including Table, Header, Footer and Footnote, can result in an invalid document state that returns a 400 bad request error.",
    )


class SuggestedParagraphStyle(BaseModel):
    paragraphStyle: ParagraphStyle | None = Field(
        None,
        description="A ParagraphStyle that only includes the changes made in this suggestion. This can be used along with the paragraph_style_suggestion_state to see which fields have changed and their new values.",
    )
    paragraphStyleSuggestionState: ParagraphStyleSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields on the base ParagraphStyle have been changed in this suggestion.",
    )


class Bullet(BaseModel):
    listId: str | None = Field(None, description="The ID of the list this paragraph belongs to.")
    nestingLevel: int | None = Field(None, description="The nesting level of this paragraph in the list.")
    textStyle: TextStyle | None = Field(None, description="The paragraph-specific text style applied to this bullet.")


class SuggestedBullet(BaseModel):
    bullet: Bullet | None = Field(
        None,
        description="A Bullet that only includes the changes made in this suggestion. This can be used along with the bullet_suggestion_state to see which fields have changed and their new values.",
    )
    bulletSuggestionState: BulletSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields on the base Bullet have been changed in this suggestion.",
    )


class TableCellStyle(BaseModel):
    class Config:
        use_enum_values = True

    rowSpan: int | None = Field(None, description="The row span of the cell. This property is read-only.")
    columnSpan: int | None = Field(None, description="The column span of the cell. This property is read-only.")
    backgroundColor: OptionalColor | None = Field(None, description="The background color of the cell.")
    borderLeft: TableCellBorder | None = Field(None, description="The left border of the cell.")
    borderRight: TableCellBorder | None = Field(None, description="The right border of the cell.")
    borderTop: TableCellBorder | None = Field(None, description="The top border of the cell.")
    borderBottom: TableCellBorder | None = Field(None, description="The bottom border of the cell.")
    paddingLeft: Dimension | None = Field(None, description="The left padding of the cell.")
    paddingRight: Dimension | None = Field(None, description="The right padding of the cell.")
    paddingTop: Dimension | None = Field(None, description="The top padding of the cell.")
    paddingBottom: Dimension | None = Field(None, description="The bottom padding of the cell.")
    contentAlignment: ContentAlignment | None = Field(
        None,
        description="The alignment of the content in the table cell. The default alignment matches the alignment for newly created table cells in the Docs editor.",
    )


class SuggestedTableCellStyle(BaseModel):
    tableCellStyle: TableCellStyle | None = Field(
        None,
        description="A TableCellStyle that only includes the changes made in this suggestion. This can be used along with the table_cell_style_suggestion_state to see which fields have changed and their new values.",
    )
    tableCellStyleSuggestionState: TableCellStyleSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields on the base TableCellStyle have been changed in this suggestion.",
    )


class DocumentStyle(BaseModel):
    background: Background | None = Field(
        None, description="The background of the document. Documents cannot have a transparent background color."
    )
    defaultHeaderId: str | None = Field(
        None,
        description="The ID of the default header. If not set, there's no default header. This property is read-only.",
    )
    defaultFooterId: str | None = Field(
        None,
        description="The ID of the default footer. If not set, there's no default footer. This property is read-only.",
    )
    evenPageHeaderId: str | None = Field(
        None,
        description="The ID of the header used only for even pages. The value of use_even_page_header_footer determines whether to use the default_header_id or this value for the header on even pages. If not set, there's no even page header. This property is read-only.",
    )
    evenPageFooterId: str | None = Field(
        None,
        description="The ID of the footer used only for even pages. The value of use_even_page_header_footer determines whether to use the default_footer_id or this value for the footer on even pages. If not set, there's no even page footer. This property is read-only.",
    )
    firstPageHeaderId: str | None = Field(
        None,
        description="The ID of the header used only for the first page. If not set then a unique header for the first page does not exist. The value of use_first_page_header_footer determines whether to use the default_header_id or this value for the header on the first page. If not set, there's no first page header. This property is read-only.",
    )
    firstPageFooterId: str | None = Field(
        None,
        description="The ID of the footer used only for the first page. If not set then a unique footer for the first page does not exist. The value of use_first_page_header_footer determines whether to use the default_footer_id or this value for the footer on the first page. If not set, there's no first page footer. This property is read-only.",
    )
    useFirstPageHeaderFooter: bool | None = Field(
        None, description="Indicates whether to use the first page header / footer IDs for the first page."
    )
    useEvenPageHeaderFooter: bool | None = Field(
        None, description="Indicates whether to use the even page header / footer IDs for the even pages."
    )
    pageNumberStart: int | None = Field(
        None, description="The page number from which to start counting the number of pages."
    )
    marginTop: Dimension | None = Field(
        None,
        description="The top page margin. Updating the top page margin on the document style clears the top page margin on all section styles.",
    )
    marginBottom: Dimension | None = Field(
        None,
        description="The bottom page margin. Updating the bottom page margin on the document style clears the bottom page margin on all section styles.",
    )
    marginRight: Dimension | None = Field(
        None,
        description="The right page margin. Updating the right page margin on the document style clears the right page margin on all section styles. It may also cause columns to resize in all sections.",
    )
    marginLeft: Dimension | None = Field(
        None,
        description="The left page margin. Updating the left page margin on the document style clears the left page margin on all section styles. It may also cause columns to resize in all sections.",
    )
    pageSize: Size | None = Field(None, description="The size of a page in the document.")
    marginHeader: Dimension | None = Field(
        None, description="The amount of space between the top of the page and the contents of the header."
    )
    marginFooter: Dimension | None = Field(
        None, description="The amount of space between the bottom of the page and the contents of the footer."
    )
    useCustomHeaderFooterMargins: bool | None = Field(
        None,
        description="Indicates whether DocumentStyle margin_header, SectionStyle margin_header and DocumentStyle margin_footer, SectionStyle margin_footer are respected. When false, the default values in the Docs editor for header and footer margin is used. This property is read-only.",
    )
    flipPageOrientation: bool | None = Field(
        None,
        description="Optional. Indicates whether to flip the dimensions of the page_size, which allows changing the page orientation between portrait and landscape.",
    )


class SuggestedDocumentStyle(BaseModel):
    documentStyle: DocumentStyle | None = Field(
        None,
        description="A DocumentStyle that only includes the changes made in this suggestion. This can be used along with the document_style_suggestion_state to see which fields have changed and their new values.",
    )
    documentStyleSuggestionState: DocumentStyleSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields on the base DocumentStyle have been changed in this suggestion.",
    )


class NamedStyle(BaseModel):
    class Config:
        use_enum_values = True

    namedStyleType: NamedStyleType | None = Field(None, description="The type of this named style.")
    textStyle: TextStyle | None = Field(None, description="The text style of this named style.")
    paragraphStyle: ParagraphStyle | None = Field(None, description="The paragraph style of this named style.")


class NestingLevel(BaseModel):
    class Config:
        use_enum_values = True

    bulletAlignment: BulletAlignment | None = Field(
        None, description="The alignment of the bullet within the space allotted for rendering the bullet."
    )
    glyphType: GlyphType | None = Field(
        None,
        description="The type of glyph used by bullets when paragraphs at this level of nesting is ordered. The glyph type determines the type of glyph used to replace placeholders within the glyph_format when paragraphs at this level of nesting are ordered. For example, if the nesting level is 0, the glyph_format is `%0.` and the glyph type is DECIMAL, then the rendered glyph would replace the placeholder `%0` in the glyph format with a number corresponding to the list item's order within the list.",
    )
    glyphSymbol: str | None = Field(
        None,
        description="A custom glyph symbol used by bullets when paragraphs at this level of nesting is unordered. The glyph symbol replaces placeholders within the glyph_format. For example, if the glyph_symbol is the solid circle corresponding to Unicode U+25cf code point and the glyph_format is `%0`, the rendered glyph would be the solid circle.",
    )
    glyphFormat: str | None = Field(
        None,
        description="The format string used by bullets at this level of nesting. The glyph format contains one or more placeholders, and these placeholders are replaced with the appropriate values depending on the glyph_type or glyph_symbol. The placeholders follow the pattern `%[nesting_level]`. Furthermore, placeholders can have prefixes and suffixes. Thus, the glyph format follows the pattern `%[nesting_level]`. Note that the prefix and suffix are optional and can be arbitrary strings. For example, the glyph format `%0.` indicates that the rendered glyph will replace the placeholder with the corresponding glyph for nesting level 0 followed by a period as the suffix. So a list with a glyph type of UPPER_ALPHA and glyph format `%0.` at nesting level 0 will result in a list with rendered glyphs `A.` `B.` `C.` The glyph format can contain placeholders for the current nesting level as well as placeholders for parent nesting levels. For example, a list can have a glyph format of `%0.` at nesting level 0 and a glyph format of `%0.%1.` at nesting level 1. Assuming both nesting levels have DECIMAL glyph types, this would result in a list with rendered glyphs `1.` `2.` ` 2.1.` ` 2.2.` `3.` For nesting levels that are ordered, the string that replaces a placeholder in the glyph format for a particular paragraph depends on the paragraph's order within the list.",
    )
    indentFirstLine: Dimension | None = Field(
        None, description="The amount of indentation for the first line of paragraphs at this level of nesting."
    )
    indentStart: Dimension | None = Field(
        None,
        description="The amount of indentation for paragraphs at this level of nesting. Applied to the side that corresponds to the start of the text, based on the paragraph's content direction.",
    )
    textStyle: TextStyle | None = Field(None, description="The text style of bullets at this level of nesting.")
    startNumber: int | None = Field(
        None,
        description="The number of the first list item at this nesting level. A value of 0 is treated as a value of 1 for lettered lists and Roman numeral lists. For values of both 0 and 1, lettered and Roman numeral lists will begin at `a` and `i` respectively. This value is ignored for nesting levels with unordered glyphs.",
    )


class EmbeddedObject(BaseModel):
    embeddedDrawingProperties: EmbeddedDrawingProperties | None = Field(
        None, description="The properties of an embedded drawing."
    )
    imageProperties: ImageProperties | None = Field(None, description="The properties of an image.")
    title: str | None = Field(
        None,
        description="The title of the embedded object. The `title` and `description` are both combined to display alt text.",
    )
    description: str | None = Field(
        None,
        description="The description of the embedded object. The `title` and `description` are both combined to display alt text.",
    )
    embeddedObjectBorder: EmbeddedObjectBorder | None = Field(None, description="The border of the embedded object.")
    size: Size | None = Field(None, description="The visible size of the image after cropping.")
    marginTop: Dimension | None = Field(None, description="The top margin of the embedded object.")
    marginBottom: Dimension | None = Field(None, description="The bottom margin of the embedded object.")
    marginRight: Dimension | None = Field(None, description="The right margin of the embedded object.")
    marginLeft: Dimension | None = Field(None, description="The left margin of the embedded object.")
    linkedContentReference: LinkedContentReference | None = Field(
        None,
        description="A reference to the external linked source content. For example, it contains a reference to the source Google Sheets chart when the embedded object is a linked chart. If unset, then the embedded object is not linked.",
    )


class InlineObjectPropertiesSuggestionState(BaseModel):
    embeddedObjectSuggestionState: EmbeddedObjectSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields in embedded_object have been changed in this suggestion.",
    )


class PositionedObjectProperties(BaseModel):
    positioning: PositionedObjectPositioning | None = Field(
        None,
        description="The positioning of this positioned object relative to the newline of the Paragraph that references this positioned object.",
    )
    embeddedObject: EmbeddedObject | None = Field(None, description="The embedded object of this positioned object.")


class SuggestedPositionedObjectProperties(BaseModel):
    positionedObjectProperties: PositionedObjectProperties | None = Field(
        None,
        description="A PositionedObjectProperties that only includes the changes made in this suggestion. This can be used along with the positioned_object_properties_suggestion_state to see which fields have changed and their new values.",
    )
    positionedObjectPropertiesSuggestionState: PositionedObjectPropertiesSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields on the base PositionedObjectProperties have been changed in this suggestion.",
    )


class UpdateTextStyleRequest(BaseModel):
    range: Range | None = Field(
        None,
        description="The range of text to style. The range may be extended to include adjacent newlines. If the range fully contains a paragraph belonging to a list, the paragraph's bullet is also updated with the matching text style. Ranges cannot be inserted inside a relative UpdateTextStyleRequest.",
    )
    textStyle: TextStyle | None = Field(
        None,
        description="The styles to set on the text. If the value for a particular style matches that of the parent, that style will be set to inherit. Certain text style changes may cause other changes in order to to mirror the behavior of the Docs editor. See the documentation of TextStyle for more information.",
    )
    fields: str | None = Field(
        None,
        description='The fields that should be updated. At least one field must be specified. The root `text_style` is implied and should not be specified. A single `"*"` can be used as short-hand for listing every field. For example, to update the text style to bold, set `fields` to `"bold"`. To reset a property to its default value, include its field name in the field mask but leave the field itself unset.',
    )


class UpdateParagraphStyleRequest(BaseModel):
    range: Range | None = Field(None, description="The range overlapping the paragraphs to style.")
    paragraphStyle: ParagraphStyle | None = Field(
        None,
        description="The styles to set on the paragraphs. Certain paragraph style changes may cause other changes in order to mirror the behavior of the Docs editor. See the documentation of ParagraphStyle for more information.",
    )
    fields: str | None = Field(
        None,
        description='The fields that should be updated. At least one field must be specified. The root `paragraph_style` is implied and should not be specified. A single `"*"` can be used as short-hand for listing every field. For example, to update the paragraph style\'s alignment property, set `fields` to `"alignment"`. To reset a property to its default value, include its field name in the field mask but leave the field itself unset.',
    )


class UpdateTableCellStyleRequest(BaseModel):
    tableRange: TableRange | None = Field(
        None, description="The table range representing the subset of the table to which the updates are applied."
    )
    tableStartLocation: Location | None = Field(
        None,
        description="The location where the table starts in the document. When specified, the updates are applied to all the cells in the table.",
    )
    tableCellStyle: TableCellStyle | None = Field(
        None,
        description="The style to set on the table cells. When updating borders, if a cell shares a border with an adjacent cell, the corresponding border property of the adjacent cell is updated as well. Borders that are merged and invisible are not updated. Since updating a border shared by adjacent cells in the same request can cause conflicting border updates, border updates are applied in the following order: - `border_right` - `border_left` - `border_bottom` - `border_top`",
    )
    fields: str | None = Field(
        None,
        description='The fields that should be updated. At least one field must be specified. The root `tableCellStyle` is implied and should not be specified. A single `"*"` can be used as short-hand for listing every field. For example to update the table cell background color, set `fields` to `"backgroundColor"`. To reset a property to its default value, include its field name in the field mask but leave the field itself unset.',
    )


class UpdateDocumentStyleRequest(BaseModel):
    documentStyle: DocumentStyle | None = Field(
        None,
        description="The styles to set on the document. Certain document style changes may cause other changes in order to mirror the behavior of the Docs editor. See the documentation of DocumentStyle for more information.",
    )
    fields: str | None = Field(
        None,
        description='The fields that should be updated. At least one field must be specified. The root `document_style` is implied and should not be specified. A single `"*"` can be used as short-hand for listing every field. For example to update the background, set `fields` to `"background"`.',
    )
    tabId: str | None = Field(
        None,
        description="The tab that contains the style to update. When omitted, the request applies to the first tab. In a document containing a single tab: - If provided, must match the singular tab's ID. - If omitted, the request applies to the singular tab. In a document containing multiple tabs: - If provided, the request applies to the specified tab. - If not provided, the request applies to the first tab in the document.",
    )


class TextRun(BaseModel):
    content: str | None = Field(
        None,
        description="The text of this run. Any non-text elements in the run are replaced with the Unicode character U+E907.",
    )
    suggestedInsertionIds: list[str] | None = Field(
        None,
        description="The suggested insertion IDs. A TextRun may have multiple insertion IDs if it's a nested suggested change. If empty, then this is not a suggested insertion.",
    )
    suggestedDeletionIds: list[str] | None = Field(
        None, description="The suggested deletion IDs. If empty, then there are no suggested deletions of this content."
    )
    textStyle: TextStyle | None = Field(None, description="The text style of this run.")
    suggestedTextStyleChanges: dict[str, SuggestedTextStyle] | None = Field(
        None, description="The suggested text style changes to this run, keyed by suggestion ID."
    )


class NamedStyles(BaseModel):
    styles: list[NamedStyle] | None = Field(
        None, description="The named styles. There's an entry for each of the possible named style types."
    )


class SuggestedNamedStyles(BaseModel):
    namedStyles: NamedStyles | None = Field(
        None,
        description="A NamedStyles that only includes the changes made in this suggestion. This can be used along with the named_styles_suggestion_state to see which fields have changed and their new values.",
    )
    namedStylesSuggestionState: NamedStylesSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields on the base NamedStyles have been changed in this suggestion.",
    )


class ListProperties(BaseModel):
    nestingLevels: list[NestingLevel] | None = Field(
        None,
        description="Describes the properties of the bullets at the associated level. A list has at most 9 levels of nesting with nesting level 0 corresponding to the top-most level and nesting level 8 corresponding to the most nested level. The nesting levels are returned in ascending order with the least nested returned first.",
    )


class SuggestedListProperties(BaseModel):
    listProperties: ListProperties | None = Field(
        None,
        description="A ListProperties that only includes the changes made in this suggestion. This can be used along with the list_properties_suggestion_state to see which fields have changed and their new values.",
    )
    listPropertiesSuggestionState: ListPropertiesSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields on the base ListProperties have been changed in this suggestion.",
    )


class InlineObjectProperties(BaseModel):
    embeddedObject: EmbeddedObject | None = Field(None, description="The embedded object of this inline object.")


class SuggestedInlineObjectProperties(BaseModel):
    inlineObjectProperties: InlineObjectProperties | None = Field(
        None,
        description="An InlineObjectProperties that only includes the changes made in this suggestion. This can be used along with the inline_object_properties_suggestion_state to see which fields have changed and their new values.",
    )
    inlineObjectPropertiesSuggestionState: InlineObjectPropertiesSuggestionState | None = Field(
        None,
        description="A mask that indicates which of the fields on the base InlineObjectProperties have been changed in this suggestion.",
    )


class PositionedObject(BaseModel):
    objectId: str | None = Field(None, description="The ID of this positioned object.")
    positionedObjectProperties: PositionedObjectProperties | None = Field(
        None, description="The properties of this positioned object."
    )
    suggestedPositionedObjectPropertiesChanges: dict[str, SuggestedPositionedObjectProperties] | None = Field(
        None, description="The suggested changes to the positioned object properties, keyed by suggestion ID."
    )
    suggestedInsertionId: str | None = Field(
        None, description="The suggested insertion ID. If empty, then this is not a suggested insertion."
    )
    suggestedDeletionIds: list[str] | None = Field(
        None, description="The suggested deletion IDs. If empty, then there are no suggested deletions of this content."
    )


class Request(BaseModel):
    replaceAllText: ReplaceAllTextRequest | None = Field(
        None, description="Replaces all instances of the specified text."
    )
    insertText: InsertTextRequest | None = Field(None, description="Inserts text at the specified location.")
    updateTextStyle: UpdateTextStyleRequest | None = Field(
        None, description="Updates the text style at the specified range."
    )
    createParagraphBullets: CreateParagraphBulletsRequest | None = Field(
        None, description="Creates bullets for paragraphs."
    )
    deleteParagraphBullets: DeleteParagraphBulletsRequest | None = Field(
        None, description="Deletes bullets from paragraphs."
    )
    createNamedRange: CreateNamedRangeRequest | None = Field(None, description="Creates a named range.")
    deleteNamedRange: DeleteNamedRangeRequest | None = Field(None, description="Deletes a named range.")
    updateParagraphStyle: UpdateParagraphStyleRequest | None = Field(
        None, description="Updates the paragraph style at the specified range."
    )
    deleteContentRange: DeleteContentRangeRequest | None = Field(None, description="Deletes content from the document.")
    insertInlineImage: InsertInlineImageRequest | None = Field(
        None, description="Inserts an inline image at the specified location."
    )
    insertTable: InsertTableRequest | None = Field(None, description="Inserts a table at the specified location.")
    insertTableRow: InsertTableRowRequest | None = Field(None, description="Inserts an empty row into a table.")
    insertTableColumn: InsertTableColumnRequest | None = Field(
        None, description="Inserts an empty column into a table."
    )
    deleteTableRow: DeleteTableRowRequest | None = Field(None, description="Deletes a row from a table.")
    deleteTableColumn: DeleteTableColumnRequest | None = Field(None, description="Deletes a column from a table.")
    insertPageBreak: InsertPageBreakRequest | None = Field(
        None, description="Inserts a page break at the specified location."
    )
    deletePositionedObject: DeletePositionedObjectRequest | None = Field(
        None, description="Deletes a positioned object from the document."
    )
    updateTableColumnProperties: UpdateTableColumnPropertiesRequest | None = Field(
        None, description="Updates the properties of columns in a table."
    )
    updateTableCellStyle: UpdateTableCellStyleRequest | None = Field(
        None, description="Updates the style of table cells."
    )
    updateTableRowStyle: UpdateTableRowStyleRequest | None = Field(
        None, description="Updates the row style in a table."
    )
    replaceImage: ReplaceImageRequest | None = Field(None, description="Replaces an image in the document.")
    updateDocumentStyle: UpdateDocumentStyleRequest | None = Field(
        None, description="Updates the style of the document."
    )
    mergeTableCells: MergeTableCellsRequest | None = Field(None, description="Merges cells in a table.")
    unmergeTableCells: UnmergeTableCellsRequest | None = Field(None, description="Unmerges cells in a table.")
    createHeader: CreateHeaderRequest | None = Field(None, description="Creates a header.")
    createFooter: CreateFooterRequest | None = Field(None, description="Creates a footer.")
    createFootnote: CreateFootnoteRequest | None = Field(None, description="Creates a footnote.")
    replaceNamedRangeContent: ReplaceNamedRangeContentRequest | None = Field(
        None, description="Replaces the content in a named range."
    )
    updateSectionStyle: UpdateSectionStyleRequest | None = Field(
        None, description="Updates the section style of the specified range."
    )
    insertSectionBreak: InsertSectionBreakRequest | None = Field(
        None, description="Inserts a section break at the specified location."
    )
    deleteHeader: DeleteHeaderRequest | None = Field(None, description="Deletes a header from the document.")
    deleteFooter: DeleteFooterRequest | None = Field(None, description="Deletes a footer from the document.")
    pinTableHeaderRows: PinTableHeaderRowsRequest | None = Field(
        None, description="Updates the number of pinned header rows in a table."
    )
    insertPerson: InsertPersonRequest | None = Field(None, description="Inserts a person mention.")


class ParagraphElement(BaseModel):
    startIndex: int | None = Field(
        None, description="The zero-based start index of this paragraph element, in UTF-16 code units."
    )
    endIndex: int | None = Field(
        None, description="The zero-base end index of this paragraph element, exclusive, in UTF-16 code units."
    )
    textRun: TextRun | None = Field(None, description="A text run paragraph element.")
    autoText: AutoText | None = Field(None, description="An auto text paragraph element.")
    pageBreak: PageBreak | None = Field(None, description="A page break paragraph element.")
    columnBreak: ColumnBreak | None = Field(None, description="A column break paragraph element.")
    footnoteReference: FootnoteReference | None = Field(None, description="A footnote reference paragraph element.")
    horizontalRule: HorizontalRule | None = Field(None, description="A horizontal rule paragraph element.")
    equation: Equation | None = Field(None, description="An equation paragraph element.")
    inlineObjectElement: InlineObjectElement | None = Field(None, description="An inline object paragraph element.")
    person: Person | None = Field(None, description="A paragraph element that links to a person or email address.")
    richLink: RichLink | None = Field(
        None,
        description="A paragraph element that links to a Google resource (such as a file in Google Drive, a YouTube video, or a Calendar event.)",
    )


class ListModel(BaseModel):
    listProperties: ListProperties | None = Field(None, description="The properties of the list.")
    suggestedListPropertiesChanges: dict[str, SuggestedListProperties] | None = Field(
        None, description="The suggested changes to the list properties, keyed by suggestion ID."
    )
    suggestedInsertionId: str | None = Field(
        None, description="The suggested insertion ID. If empty, then this is not a suggested insertion."
    )
    suggestedDeletionIds: list[str] | None = Field(
        None, description="The suggested deletion IDs. If empty, then there are no suggested deletions of this list."
    )


class InlineObject(BaseModel):
    objectId: str | None = Field(
        None, description="The ID of this inline object. Can be used to update an object’s properties."
    )
    inlineObjectProperties: InlineObjectProperties | None = Field(
        None, description="The properties of this inline object."
    )
    suggestedInlineObjectPropertiesChanges: dict[str, SuggestedInlineObjectProperties] | None = Field(
        None, description="The suggested changes to the inline object properties, keyed by suggestion ID."
    )
    suggestedInsertionId: str | None = Field(
        None, description="The suggested insertion ID. If empty, then this is not a suggested insertion."
    )
    suggestedDeletionIds: list[str] | None = Field(
        None, description="The suggested deletion IDs. If empty, then there are no suggested deletions of this content."
    )


class BatchUpdateDocumentRequest(BaseModel):
    requests: list[Request] | None = Field(None, description="A list of updates to apply to the document.")
    writeControl: WriteControl | None = Field(
        None, description="Provides control over how write requests are executed."
    )


class Paragraph(BaseModel):
    elements: list[ParagraphElement] | None = Field(
        None, description="The content of the paragraph, broken down into its component parts."
    )
    paragraphStyle: ParagraphStyle | None = Field(None, description="The style of this paragraph.")
    suggestedParagraphStyleChanges: dict[str, SuggestedParagraphStyle] | None = Field(
        None, description="The suggested paragraph style changes to this paragraph, keyed by suggestion ID."
    )
    bullet: Bullet | None = Field(
        None, description="The bullet for this paragraph. If not present, the paragraph does not belong to a list."
    )
    suggestedBulletChanges: dict[str, SuggestedBullet] | None = Field(
        None, description="The suggested changes to this paragraph's bullet."
    )
    positionedObjectIds: list[str] | None = Field(
        None, description="The IDs of the positioned objects tethered to this paragraph."
    )
    suggestedPositionedObjectIds: dict[str, ObjectReferences] | None = Field(
        None,
        description="The IDs of the positioned objects suggested to be attached to this paragraph, keyed by suggestion ID.",
    )


class Document(BaseModel):
    class Config:
        use_enum_values = True

    documentId: str | None = Field(None, description="Output only. The ID of the document.")
    title: str | None = Field(None, description="The title of the document.")
    tabs: list[Tab] | None = Field(
        None,
        description="Tabs that are part of a document. Tabs can contain child tabs, a tab nested within another tab. Child tabs are represented by the Tab.childTabs field.",
    )
    revisionId: str | None = Field(
        None,
        description="Output only. The revision ID of the document. Can be used in update requests to specify which revision of a document to apply updates to and how the request should behave if the document has been edited since that revision. Only populated if the user has edit access to the document. The revision ID is not a sequential number but an opaque string. The format of the revision ID might change over time. A returned revision ID is only guaranteed to be valid for 24 hours after it has been returned and cannot be shared across users. If the revision ID is unchanged between calls, then the document has not changed. Conversely, a changed ID (for the same document and user) usually means the document has been updated. However, a changed ID can also be due to internal factors such as ID format changes.",
    )
    suggestionsViewMode: SuggestionsViewMode | None = Field(
        None,
        description="Output only. The suggestions view mode applied to the document. Note: When editing a document, changes must be based on a document with SUGGESTIONS_INLINE.",
    )
    body: Body | None = Field(
        None,
        description="Output only. The main body of the document. Legacy field: Instead, use Document.tabs.documentTab.body, which exposes the actual document content from all tabs when the includeTabsContent parameter is set to `true`. If `false` or unset, this field contains information about the first tab in the document.",
    )
    headers: dict[str, Header] | None = Field(
        None,
        description="Output only. The headers in the document, keyed by header ID. Legacy field: Instead, use Document.tabs.documentTab.headers, which exposes the actual document content from all tabs when the includeTabsContent parameter is set to `true`. If `false` or unset, this field contains information about the first tab in the document.",
    )
    footers: dict[str, Footer] | None = Field(
        None,
        description="Output only. The footers in the document, keyed by footer ID. Legacy field: Instead, use Document.tabs.documentTab.footers, which exposes the actual document content from all tabs when the includeTabsContent parameter is set to `true`. If `false` or unset, this field contains information about the first tab in the document.",
    )
    footnotes: dict[str, Footnote] | None = Field(
        None,
        description="Output only. The footnotes in the document, keyed by footnote ID. Legacy field: Instead, use Document.tabs.documentTab.footnotes, which exposes the actual document content from all tabs when the includeTabsContent parameter is set to `true`. If `false` or unset, this field contains information about the first tab in the document.",
    )
    documentStyle: DocumentStyle | None = Field(
        None,
        description="Output only. The style of the document. Legacy field: Instead, use Document.tabs.documentTab.documentStyle, which exposes the actual document content from all tabs when the includeTabsContent parameter is set to `true`. If `false` or unset, this field contains information about the first tab in the document.",
    )
    suggestedDocumentStyleChanges: dict[str, SuggestedDocumentStyle] | None = Field(
        None,
        description="Output only. The suggested changes to the style of the document, keyed by suggestion ID. Legacy field: Instead, use Document.tabs.documentTab.suggestedDocumentStyleChanges, which exposes the actual document content from all tabs when the includeTabsContent parameter is set to `true`. If `false` or unset, this field contains information about the first tab in the document.",
    )
    namedStyles: NamedStyles | None = Field(
        None,
        description="Output only. The named styles of the document. Legacy field: Instead, use Document.tabs.documentTab.namedStyles, which exposes the actual document content from all tabs when the includeTabsContent parameter is set to `true`. If `false` or unset, this field contains information about the first tab in the document.",
    )
    suggestedNamedStylesChanges: dict[str, SuggestedNamedStyles] | None = Field(
        None,
        description="Output only. The suggested changes to the named styles of the document, keyed by suggestion ID. Legacy field: Instead, use Document.tabs.documentTab.suggestedNamedStylesChanges, which exposes the actual document content from all tabs when the includeTabsContent parameter is set to `true`. If `false` or unset, this field contains information about the first tab in the document.",
    )
    lists: dict[str, ListModel] | None = Field(
        None,
        description="Output only. The lists in the document, keyed by list ID. Legacy field: Instead, use Document.tabs.documentTab.lists, which exposes the actual document content from all tabs when the includeTabsContent parameter is set to `true`. If `false` or unset, this field contains information about the first tab in the document.",
    )
    namedRanges: dict[str, NamedRanges] | None = Field(
        None,
        description="Output only. The named ranges in the document, keyed by name. Legacy field: Instead, use Document.tabs.documentTab.namedRanges, which exposes the actual document content from all tabs when the includeTabsContent parameter is set to `true`. If `false` or unset, this field contains information about the first tab in the document.",
    )
    inlineObjects: dict[str, InlineObject] | None = Field(
        None,
        description="Output only. The inline objects in the document, keyed by object ID. Legacy field: Instead, use Document.tabs.documentTab.inlineObjects, which exposes the actual document content from all tabs when the includeTabsContent parameter is set to `true`. If `false` or unset, this field contains information about the first tab in the document.",
    )
    positionedObjects: dict[str, PositionedObject] | None = Field(
        None,
        description="Output only. The positioned objects in the document, keyed by object ID. Legacy field: Instead, use Document.tabs.documentTab.positionedObjects, which exposes the actual document content from all tabs when the includeTabsContent parameter is set to `true`. If `false` or unset, this field contains information about the first tab in the document.",
    )


class Tab(BaseModel):
    tabProperties: TabProperties | None = Field(None, description="The properties of the tab, like ID and title.")
    childTabs: list[Tab] | None = Field(None, description="The child tabs nested within this tab.")
    documentTab: DocumentTab | None = Field(None, description="A tab with document contents, like text and images.")


class DocumentTab(BaseModel):
    body: Body | None = Field(None, description="The main body of the document tab.")
    headers: dict[str, Header] | None = Field(None, description="The headers in the document tab, keyed by header ID.")
    footers: dict[str, Footer] | None = Field(None, description="The footers in the document tab, keyed by footer ID.")
    footnotes: dict[str, Footnote] | None = Field(
        None, description="The footnotes in the document tab, keyed by footnote ID."
    )
    documentStyle: DocumentStyle | None = Field(None, description="The style of the document tab.")
    suggestedDocumentStyleChanges: dict[str, SuggestedDocumentStyle] | None = Field(
        None, description="The suggested changes to the style of the document tab, keyed by suggestion ID."
    )
    namedStyles: NamedStyles | None = Field(None, description="The named styles of the document tab.")
    suggestedNamedStylesChanges: dict[str, SuggestedNamedStyles] | None = Field(
        None, description="The suggested changes to the named styles of the document tab, keyed by suggestion ID."
    )
    lists: dict[str, ListModel] | None = Field(None, description="The lists in the document tab, keyed by list ID.")
    namedRanges: dict[str, NamedRanges] | None = Field(
        None, description="The named ranges in the document tab, keyed by name."
    )
    inlineObjects: dict[str, InlineObject] | None = Field(
        None, description="The inline objects in the document tab, keyed by object ID."
    )
    positionedObjects: dict[str, PositionedObject] | None = Field(
        None, description="The positioned objects in the document tab, keyed by object ID."
    )


class Body(BaseModel):
    content: list[StructuralElement] | None = Field(
        None, description="The contents of the body. The indexes for the body's content begin at zero."
    )


class StructuralElement(BaseModel):
    startIndex: int | None = Field(
        None, description="The zero-based start index of this structural element, in UTF-16 code units."
    )
    endIndex: int | None = Field(
        None, description="The zero-based end index of this structural element, exclusive, in UTF-16 code units."
    )
    paragraph: Paragraph | None = Field(None, description="A paragraph type of structural element.")
    sectionBreak: SectionBreak | None = Field(None, description="A section break type of structural element.")
    table: Table | None = Field(None, description="A table type of structural element.")
    tableOfContents: TableOfContents | None = Field(None, description="A table of contents type of structural element.")


class Table(BaseModel):
    rows: int | None = Field(None, description="Number of rows in the table.")
    columns: int | None = Field(
        None,
        description="Number of columns in the table. It's possible for a table to be non-rectangular, so some rows may have a different number of cells.",
    )
    tableRows: list[TableRow] | None = Field(None, description="The contents and style of each row.")
    suggestedInsertionIds: list[str] | None = Field(
        None,
        description="The suggested insertion IDs. A Table may have multiple insertion IDs if it's a nested suggested change. If empty, then this is not a suggested insertion.",
    )
    suggestedDeletionIds: list[str] | None = Field(
        None, description="The suggested deletion IDs. If empty, then there are no suggested deletions of this content."
    )
    tableStyle: TableStyle | None = Field(None, description="The style of the table.")


class TableRow(BaseModel):
    startIndex: int | None = Field(None, description="The zero-based start index of this row, in UTF-16 code units.")
    endIndex: int | None = Field(
        None, description="The zero-based end index of this row, exclusive, in UTF-16 code units."
    )
    tableCells: list[TableCell] | None = Field(
        None,
        description="The contents and style of each cell in this row. It's possible for a table to be non-rectangular, so some rows may have a different number of cells than other rows in the same table.",
    )
    suggestedInsertionIds: list[str] | None = Field(
        None,
        description="The suggested insertion IDs. A TableRow may have multiple insertion IDs if it's a nested suggested change. If empty, then this is not a suggested insertion.",
    )
    suggestedDeletionIds: list[str] | None = Field(
        None, description="The suggested deletion IDs. If empty, then there are no suggested deletions of this content."
    )
    tableRowStyle: TableRowStyle | None = Field(None, description="The style of the table row.")
    suggestedTableRowStyleChanges: dict[str, SuggestedTableRowStyle] | None = Field(
        None, description="The suggested style changes to this row, keyed by suggestion ID."
    )


class TableCell(BaseModel):
    startIndex: int | None = Field(None, description="The zero-based start index of this cell, in UTF-16 code units.")
    endIndex: int | None = Field(
        None, description="The zero-based end index of this cell, exclusive, in UTF-16 code units."
    )
    content: list[StructuralElement] | None = Field(None, description="The content of the cell.")
    tableCellStyle: TableCellStyle | None = Field(None, description="The style of the cell.")
    suggestedInsertionIds: list[str] | None = Field(
        None,
        description="The suggested insertion IDs. A TableCell may have multiple insertion IDs if it's a nested suggested change. If empty, then this is not a suggested insertion.",
    )
    suggestedDeletionIds: list[str] | None = Field(
        None, description="The suggested deletion IDs. If empty, then there are no suggested deletions of this content."
    )
    suggestedTableCellStyleChanges: dict[str, SuggestedTableCellStyle] | None = Field(
        None, description="The suggested changes to the table cell style, keyed by suggestion ID."
    )


class TableOfContents(BaseModel):
    content: list[StructuralElement] | None = Field(None, description="The content of the table of contents.")
    suggestedInsertionIds: list[str] | None = Field(
        None,
        description="The suggested insertion IDs. A TableOfContents may have multiple insertion IDs if it is a nested suggested change. If empty, then this is not a suggested insertion.",
    )
    suggestedDeletionIds: list[str] | None = Field(
        None, description="The suggested deletion IDs. If empty, then there are no suggested deletions of this content."
    )


class Header(BaseModel):
    headerId: str | None = Field(None, description="The ID of the header.")
    content: list[StructuralElement] | None = Field(
        None, description="The contents of the header. The indexes for a header's content begin at zero."
    )


class Footer(BaseModel):
    footerId: str | None = Field(None, description="The ID of the footer.")
    content: list[StructuralElement] | None = Field(
        None, description="The contents of the footer. The indexes for a footer's content begin at zero."
    )


class Footnote(BaseModel):
    footnoteId: str | None = Field(None, description="The ID of the footnote.")
    content: list[StructuralElement] | None = Field(
        None, description="The contents of the footnote. The indexes for a footnote's content begin at zero."
    )


Document.model_rebuild()
Tab.model_rebuild()
DocumentTab.model_rebuild()
Body.model_rebuild()
StructuralElement.model_rebuild()
Table.model_rebuild()
TableRow.model_rebuild()
