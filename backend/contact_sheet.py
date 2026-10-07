"""X/Y plot contact sheets (F18): the images of a parameter grid laid out in a
labelled table -- a column per X value, a row per Y value -- like Forge's X/Y
plot. The frontend queues one prompt per cell (frontend/app.js, "X/Y Plot") and
posts the results to /api/contact_sheet, which calls compose() and saves the
sheet to outputs/.
"""
from __future__ import annotations

from collections.abc import Sequence

from PIL import Image, ImageDraw, ImageFont

MAX_CELLS = 100
BACKGROUND = (24, 24, 28)
TEXT = (230, 230, 235)
MISSING = (60, 30, 30)


def _font(size: int) -> ImageFont.ImageFont:
    return ImageFont.load_default(size=size)


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """`image` scaled to fit `size`, centred on the background."""
    cell = Image.new("RGB", size, BACKGROUND)
    copy = image.convert("RGB")
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    cell.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return cell


def compose(
    cells: Sequence[Image.Image | None],
    columns: int,
    x_values: Sequence[str],
    y_values: Sequence[str],
    x_label: str = "",
    y_label: str = "",
    max_cell: int = 512,
) -> Image.Image:
    """The sheet for `cells` in row-major order (None: a failed cell, drawn
    dark red). Every cell is as large as the largest image (each image fitted
    and centred in it), scaled so its long side is at most `max_cell` px."""
    if not cells or columns < 1 or len(cells) > MAX_CELLS:
        raise ValueError(f"a contact sheet needs 1-{MAX_CELLS} cells and at least one column")
    rows = -(-len(cells) // columns)
    if len(x_values) != columns or len(y_values) != rows:
        raise ValueError(f"expected {columns} X labels and {rows} Y labels, got {len(x_values)} and {len(y_values)}")
    present = [cell for cell in cells if cell is not None]
    width = max((cell.width for cell in present), default=max_cell)
    height = max((cell.height for cell in present), default=max_cell)
    scale = min(1.0, max_cell / max(width, height))
    cell_w, cell_h = max(1, round(width * scale)), max(1, round(height * scale))
    font_size = max(14, cell_h // 18)
    font, title_font = _font(font_size), _font(int(font_size * 1.15))
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    def text_width(text: str, which=font) -> int:
        return int(measure.textlength(text, font=which))

    gap = max(4, font_size // 3)
    left = gap * 2 + max([text_width(value) for value in y_values] + [text_width(y_label, title_font)])
    top = gap * 2 + font_size + (font_size + gap if x_label else 0)
    sheet = Image.new("RGB", (left + columns * (cell_w + gap) + gap, top + rows * (cell_h + gap) + gap), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    if x_label:  # centred over the columns, clear of the Y title in the corner
        grid_width = columns * (cell_w + gap) - gap
        draw.text((left + max(0, (grid_width - text_width(x_label, title_font)) // 2), gap), x_label, fill=TEXT, font=title_font)
    if y_label:
        draw.text((gap, gap), y_label, fill=TEXT, font=title_font)
    for column, value in enumerate(x_values):
        x = left + column * (cell_w + gap)
        draw.text((x + (cell_w - text_width(value)) // 2, top - font_size - gap), value, fill=TEXT, font=font)
    for row, value in enumerate(y_values):
        y = top + row * (cell_h + gap)
        draw.text((gap, y + (cell_h - font_size) // 2), value, fill=TEXT, font=font)
    for index, cell in enumerate(cells):
        row, column = divmod(index, columns)
        x, y = left + column * (cell_w + gap), top + row * (cell_h + gap)
        sheet.paste(_fit(cell, (cell_w, cell_h)) if cell is not None else Image.new("RGB", (cell_w, cell_h), MISSING), (x, y))
    return sheet
