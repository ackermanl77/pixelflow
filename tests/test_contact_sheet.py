"""F18: X/Y plot contact sheets (backend/contact_sheet.py and /api/contact_sheet)."""
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import contact_sheet, main
from backend.engine.graph import execute_prompt

BASE = "http://127.0.0.1:8288"


def _solid(color, size=(64, 48)):
    return Image.new("RGB", size, color)


def test_compose_lays_cells_out_row_major_with_labels():
    cells = [_solid((255, 0, 0)), _solid((0, 255, 0)), _solid((0, 0, 255)), None]
    sheet = contact_sheet.compose(cells, 2, ["cfg 1", "cfg 7"], ["4 steps", "20 steps"], "cfg", "steps")
    assert sheet.width > 2 * 64 and sheet.height > 2 * 48
    pixels = sheet.load()
    # Find each cell's centre colour: red, green / blue, missing (dark red), in row-major order.
    colours = {pixels[x, y] for x in range(sheet.width) for y in range(sheet.height)}
    for expected in ((255, 0, 0), (0, 255, 0), (0, 0, 255), contact_sheet.MISSING):
        assert expected in colours


def test_large_cells_are_scaled_down():
    sheet = contact_sheet.compose([_solid((9, 9, 9), (2048, 1024))], 1, ["a"], ["b"], max_cell=256)
    assert sheet.width < 400


@pytest.mark.parametrize("cells, columns, xs, ys", [
    ([], 1, [], []),
    ([_solid((0, 0, 0))] * 3, 2, ["a"], ["b", "c"]),  # one X label for two columns
    ([_solid((0, 0, 0))] * 3, 2, ["a", "b"], ["c"]),  # two rows need two Y labels
])
def test_bad_grids_are_rejected(cells, columns, xs, ys):
    with pytest.raises(ValueError):
        contact_sheet.compose(cells, columns, xs, ys)


def test_the_endpoint_saves_a_sheet_of_real_outputs(output_dir):
    client = TestClient(main.app, base_url=BASE)
    names = []
    for color in ("#ff0000", "#00ff00"):
        outputs = execute_prompt({
            "1": {"class_type": "EmptyImage", "inputs": {"width": 32, "height": 32, "color": color}},
            "2": {"class_type": "SaveImage", "inputs": {"image": ["1", 0], "filename_prefix": "cell"}},
        })
        names.append(outputs["2"]["images"][0])
    body = {"cells": [{"filename": n} for n in names] + [None], "columns": 3, "x_values": ["a", "b", "c"],
            "y_values": ["row"], "x_label": "x", "y_label": "y"}
    result = client.post("/api/contact_sheet", json=body)
    assert result.status_code == 200, result.text
    saved = result.json()
    assert saved["type"] == "output" and saved["filename"].startswith("xy_plot/")
    assert (output_dir / saved["filename"]).is_file()
    assert client.get(f"/outputs/{saved['filename']}").status_code == 200
    # Bad input: a traversal path, and labels that don't match the grid.
    assert client.post("/api/contact_sheet", json={**body, "cells": [{"filename": "../x.png"}], "columns": 1, "x_values": ["a"]}).status_code == 404
    assert client.post("/api/contact_sheet", json={**body, "x_values": ["a"]}).status_code == 400


def test_cells_take_the_largest_image_size_so_nothing_is_shrunk():
    sheet = contact_sheet.compose([_solid((200, 0, 0), (100, 50)), _solid((0, 0, 200), (200, 50))], 2, ["a", "b"], ["r"])
    reds = sum(1 for p in sheet.getdata() if p == (200, 0, 0))
    blues = sum(1 for p in sheet.getdata() if p == (0, 0, 200))
    assert reds == 100 * 50 and blues == 200 * 50  # both at their own size
