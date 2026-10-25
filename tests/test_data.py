import json

from receipt.data import line_to_bio, normalize_box, parse_ground_truth, quad_to_box


def word(text, x0, y0, x1, y1, is_key=0):
    quad = {"x1": x0, "y1": y0, "x2": x1, "y2": y0, "x3": x1, "y3": y1, "x4": x0, "y4": y1}
    return {"text": text, "quad": quad, "is_key": is_key}


def test_line_to_bio():
    name = [word("Nasi", 0, 0, 1, 1), word("Campur", 0, 0, 1, 1), word("Bali", 0, 0, 1, 1)]
    assert line_to_bio(name, "menu.nm") == ["B-menu.nm", "I-menu.nm", "I-menu.nm"]

    total = [word("Grand", 0, 0, 1, 1, is_key=1), word("Total", 0, 0, 1, 1, is_key=1), word("75,000", 0, 0, 1, 1)]
    assert line_to_bio(total, "total.total_price") == ["O", "O", "B-total.total_price"]
    assert line_to_bio(total, "total.total_price", include_keys=True) == [
        "B-total.total_price", "I-total.total_price", "I-total.total_price",
    ]

    # A key word inside the value span stays part of the entity; a trailing key word does not.
    mixed = [word("2", 0, 0, 1, 1), word("x", 0, 0, 1, 1, is_key=1), word("@5", 0, 0, 1, 1), word("pcs", 0, 0, 1, 1, is_key=1)]
    assert line_to_bio(mixed, "menu.cnt") == ["B-menu.cnt", "I-menu.cnt", "I-menu.cnt", "O"]

    assert line_to_bio([word("TOTAL", 0, 0, 1, 1, is_key=1)], "total.total_price") == ["O"]


def test_parse_ground_truth():
    gt = {
        "meta": {"image_id": 7, "image_size": {"width": 200, "height": 100}},
        "valid_line": [
            {"category": "menu.nm", "words": [word("Ice", 10, 10, 30, 20), word("Tea", 32, 10, 50, 20)]},
            {"category": "total.total_price", "words": [word("TOTAL", 10, 80, 40, 90, is_key=1), word("8,000", 150, 80, 190, 90)]},
        ],
    }
    r = parse_ground_truth(json.dumps(gt))
    assert r.image_id == 7
    assert r.words == ["Ice", "Tea", "TOTAL", "8,000"]
    assert r.tags == ["B-menu.nm", "I-menu.nm", "O", "B-total.total_price"]
    assert r.boxes[3] == [150, 80, 190, 90]
    assert r.norm_boxes[3] == [750, 800, 950, 900]


def test_box_conversion_and_normalization():
    # A slanted quadrilateral: its bounding rectangle is the min / max over the 4 corners.
    quad = {"x1": 12, "y1": 5, "x2": 40, "y2": 3, "x3": 41, "y3": 15, "x4": 11, "y4": 17}
    assert quad_to_box(quad) == [11, 3, 41, 17]

    assert normalize_box([50, 25, 150, 75], width=200, height=100) == [250, 250, 750, 750]
    # Boxes that stick out of the image are clipped to the 0-1000 grid.
    assert normalize_box([-5, -1, 210, 120], width=200, height=100) == [0, 0, 1000, 1000]
