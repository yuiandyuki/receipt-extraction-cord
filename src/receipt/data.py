"""Load CORD-v2 and turn each receipt into word-level BIO sequences."""

import json
from dataclasses import dataclass

import numpy as np

DATASET = "naver-clova-ix/cord-v2"

# The 29 line categories that occur in CORD-v2 (train, validation and test all use this set).
CATEGORIES = [
    "menu.cnt", "menu.discountprice", "menu.etc", "menu.itemsubtotal", "menu.nm", "menu.num",
    "menu.price", "menu.sub.cnt", "menu.sub.nm", "menu.sub.price", "menu.sub.unitprice",
    "menu.unitprice", "menu.vatyn",
    "sub_total.discount_price", "sub_total.etc", "sub_total.othersvc_price",
    "sub_total.service_price", "sub_total.subtotal_price", "sub_total.tax_price",
    "total.cashprice", "total.changeprice", "total.creditcardprice", "total.emoneyprice",
    "total.menuqty_cnt", "total.menutype_cnt", "total.total_etc", "total.total_price",
    "void_menu.nm", "void_menu.price",
]
LABELS = ["O"] + [f"{p}-{c}" for c in CATEGORIES for p in ("B", "I")]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


@dataclass
class Receipt:
    image_id: int
    words: list[str]
    boxes: list[list[int]]  # pixel coordinates [x0, y0, x1, y1]
    tags: list[str]
    width: int
    height: int
    pixel_values: np.ndarray | None = None  # filled in only for the layout model

    @property
    def norm_boxes(self) -> list[list[int]]:
        return [normalize_box(b, self.width, self.height) for b in self.boxes]


def quad_to_box(quad: dict) -> list[int]:
    """Axis-aligned bounding rectangle of a CORD quadrilateral (x1..x4, y1..y4)."""
    xs = [quad[f"x{i}"] for i in range(1, 5)]
    ys = [quad[f"y{i}"] for i in range(1, 5)]
    return [min(xs), min(ys), max(xs), max(ys)]


def normalize_box(box: list[int], width: int, height: int) -> list[int]:
    """Scale pixel coordinates to the 0-1000 grid LayoutLMv3 expects, clipping boxes that
    stick out of the image (a few CORD words do)."""
    def scale(v, size):
        return min(max(int(1000 * v / size), 0), 1000)

    x0, y0, x1, y1 = box
    return [scale(x0, width), scale(y0, height), scale(x1, width), scale(y1, height)]


def line_to_bio(words: list[dict], category: str, include_keys: bool = False) -> list[str]:
    """BIO tags for one annotated line: B- on the first word, I- on the rest.

    CORD puts key words such as "TOTAL" in the same line as the amount (is_key=1).
    By default they are tagged O, so the entity is the value itself ("75,000"), which is
    what bookkeeping needs. Only leading/trailing key words are dropped; a key word inside
    the value span stays I- so that each line is still one contiguous entity.
    """
    keep = [include_keys or not w["is_key"] for w in words]
    if not any(keep):
        return ["O"] * len(words)
    first = keep.index(True)
    last = len(keep) - 1 - keep[::-1].index(True)
    tags = []
    for i in range(len(words)):
        if i < first or i > last:
            tags.append("O")
        else:
            tags.append(("B-" if i == first else "I-") + category)
    return tags


def parse_ground_truth(ground_truth: str, include_keys: bool = False) -> Receipt:
    gt = json.loads(ground_truth)
    size = gt["meta"]["image_size"]
    words, boxes, tags = [], [], []
    for line in gt["valid_line"]:
        tags.extend(line_to_bio(line["words"], line["category"], include_keys))
        for w in line["words"]:
            words.append(w["text"])
            boxes.append(quad_to_box(w["quad"]))
    return Receipt(
        image_id=gt["meta"]["image_id"],
        words=words,
        boxes=boxes,
        tags=tags,
        width=size["width"],
        height=size["height"],
    )


def load_split(split: str, image_processor=None, include_keys: bool = False) -> list[Receipt]:
    """Load one CORD-v2 split. With an image processor, each receipt also gets its
    preprocessed `pixel_values` (computed once here, not every epoch)."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=split)
    receipts = []
    for ex in ds:
        r = parse_ground_truth(ex["ground_truth"], include_keys)
        if image_processor is not None:
            image = ex["image"].convert("RGB")
            r.pixel_values = image_processor(images=image, return_tensors="np")["pixel_values"][0]
        receipts.append(r)
    return receipts
