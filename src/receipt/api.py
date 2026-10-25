"""HTTP API: POST /parse with a receipt image and its OCR words/boxes, get extracted fields back.

    MODEL_DIR=model uvicorn receipt.api:app --port 8000

    curl -s -F image=@examples/receipt.jpg -F "ocr=<examples/ocr.json" localhost:8000/parse

`ocr` is a JSON object {"words": [...], "boxes": [[x0, y0, x1, y1], ...]} in pixel coordinates
of the uploaded image. The model directory can hold either fine-tuned model (text or layout);
the image is only used by the layout model.
"""

import io
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ValidationError, field_validator
from transformers import AutoImageProcessor, AutoModelForTokenClassification, AutoTokenizer

from .bio import bio_entities
from .data import Receipt
from .evaluate import is_layout_model, predict


class OCRInput(BaseModel):
    words: list[str]
    boxes: list[list[float]]

    @field_validator("boxes")
    @classmethod
    def four_coordinates(cls, boxes):
        if any(len(b) != 4 for b in boxes):
            raise ValueError("each box must be [x0, y0, x1, y1]")
        return boxes


class ExtractedField(BaseModel):
    field: str
    text: str
    word_indices: list[int]


class ParseResponse(BaseModel):
    model_type: str
    fields: list[ExtractedField]


@dataclass
class Bundle:
    model: object
    tokenizer: object
    image_processor: object | None


def load_bundle(model_dir: Path) -> Bundle:
    if not (model_dir / "config.json").exists():
        raise RuntimeError(f"no model found in {model_dir} (see README: 'Getting model/')")
    model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    image_processor = AutoImageProcessor.from_pretrained(model_dir) if is_layout_model(model) else None
    return Bundle(model, tokenizer, image_processor)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = load_bundle(Path(os.environ.get("MODEL_DIR", "model")))
    yield


app = FastAPI(title="Receipt field extraction (CORD-v2)", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_type": app.state.bundle.model.config.model_type}


@app.post("/parse", response_model=ParseResponse)
async def parse(image: UploadFile = File(...), ocr: str = Form(...)):
    try:
        data = OCRInput.model_validate_json(ocr)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors(include_url=False))
    if len(data.words) != len(data.boxes):
        raise HTTPException(status_code=422, detail="words and boxes must have the same length")
    try:
        img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="image is not a readable image file")

    bundle = app.state.bundle
    if not data.words:
        return ParseResponse(model_type=bundle.model.config.model_type, fields=[])

    width, height = img.size
    receipt = Receipt(
        image_id=0,
        words=data.words,
        boxes=[[round(v) for v in b] for b in data.boxes],
        tags=["O"] * len(data.words),
        width=width,
        height=height,
    )
    if bundle.image_processor is not None:
        receipt.pixel_values = bundle.image_processor(images=img, return_tensors="np")["pixel_values"][0]

    tags = predict(bundle.model, bundle.tokenizer, [receipt])[0]
    fields = [
        ExtractedField(
            field=name,
            text=" ".join(data.words[start:end + 1]),
            word_indices=list(range(start, end + 1)),
        )
        for name, start, end in bio_entities(tags)
    ]
    return ParseResponse(model_type=bundle.model.config.model_type, fields=fields)
