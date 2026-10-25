"""API test with a tiny randomly initialised model built here (no pretrained weights downloaded).
Its predictions are random; the test checks the request/response contract."""

import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from transformers import (
    LayoutLMv3Config,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ImageProcessor,
    LayoutLMv3Tokenizer,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
)

from receipt.data import CATEGORIES, LABEL2ID, LABELS

SPECIAL = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


def byte_level_vocab():
    """Byte-level BPE vocab with no merges: every byte is its own token (GPT-2 byte mapping)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {tok: i for i, tok in enumerate(SPECIAL + [chr(c) for c in cs])}


def tiny_model_dir(tmp_path, kind):
    vocab = byte_level_vocab()
    common = dict(
        vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=600, pad_token_id=1, bos_token_id=0, eos_token_id=2,
        num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID,
    )
    if kind == "layoutlmv3":
        # hidden_size = 4 * coordinate_size + 2 * shape_size
        model = LayoutLMv3ForTokenClassification(LayoutLMv3Config(coordinate_size=4, shape_size=8, **common))
        tokenizer = LayoutLMv3Tokenizer(vocab=vocab, merges=[])
        LayoutLMv3ImageProcessor(apply_ocr=False).save_pretrained(tmp_path)
    else:
        model = XLMRobertaForTokenClassification(XLMRobertaConfig(**common))
        tokenizer = RobertaTokenizer(vocab=vocab, merges=[], add_prefix_space=True)
    model.save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    return tmp_path


@pytest.mark.parametrize("kind", ["layoutlmv3", "xlm-roberta"])
def test_parse(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("MODEL_DIR", str(tiny_model_dir(tmp_path, kind)))
    from receipt.api import app

    buf = io.BytesIO()
    Image.new("RGB", (300, 400), "white").save(buf, format="PNG")
    ocr = {
        "words": ["Nasi", "Goreng", "25,000", "TOTAL", "25,000"],
        "boxes": [[10, 10, 60, 30], [65, 10, 140, 30], [200, 10, 280, 30], [10, 350, 80, 370], [200, 350, 280, 370]],
    }

    with TestClient(app) as client:
        r = client.post("/parse", files={"image": ("r.png", buf.getvalue(), "image/png")}, data={"ocr": json.dumps(ocr)})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model_type"] == kind
        for f in body["fields"]:
            assert f["field"] in CATEGORIES
            assert f["word_indices"] == list(range(f["word_indices"][0], f["word_indices"][-1] + 1))
            assert f["text"] == " ".join(ocr["words"][i] for i in f["word_indices"])

        bad = {"words": ["a", "b"], "boxes": [[0, 0, 1, 1]]}
        r = client.post("/parse", files={"image": ("r.png", buf.getvalue(), "image/png")}, data={"ocr": json.dumps(bad)})
        assert r.status_code == 422
