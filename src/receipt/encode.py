"""Turn word-level receipts into model inputs. Training, evaluation and the API all go through
`encode`, so every path tokenizes and aligns labels the same way."""

import numpy as np
import torch

from .data import LABEL2ID

MAX_LENGTH = 512


def align_labels(word_ids: list[int | None], tags: list[str]) -> list[int]:
    """Label only the first sub-token of each word; other sub-tokens and special tokens get -100
    so the loss ignores them."""
    labels, prev = [], None
    for wid in word_ids:
        if wid is None or wid == prev:
            labels.append(-100)
        else:
            labels.append(LABEL2ID[tags[wid]])
        prev = wid
    return labels


def encode(tokenizer, words, boxes=None, tags=None, pixel_values=None, max_length=MAX_LENGTH):
    """Batch-encode receipts.

    words: list of word lists. boxes: list of 0-1000 box lists, only for the layout model.
    Returns the tensors for the model and, per receipt, the token -> word index mapping.
    """
    kwargs = dict(padding="longest", truncation=True, max_length=max_length, return_tensors="pt")
    if boxes is not None:
        # The LayoutLMv3 tokenizer takes words and their boxes together and repeats each
        # word's box on all of its sub-tokens.
        enc = tokenizer(words, boxes=boxes, **kwargs)
    else:
        enc = tokenizer(words, is_split_into_words=True, **kwargs)
    word_ids = [enc.word_ids(i) for i in range(len(words))]
    if tags is not None:
        enc["labels"] = torch.tensor([align_labels(w, t) for w, t in zip(word_ids, tags)])
    if pixel_values is not None:
        enc["pixel_values"] = torch.from_numpy(np.stack(pixel_values))
    return enc, word_ids


class ReceiptDataset(torch.utils.data.Dataset):
    def __init__(self, receipts):
        self.receipts = receipts

    def __len__(self):
        return len(self.receipts)

    def __getitem__(self, i):
        return self.receipts[i]


class Collator:
    """Encodes a batch of `Receipt`s at batch time, padding only to the longest receipt."""

    def __init__(self, tokenizer, use_layout: bool, max_length: int = MAX_LENGTH):
        self.tokenizer = tokenizer
        self.use_layout = use_layout
        self.max_length = max_length

    def __call__(self, receipts):
        enc, _ = encode(
            self.tokenizer,
            [r.words for r in receipts],
            boxes=[r.norm_boxes for r in receipts] if self.use_layout else None,
            tags=[r.tags for r in receipts],
            pixel_values=[r.pixel_values for r in receipts] if self.use_layout else None,
            max_length=self.max_length,
        )
        return dict(enc)
