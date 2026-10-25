"""BIO tag helpers with no third-party dependencies (shared by evaluation and the API)."""


def bio_entities(tags: list[str]) -> list[tuple[str, int, int]]:
    """Return (field, start, end) spans, end inclusive.

    Same chunking rule as seqeval's default (conlleval) mode for BIO tags:
    a span starts at B-X, or at I-X when the previous tag is not B-X/I-X;
    it continues while the tag is I-X.
    """
    spans = []
    field, start = None, None
    for i, tag in enumerate(tags + ["O"]):
        prefix, _, name = tag.partition("-")
        continues = prefix == "I" and name == field
        if field is not None and not continues:
            spans.append((field, start, i - 1))
            field = None
        if prefix in ("B", "I") and not continues:
            field, start = name, i
    return spans
