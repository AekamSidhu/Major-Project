"""Turn any answer into what a speech transcript of it would look like.

A candidate who reads an AI answer aloud loses all its formatting: no
headings, bullets, numbered lists or line breaks, and the transcriber decides
the punctuation. HC3's human answers carry their own artefacts too (Reddit
text is pre-tokenised as "do n't" and " , ", links are "URL_0"). A detector
trained on raw HC3 learns those artefacts instead of how the answers are
written. So every answer, human or AI, in training or in the app, goes
through `to_transcript` first: lower case, words only.
"""

from __future__ import annotations

import re

_TOKENISED = [
    (re.compile(r"\s+n't\b"), "n't"),
    (re.compile(r"\s+'(s|re|ve|ll|d|m)\b"), r"'\1"),
    (re.compile(r"\b(ca|wo|do|does|did|is|are|was|were|has|have|had|could|would|should|must)\s+n't\b"), r"\1n't"),
]
_URL = re.compile(r"(URL_\d+|https?://\S+|www\.\S+)")
_MARKDOWN = re.compile(r"(^|\n)\s*(#+|[-*•]|\d+[.)])\s+")
_CHARS = re.compile(r"[^a-z0-9' ]+")
_SPACES = re.compile(r"\s+")


def to_transcript(text: str) -> str:
    """Lower-case words separated by single spaces, as a transcript would give."""
    text = _URL.sub(" ", text)
    for pattern, replacement in _TOKENISED:
        text = pattern.sub(replacement, text)
    text = _MARKDOWN.sub(" ", text)
    text = text.replace("**", " ").replace("__", " ").replace("`", " ")
    text = text.lower().replace("’", "'")
    text = _CHARS.sub(" ", text)
    text = text.replace("'", "")  # "don't" and "dont" are the same word to a transcriber
    return _SPACES.sub(" ", text).strip()


def first_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])
