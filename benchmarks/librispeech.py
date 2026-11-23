"""LibriSpeech test sets for the transcription benchmark: download, check, extract, pick a seeded subset.

    python benchmarks/librispeech.py --root M:/openmeet-data/librispeech     # download + extract test-clean, test-other

LibriSpeech (Panayotov et al. 2015, CC BY 4.0) from openslr.org, no login.
test-clean is 346 MB and test-other 328 MB; both are checked against
openslr's md5 sums and extracted under ROOT/LibriSpeech/<split>/.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

URL = "https://www.openslr.org/resources/12/{split}.tar.gz"
MD5 = {"test-clean": "32fa31d27d2e1cad72775fee3f4849a9", "test-other": "fb5a50374b501bb3bac4815ee91d3135"}
SPLITS = tuple(MD5)


@dataclass(frozen=True)
class Utterance:
    id: str  # speaker-chapter-index, e.g. 1089-134686-0000
    split: str
    speaker: str
    path: Path  # the .flac file
    text: str  # reference transcript (upper case, no punctuation)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(root: Path, split: str) -> Path:
    """Downloads and extracts one split (once); returns ROOT/LibriSpeech/<split>."""
    folder = root / "LibriSpeech" / split
    if folder.is_dir() and any(folder.glob("*/*/*.trans.txt")):
        return folder
    archive = root / f"{split}.tar.gz"
    root.mkdir(parents=True, exist_ok=True)
    if not archive.exists() or md5(archive) != MD5[split]:
        print(f"downloading {URL.format(split=split)} ...", flush=True)
        urllib.request.urlretrieve(URL.format(split=split), archive)
    if md5(archive) != MD5[split]:
        raise RuntimeError(f"{archive}: md5 does not match openslr's")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root, filter="data")  # never writes outside root
    return folder


def utterances(root: Path, split: str) -> list[Utterance]:
    """Every utterance of an extracted split, sorted by id."""
    folder = root / "LibriSpeech" / split
    out = []
    for trans in sorted(folder.glob("*/*/*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            uid, text = line.split(" ", 1)
            out.append(Utterance(uid, split, uid.split("-")[0], trans.parent / f"{uid}.flac", text.strip()))
    return sorted(out, key=lambda u: u.id)


def subset(items: list[Utterance], n: int, seed: int) -> list[Utterance]:
    """A fixed random `n` of the utterances (Python's seeded random.sample over the sorted ids), sorted by id."""
    ordered = sorted(items, key=lambda u: u.id)
    if n >= len(ordered):
        return ordered
    return sorted(random.Random(seed).sample(ordered, n), key=lambda u: u.id)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    a = ap.parse_args()
    for split in a.splits:
        fetch(a.root, split)
        items = utterances(a.root, split)
        print(f"{split}: {len(items)} utterances, {len({u.speaker for u in items})} speakers")


if __name__ == "__main__":
    main()
