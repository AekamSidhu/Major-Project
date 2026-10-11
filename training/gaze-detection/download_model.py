"""Download MediaPipe's Face Landmarker model (3.8 MB) into models/.

    python download_model.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from readgaze.landmarks import MODEL_URL

DEFAULT_PATH = Path(__file__).parent / "models" / "face_landmarker.task"


def main() -> None:
    DEFAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DEFAULT_PATH.exists():
        print(f"Already present: {DEFAULT_PATH}")
        return
    print(f"Downloading {MODEL_URL}")
    partial = DEFAULT_PATH.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, partial)
    partial.replace(DEFAULT_PATH)
    print(f"Saved {DEFAULT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
