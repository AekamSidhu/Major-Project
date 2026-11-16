"""Download GazeBase (6.7 GB) from figshare, resuming if interrupted.

    python download_gazebase.py                    # into data/gazebase/
    python download_gazebase.py --dest D:/datasets/gazebase --verify

The zip is read directly by the training code; there is no need to extract
it. Keep it outside OneDrive or any synced folder if you can.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from readgaze.datasets.gazebase import ARTICLE_API, ZIP_NAME

CHUNK = 1 << 20  # 1 MiB


def file_info() -> dict:
    """Name, size, md5 and download link of the GazeBase zip, from the figshare API."""
    with urllib.request.urlopen(ARTICLE_API, timeout=60) as response:
        article = json.load(response)
    for item in article["files"]:
        if item["name"] == ZIP_NAME:
            return item
    raise SystemExit(f"{ZIP_NAME} not found in figshare article {ARTICLE_API}")


def download(dest_dir: Path, verify: bool) -> Path:
    info = file_info()
    size = int(info["size"])
    target = dest_dir / ZIP_NAME
    dest_dir.mkdir(parents=True, exist_ok=True)

    have = target.stat().st_size if target.exists() else 0
    if have > size:
        raise SystemExit(f"{target} is larger than expected ({have} > {size} bytes); delete it and retry")
    if have < size:
        print(f"Downloading {ZIP_NAME} ({size / 1e9:.2f} GB) to {target}")
        # figshare redirects to a short-lived signed link, so ask for a fresh
        # one each time and resume with a Range request.
        request = urllib.request.Request(info["download_url"], headers={"Range": f"bytes={have}-"} if have else {})
        with urllib.request.urlopen(request, timeout=60) as response, target.open("ab" if have else "wb") as out:
            if have and response.status != 206:
                raise SystemExit("server ignored the resume request; delete the partial file and retry")
            done, last_shown = have, -1
            while chunk := response.read(CHUNK):
                out.write(chunk)
                done += len(chunk)
                percent = int(100 * done / size)
                if percent != last_shown:
                    print(f"\r  {percent:3d}%  {done / 1e9:.2f} / {size / 1e9:.2f} GB", end="", flush=True)
                    last_shown = percent
        print()
    if target.stat().st_size != size:
        raise SystemExit("download incomplete; run the script again to resume")

    if verify:
        print("Checking md5 ...")
        digest = hashlib.md5()
        with target.open("rb") as f:
            while chunk := f.read(CHUNK * 8):
                digest.update(chunk)
        if digest.hexdigest() != info.get("computed_md5"):
            raise SystemExit("md5 mismatch: the file is corrupt; delete it and download again")
        print("md5 OK")
    print(f"Done: {target}")
    return target


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=Path(__file__).parent / "data" / "gazebase")
    parser.add_argument("--verify", action="store_true", help="check the md5 after downloading (takes a minute)")
    args = parser.parse_args(argv)
    download(args.dest, args.verify)


if __name__ == "__main__":
    sys.exit(main())
