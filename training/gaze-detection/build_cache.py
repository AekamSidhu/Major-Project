"""Turn a downloaded eye-tracking dataset into a 30 Hz frame cache (one .npz).

    python build_cache.py gazebase        D:/data/GazeBase_v2_0.zip   D:/data/cache/gazebase.npz
    python build_cache.py gazebasevr      D:/data/gazebasevr.zip      D:/data/cache/gazebasevr.npz
    python build_cache.py desktopactivity D:/data/DesktopActivity.zip D:/data/cache/desktopactivity.npz

Do this once per dataset. Everything after it (`reading_experiments.py`)
works from the cache, so trying another frame rate, noise level or model
takes minutes. GazeBase takes about 15 minutes on four cores.

The cache holds per-person gaze data, so keep it with the dataset, outside
the repository.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from readgaze.cache import Frames, TraceCache, bin_samples  # noqa: E402
from readgaze.datasets import desktopactivity, gazebase, gazebasevr  # noqa: E402


def _gazebase_recording(name: str, raw: bytes) -> tuple[dict, Frames] | None:
    rec = gazebase.RecordingId.parse(name)
    if rec is None:
        return None
    frame = pd.read_csv(io.BytesIO(raw), usecols=["n", "x", "y", "val"])
    frame = frame.drop_duplicates(subset="n").sort_values("n")
    invalid = frame["val"].to_numpy() != 0
    x = frame["x"].to_numpy(dtype=float, copy=True)
    y = frame["y"].to_numpy(dtype=float, copy=True)
    x[invalid] = np.nan
    y[invalid] = np.nan
    info = {
        "dataset": "GazeBase",
        "participant": f"gb{rec.participant:03d}",
        "recording": f"r{rec.round}-p{rec.participant:03d}-s{rec.session}-{rec.task}",
        "task": rec.task,
        "label": int(rec.is_reading),
        "round": rec.round,
        "session": rec.session,
    }
    return info, bin_samples(frame["n"].to_numpy(dtype=float) / 1000.0, x, y)


def _gazebasevr_recording(name: str, raw: bytes) -> tuple[dict, Frames] | None:
    rec = gazebasevr.RecordingId.parse(name)
    if rec is None:
        return None
    t, x, y = gazebasevr.load_csv(raw)
    info = {
        "dataset": "GazeBaseVR",
        "participant": f"vr{rec.participant:03d}",
        "recording": f"r{rec.round}-p{rec.participant:03d}-s{rec.session}-{rec.task}",
        "task": rec.task,
        "label": int(rec.is_reading),
        "round": rec.round,
        "session": rec.session,
    }
    return info, bin_samples(t, x, y)


READERS = {"gazebase": _gazebase_recording, "gazebasevr": _gazebasevr_recording}


def _walk(archive: zipfile.ZipFile, names: list[str], reader) -> list[tuple[dict, Frames]]:
    """Every recording under `names`, opening nested zips in memory."""
    out = []
    for name in names:
        lower = name.lower()
        if lower.endswith(".csv"):
            try:
                item = reader(name, archive.read(name))
            except ValueError:
                item = None  # shorter than one frame, or an empty file
            if item is not None:
                out.append(item)
        elif lower.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(archive.read(name))) as inner:
                out += _walk(inner, sorted(inner.namelist()), reader)
    return out


def _work(job: tuple[str, str, list[str]]) -> list[tuple[dict, Frames]]:
    kind, zip_path, names = job
    with zipfile.ZipFile(zip_path) as archive:
        return _walk(archive, names, READERS[kind])


def build_nested(kind: str, zip_path: Path, workers: int) -> TraceCache:
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(n for n in archive.namelist() if n.lower().endswith((".csv", ".zip")))
    # A job is a few participant zips (GazeBase) or a few dozen CSVs (GazeBaseVR).
    chunk = 4 if any(n.lower().endswith(".zip") for n in names) else 50
    jobs = [(kind, str(zip_path), names[i : i + chunk]) for i in range(0, len(names), chunk)]
    rows, frames = [], []
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, result in enumerate(pool.map(_work, jobs), start=1):
            for info, item in result:
                rows.append(info)
                frames.append(item)
            if done % 10 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} chunks, {len(rows)} recordings, {time.time() - started:.0f} s", flush=True)
    order = sorted(range(len(rows)), key=lambda i: rows[i]["recording"])
    return TraceCache.from_recordings([rows[i] for i in order], [frames[i] for i in order])


def build_desktopactivity(zip_path: Path) -> TraceCache:
    rows, frames = [], []
    for info, t, x, y in desktopactivity.iter_recordings(zip_path):
        rows.append(info)
        frames.append(bin_samples(t, x, y, desktopactivity.RATE))
    return TraceCache.from_recordings(rows, frames, desktopactivity.RATE)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=["gazebase", "gazebasevr", "desktopactivity"])
    parser.add_argument("source", type=Path, help="the downloaded zip")
    parser.add_argument("out", type=Path, help="the cache to write (.npz)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    if args.dataset == "desktopactivity":
        cache = build_desktopactivity(args.source)
    else:
        cache = build_nested(args.dataset, args.source, args.workers)
    cache.save(args.out)
    hours = len(cache.x) / cache.rate / 3600
    print(f"{len(cache)} recordings, {cache.meta['participant'].nunique()} people, {hours:.1f} hours -> {args.out}")
    print(cache.meta.groupby("task").size().to_string())


if __name__ == "__main__":
    main()
