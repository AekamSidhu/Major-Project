"""Exploration helper: one degraded feature table (+ raw windows) from a cache."""
import argparse, os, sys, time
from pathlib import Path
import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from readgaze.pipeline import WindowConfig, cache_windows  # noqa: E402
from readgaze.simulate import ProfileRange, WebcamProfile  # noqa: E402

DATA = Path(os.environ.get("OPENMEET_DATA", r"C:/Users/karaa/openmeet-data"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("name")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--noise", type=float, default=1.5)
    ap.add_argument("--drift", type=float, default=1.0)
    ap.add_argument("--amplitude", type=float, default=1.0)
    ap.add_argument("--blinks", type=float, default=15.0)
    ap.add_argument("--dropout", type=float, default=0.02)
    ap.add_argument("--length", type=float, default=10.0)
    ap.add_argument("--hop", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--vary", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--dev-only", action="store_true")
    ap.add_argument("--no-ext", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    profile = WebcamProfile(fps=a.fps, noise=a.noise, drift=a.drift, amplitude=a.amplitude, blinks_per_minute=a.blinks, dropout_probability=a.dropout)
    people = None
    if a.dev_only:
        from readgaze.cache import TraceCache
        from readgaze.protocol import split_people
        people = split_people(TraceCache.load(str(DATA / "cache" / f"{a.dataset}.npz")).meta["participant"])[0]
    t0 = time.time()
    table, raw = cache_windows(str(DATA / "cache" / f"{a.dataset}.npz"), profile, WindowConfig(a.length, a.hop), seed=a.seed, workers=a.workers, extended=not a.no_ext, clean=a.clean, vary=ProfileRange() if a.vary else None, speed=a.speed, participants=people)
    out = DATA / "tables"
    out.mkdir(exist_ok=True)
    table.to_pickle(out / f"{a.name}.pkl")
    np.save(out / f"{a.name}_raw.npy", raw)
    print(f"{a.name}: {len(table)} windows, {table['recording'].nunique()} recordings, {table['participant'].nunique()} people, raw {raw.shape}, {time.time()-t0:.0f} s")
    print(table.groupby("task").size().to_string())
