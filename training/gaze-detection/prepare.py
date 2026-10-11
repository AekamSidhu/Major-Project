"""Build the feature table: recordings → webcam-quality traces → window features.

    python prepare.py                                   # all of GazeBase
    python prepare.py --max-participants 10             # quick run
    python prepare.py --noise 3 --amplitude 0.5         # harder conditions
    python prepare.py --synthetic 8                     # no download needed (smoke test only)

Writes a CSV (one row per window) and a .json file next to it with the exact
settings, so every result can be traced back to how its data was made.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from readgaze.datasets.gazebase import TASKS, ZIP_NAME
from readgaze.features import FeatureConfig
from readgaze.pipeline import WindowConfig, gazebase_table, synthetic_table
from readgaze.simulate import WebcamProfile

HERE = Path(__file__).parent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=HERE / "data" / "gazebase" / ZIP_NAME, help="GazeBase zip or folder")
    parser.add_argument("--out", type=Path, help="output CSV (default: data/features/<name>.csv)")
    parser.add_argument("--synthetic", type=int, metavar="N", help="use N synthetic participants instead of GazeBase")
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), help="GazeBase tasks to include (default: all)")
    parser.add_argument("--max-participants", type=int)
    parser.add_argument("--fps", type=float, default=WebcamProfile.fps)
    parser.add_argument("--noise", type=float, default=WebcamProfile.noise)
    parser.add_argument("--drift", type=float, default=WebcamProfile.drift)
    parser.add_argument("--amplitude", type=float, default=WebcamProfile.amplitude)
    parser.add_argument("--window", type=float, default=WindowConfig.length)
    parser.add_argument("--hop", type=float, default=WindowConfig.hop)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    profile = WebcamProfile(fps=args.fps, noise=args.noise, drift=args.drift, amplitude=args.amplitude)
    window = WindowConfig(length=args.window, hop=args.hop)
    features = FeatureConfig()

    started = time.time()
    if args.synthetic:
        name = f"synthetic_{args.fps:g}fps"
        table = synthetic_table(args.synthetic, profile=profile, window=window, features=features, seed=args.seed)
    else:
        if not args.data.exists():
            raise SystemExit(f"{args.data} not found. Run download_gazebase.py first, or pass --data.")
        name = f"gazebase_{args.fps:g}fps"
        table = gazebase_table(
            str(args.data), profile, window, features, args.tasks, args.max_participants, args.seed
        )
    if table.empty:
        raise SystemExit("no usable windows; check --data and --tasks")

    out = args.out or HERE / "data" / "features" / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    settings = {
        "source": "synthetic" if args.synthetic else str(args.data),
        "tasks": args.tasks,
        "max_participants": args.max_participants,
        "seed": args.seed,
        "webcam": asdict(profile),
        "window": asdict(window),
        "features": asdict(features),
    }
    out.with_suffix(".json").write_text(json.dumps(settings, indent=2))

    counts = table.groupby("task")["recording"].nunique().to_dict()
    print(f"{len(table)} windows from {table['participant'].nunique()} participants in {time.time() - started:.0f} s")
    print(f"recordings per task: {counts}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
