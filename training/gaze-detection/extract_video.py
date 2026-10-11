"""Run the gaze pipeline on a video and print a reading score per window.

    python extract_video.py clip.mp4
    python extract_video.py meeting.mp4 --roi 640 0 640 360 --model results/model.json

Without --model, the rule-based detector scores the windows (no training
needed). The per-frame gaze is saved as a CSV next to the data, never in git:
treat it as participant data if the video shows a real person.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from readgaze.detector import aggregate, rule_score
from readgaze.features import FeatureConfig, compute_features
from readgaze.landmarks import FaceTracker
from readgaze.model import score_from_json
from readgaze.video import face_found_fraction, frames_to_trace, video_frames

HERE = Path(__file__).parent


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), help="crop to the candidate tile")
    parser.add_argument("--window", type=float, default=10.0, help="window length in seconds")
    parser.add_argument("--hop", type=float, default=5.0, help="seconds between window starts")
    parser.add_argument("--model", type=Path, help="model.json from train.py (default: rule-based detector)")
    parser.add_argument("--face-model", type=Path, default=HERE / "models" / "face_landmarker.task")
    parser.add_argument("--out", type=Path, help="per-frame CSV (default: data/videos/<name>_frames.csv)")
    args = parser.parse_args(argv)

    model = json.loads(args.model.read_text()) if args.model else None
    with FaceTracker(args.face_model) as tracker:
        frames = video_frames(str(args.video), tracker, fps=args.fps, roi=tuple(args.roi) if args.roi else None)

    out = args.out or HERE / "data" / "videos" / f"{args.video.stem}_frames.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames.to_csv(out, index=False)
    print(f"{len(frames)} frames, face found in {face_found_fraction(frames):.0%}; saved {out}")

    trace = frames_to_trace(frames, str(args.video))
    scores = []
    print(f"{'start':>6}  {'score':>5}  sweeps/min  skew  asym")
    for window in trace.windows(args.window, args.hop):
        features = compute_features(window, FeatureConfig())
        start = window.meta["window_start"]
        if features is None:
            print(f"{start:6.1f}      -  (not enough valid gaze)")
            continue
        score = score_from_json(model, features) if model else rule_score(features)
        scores.append(score)
        print(
            f"{start:6.1f}  {score:5.2f}  {features['sweep_rate']:10.1f}"
            f"  {features['velocity_skew']:4.1f}  {features['sweep_asymmetry']:4.2f}"
        )
    overall = aggregate(scores)
    detector = "model" if model else "rule-based"
    print(f"Answer-level reading score ({detector}): {overall:.2f}")
    if model and overall == overall:  # not NaN
        verdict = "would be flagged" if overall >= model["threshold"] else "would not be flagged"
        print(f"Threshold {model['threshold']:.2f}: {verdict} (a signal for the interviewer, not a verdict)")


if __name__ == "__main__":
    main()
