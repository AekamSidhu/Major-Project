"""Run the production reading model on a video file and produce a report.

    python validate_video.py clip.mp4
    python validate_video.py clip.mp4 --roi 640 0 640 360   # crop to candidate tile
    python validate_video.py clip.mp4 --label reading        # ground truth for the report

Outputs a JSON report + a PNG timeline plot (if matplotlib is available).
Use this to validate on real interview recordings or webcam captures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "local-ai"))

from integrity.models import ReadingModel, to_input  # noqa: E402

MODEL_DIR = _ROOT / "local-ai" / "integrity" / "models"
FPS = 10.0
WINDOW_FRAMES = 100
HOP_FRAMES = 50

_EYES = ((33, 133, 159, 145, 468), (362, 263, 386, 374, 473))


def iris_offset(points, blink_threshold=0.12):
    xs, ys, openness = [], [], []
    for left_i, right_i, up_i, low_i, iris_i in _EYES:
        left, right = points[left_i, :2], points[right_i, :2]
        axis = right - left
        width = float(np.hypot(*axis))
        if width < 1e-6:
            continue
        along = axis / width
        across = np.array([-along[1], along[0]])
        offset = points[iris_i, :2] - (left + right) / 2
        xs.append(float(offset @ along / width))
        ys.append(float(offset @ across / width))
        openness.append(float(np.hypot(*(points[low_i, :2] - points[up_i, :2])) / width))
    if not xs or float(np.mean(openness)) < blink_threshold:
        return float("nan"), float("nan")
    return float(np.mean(xs)), float(np.mean(ys))


def extract_gaze(video_path: str, roi=None, max_seconds=None):
    """Extract gaze series from video using MediaPipe."""
    import mediapipe as mp

    face_model_path = Path(__file__).parent / "models" / "face_landmarker.task"
    if not face_model_path.exists():
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        face_model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face model...")
        urllib.request.urlretrieve(url, face_model_path)

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(face_model_path)),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Cannot open {video_path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / source_fps

    gaze_points = []
    next_sample = 0.0
    frame_idx = 0

    print(f"Processing {video_path} ({duration:.1f}s, {source_fps:.0f} fps)...")
    t0 = time.time()

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = frame_idx / source_fps
            frame_idx += 1
            if max_seconds and t > max_seconds:
                break
            if t + 1e-9 < next_sample:
                continue
            next_sample += 1.0 / FPS

            if roi:
                x, y, w, h = roi
                frame = frame[y:y+h, x:x+w]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int(round(t * 1000)))

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                ih, iw = frame.shape[:2]
                points = np.array([[p.x * iw, p.y * ih] for p in lm])
                gx, gy = iris_offset(points)
            else:
                gx, gy = float("nan"), float("nan")
            gaze_points.append((gx, gy))

    cap.release()
    elapsed = time.time() - t0
    print(f"  Extracted {len(gaze_points)} gaze samples in {elapsed:.1f}s")
    return np.array(gaze_points, dtype=np.float32)


def score_gaze(model: ReadingModel, gaze: np.ndarray):
    """Score a gaze series through the ONNX reading model."""
    if len(gaze) < WINDOW_FRAMES:
        return {"error": "too short", "frames": len(gaze)}

    starts = list(range(0, len(gaze) - WINDOW_FRAMES + 1, HOP_FRAMES))
    if starts[-1] != len(gaze) - WINDOW_FRAMES:
        starts.append(len(gaze) - WINDOW_FRAMES)

    windows = np.stack([gaze[s:s + WINDOW_FRAMES] for s in starts])
    valid_fracs = np.isfinite(windows).all(axis=2).mean(axis=1)

    keep = valid_fracs >= 0.6
    if not keep.any():
        return {"error": "insufficient valid gaze", "frames": len(gaze)}

    windows = windows[keep]
    kept_starts = [s for s, k in zip(starts, keep) if k]
    inp = to_input(windows)
    probs = model.session.run(None, {"window": inp})[0].astype(float)

    # Answer-level: mean of top half
    sorted_probs = np.sort(probs)[::-1]
    answer_score = float(np.mean(sorted_probs[:max(len(sorted_probs) // 2, 1)]))

    from integrity.models import level
    answer_level = level(answer_score, model.window_possible, model.window_high)

    return {
        "frames": len(gaze),
        "valid_fraction": float(np.isfinite(gaze).all(axis=1).mean()),
        "windows_total": len(starts),
        "windows_scored": int(keep.sum()),
        "window_scores": [{"start_s": s / FPS, "score": float(p)} for s, p in zip(kept_starts, probs)],
        "answer_score": answer_score,
        "answer_level": answer_level,
        "thresholds": {"possible": model.window_possible, "high": model.window_high},
    }


def plot_timeline(result, out_path, label=None):
    """Save a timeline plot of window scores."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping plot")
        return

    scores = result["window_scores"]
    times = [s["start_s"] for s in scores]
    vals = [s["score"] for s in scores]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(times, vals, alpha=0.3, color="#2196F3")
    ax.plot(times, vals, color="#2196F3", linewidth=2)
    ax.axhline(result["thresholds"]["possible"], color="orange", linestyle="--", linewidth=1, label="possible")
    ax.axhline(result["thresholds"]["high"], color="red", linestyle="--", linewidth=1, label="high")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reading probability")
    title = f"Reading Detection Timeline"
    if label:
        title += f" [{label}]"
    title += f" — answer score: {result['answer_score']:.3f} ({result['answer_level']})"
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    ap.add_argument("--label", help="ground truth label for the report (reading / normal)")
    ap.add_argument("--max-seconds", type=float)
    ap.add_argument("--out", type=Path, help="output directory (default: alongside video)")
    args = ap.parse_args()

    out_dir = args.out or args.video.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ReadingModel(MODEL_DIR)
    print(f"Model: {model.name}")
    print(f"Thresholds: possible={model.window_possible:.4f}, high={model.window_high:.4f}")

    gaze = extract_gaze(str(args.video), roi=tuple(args.roi) if args.roi else None,
                        max_seconds=args.max_seconds)
    result = score_gaze(model, gaze)

    if "error" in result:
        print(f"  Error: {result['error']}")
        return

    # Report
    report = {
        "video": str(args.video),
        "label": args.label,
        **result,
    }

    stem = args.video.stem
    json_path = out_dir / f"{stem}_reading_report.json"
    json_path.write_text(json.dumps(report, indent=2))
    print(f"  Report: {json_path}")

    plot_timeline(result, out_dir / f"{stem}_reading_timeline.png", label=args.label)

    # Summary
    print(f"\n{'='*50}")
    print(f"VIDEO: {args.video.name}")
    if args.label:
        print(f"LABEL: {args.label}")
    print(f"ANSWER SCORE: {result['answer_score']:.3f}")
    print(f"LEVEL: {result['answer_level']}")
    print(f"Windows: {result['windows_scored']}/{result['windows_total']} scored")
    print(f"Valid gaze: {result['valid_fraction']:.0%}")
    if args.label == "reading":
        correct = result["answer_level"] in ("high", "possible")
        print(f"DETECTION: {'CORRECT' if correct else 'MISSED'}")
    elif args.label == "normal":
        correct = result["answer_level"] == "low"
        print(f"DETECTION: {'CORRECT (no flag)' if correct else 'FALSE POSITIVE'}")


if __name__ == "__main__":
    main()
