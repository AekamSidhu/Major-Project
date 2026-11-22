"""Live webcam demo: reading detection in real-time.

Shows a window with the webcam feed, overlaid with:
  - Current iris position (green dot on each eye)
  - Rolling 10-second reading score (bar + level)
  - Per-window score history

Usage:
    cd training/gaze-detection
    python demo_reading.py                  # webcam
    python demo_reading.py video.mp4        # recorded video
    python demo_reading.py --onnx           # use the production ONNX model

Press Q or Esc to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "local-ai"))

MODEL_DIR = _ROOT / "local-ai" / "integrity" / "models"
FPS = 10.0
WINDOW_FRAMES = 100  # 10 seconds
HOP_FRAMES = 50      # 5 seconds


def load_onnx_model():
    from integrity.models import ReadingModel
    return ReadingModel(MODEL_DIR)


def load_mediapipe_tracker():
    model_path = Path(__file__).parent / "models" / "face_landmarker.task"
    if not model_path.exists():
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face model to {model_path} ...")
        urllib.request.urlretrieve(url, model_path)
    from readgaze.landmarks import FaceTracker
    return FaceTracker(model_path)


def iris_offset_from_landmarks(points: np.ndarray, blink_threshold: float = 0.12):
    """Same iris_offset as local-ai/integrity/gaze.py."""
    _EYES = ((33, 133, 159, 145, 468), (362, 263, 386, 374, 473))
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


def draw_overlay(frame, gaze_history, scores, current_level, model_possible, model_high):
    """Draw gaze tracking overlay on the frame."""
    h, w = frame.shape[:2]

    # Score bar in top-right
    bar_x, bar_y, bar_w, bar_h = w - 220, 20, 200, 30
    cv2.rectangle(frame, (bar_x - 5, bar_y - 25), (bar_x + bar_w + 5, bar_y + bar_h + 50), (0, 0, 0), -1)
    cv2.putText(frame, "Reading Score", (bar_x, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if scores:
        latest = scores[-1]
        fill_w = int(bar_w * min(latest, 1.0))
        color = (0, 0, 255) if latest >= model_high else (0, 165, 255) if latest >= model_possible else (0, 200, 0)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)

        # Threshold markers
        px_possible = bar_x + int(bar_w * model_possible)
        px_high = bar_x + int(bar_w * model_high)
        cv2.line(frame, (px_possible, bar_y), (px_possible, bar_y + bar_h), (0, 255, 255), 1)
        cv2.line(frame, (px_high, bar_y), (px_high, bar_y + bar_h), (0, 0, 255), 1)

        level_text = f"{latest:.2f} [{current_level}]"
        cv2.putText(frame, level_text, (bar_x, bar_y + bar_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f"P={model_possible:.2f} H={model_high:.2f}", (bar_x, bar_y + bar_h + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    # Score history graph (bottom-right)
    if len(scores) > 1:
        graph_x, graph_y, graph_w, graph_h = w - 220, h - 100, 200, 80
        cv2.rectangle(frame, (graph_x - 2, graph_y - 2), (graph_x + graph_w + 2, graph_y + graph_h + 2), (0, 0, 0), -1)
        cv2.putText(frame, "History", (graph_x, graph_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        pts = np.array(scores[-20:])  # last 20 windows
        for i in range(len(pts) - 1):
            x1 = graph_x + int(i / max(len(pts) - 1, 1) * graph_w)
            x2 = graph_x + int((i + 1) / max(len(pts) - 1, 1) * graph_w)
            y1 = graph_y + graph_h - int(min(pts[i], 1.0) * graph_h)
            y2 = graph_y + graph_h - int(min(pts[i + 1], 1.0) * graph_h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 200), 2)
        # Threshold lines
        y_p = graph_y + graph_h - int(min(model_possible, 1.0) * graph_h)
        y_h = graph_y + graph_h - int(min(model_high, 1.0) * graph_h)
        cv2.line(frame, (graph_x, y_p), (graph_x + graph_w, y_p), (0, 255, 255), 1)
        cv2.line(frame, (graph_x, y_h), (graph_x + graph_w, y_h), (0, 0, 255), 1)

    # Gaze position trail (top-left mini-map)
    map_x, map_y, map_sz = 20, 20, 120
    cv2.rectangle(frame, (map_x, map_y), (map_x + map_sz, map_y + map_sz), (40, 40, 40), -1)
    cv2.putText(frame, "Gaze", (map_x + 2, map_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    # Draw last 50 gaze points
    recent = list(gaze_history)[-50:]
    for i, (gx, gy) in enumerate(recent):
        if np.isnan(gx) or np.isnan(gy):
            continue
        px = map_x + map_sz // 2 + int(gx * map_sz * 2)
        py = map_y + map_sz // 2 + int(gy * map_sz * 2)
        px = max(map_x, min(px, map_x + map_sz))
        py = max(map_y, min(py, map_y + map_sz))
        alpha = int(100 + 155 * i / max(len(recent), 1))
        cv2.circle(frame, (px, py), 2, (0, alpha, 0), -1)
    cv2.rectangle(frame, (map_x, map_y), (map_x + map_sz, map_y + map_sz), (100, 100, 100), 1)

    # Status
    n_gaze = sum(1 for gx, gy in gaze_history if not np.isnan(gx))
    secs = n_gaze / FPS
    cv2.putText(frame, f"Gaze: {secs:.0f}s tracked", (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", help="video file (default: webcam)")
    ap.add_argument("--onnx", action="store_true", help="use the production ONNX model")
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    args = ap.parse_args()

    print("Loading models...")
    tracker = load_mediapipe_tracker()
    if args.onnx:
        onnx_model = load_onnx_model()
        possible_thresh = onnx_model.window_possible
        high_thresh = onnx_model.window_high
        print(f"ONNX model loaded. Thresholds: possible={possible_thresh:.3f}, high={high_thresh:.3f}")
    else:
        possible_thresh = 0.14
        high_thresh = 0.59
        print("Using default thresholds (no ONNX model).")

    # Open video source
    if args.video:
        cap = cv2.VideoCapture(args.video)
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        source_fps = 30.0

    if not cap.isOpened():
        print("Cannot open video source.")
        return

    gaze_history: deque[tuple[float, float]] = deque(maxlen=int(FPS * 600))
    window_scores: list[float] = []
    current_level = "low"
    next_sample = 0.0
    frame_idx = 0

    print("Running... Press Q or Esc to quit.")
    print("Try: look at the camera normally, then read text on your screen.")

    with tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t = frame_idx / source_fps
            frame_idx += 1

            # Downsample to 10 fps
            if t + 1e-9 < next_sample:
                # Still draw overlay and show
                draw_overlay(frame, gaze_history, window_scores, current_level, possible_thresh, high_thresh)
                cv2.imshow("Reading Detection Demo", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue
            next_sample += 1.0 / FPS

            # Track gaze
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gaze = tracker.track(rgb, int(round(t * 1000)))
            gaze_history.append((gaze.x, gaze.y))

            # Score when we have enough windows
            n = len(gaze_history)
            if n >= WINDOW_FRAMES and (n - WINDOW_FRAMES) % HOP_FRAMES == 0:
                window_data = np.array(list(gaze_history)[-WINDOW_FRAMES:], dtype=np.float32)
                valid_frac = np.isfinite(window_data).all(axis=1).mean()
                if valid_frac >= 0.6:
                    if args.onnx:
                        from integrity.models import to_input
                        inp = to_input(window_data[np.newaxis])
                        prob = float(onnx_model.session.run(None, {"window": inp})[0][0])
                    else:
                        prob = score_window_rule(window_data)
                    window_scores.append(prob)
                    # Answer-level: mean of top half
                    if len(window_scores) >= 2:
                        sorted_scores = sorted(window_scores[-20:], reverse=True)
                        answer_score = np.mean(sorted_scores[:max(len(sorted_scores) // 2, 1)])
                    else:
                        answer_score = prob
                    if answer_score >= high_thresh:
                        current_level = "HIGH"
                    elif answer_score >= possible_thresh:
                        current_level = "possible"
                    else:
                        current_level = "low"

            # Draw overlay
            draw_overlay(frame, gaze_history, window_scores, current_level, possible_thresh, high_thresh)
            cv2.imshow("Reading Detection Demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Final report
    if window_scores:
        print(f"\n{'='*50}")
        print("SESSION RESULTS")
        print(f"{'='*50}")
        print(f"Windows scored: {len(window_scores)}")
        print(f"Score range: {min(window_scores):.3f} - {max(window_scores):.3f}")
        print(f"Mean: {np.mean(window_scores):.3f}")
        sorted_all = sorted(window_scores, reverse=True)
        answer = np.mean(sorted_all[:max(len(sorted_all) // 2, 1)])
        print(f"Answer-level (top half mean): {answer:.3f}")
        print(f"Level: {current_level}")
    else:
        print("\nNot enough gaze data collected for scoring.")


def score_window_rule(window: np.ndarray) -> float:
    """Simple rule-based scoring fallback when ONNX is not used."""
    valid = np.isfinite(window).all(axis=1)
    if valid.sum() < 60:
        return 0.0
    x = window[valid, 0]
    dx = np.diff(x)
    positive_runs = 0
    run_len = 0
    for d in dx:
        if d > 0.001:
            run_len += 1
        else:
            if run_len >= 5:
                positive_runs += 1
            run_len = 0
    if run_len >= 5:
        positive_runs += 1
    sweep_rate = positive_runs / (valid.sum() / FPS) * 60
    return min(sweep_rate / 20.0, 1.0)


if __name__ == "__main__":
    main()
