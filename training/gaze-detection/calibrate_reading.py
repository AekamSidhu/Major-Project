"""Calibrate the reading detector against synthetic interview gaze patterns.

Generates two types of gaze:
  - "Reading from overlay": systematic left-to-right sweeps with return
    saccades, as a candidate reading Parakeet AI output would produce.
  - "Normal interview": mostly centred with natural drift and occasional
    glances (looking at camera, thinking, gesturing).

Runs both through the ONNX reading model and reports detection accuracy.
This validates whether the GazeBase-trained model generalises to the
interview reading scenario without needing labelled interview video.

    python calibrate_reading.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "local-ai"))
from integrity.models import ReadingModel, to_input  # noqa: E402

MODEL_DIR = _ROOT / "local-ai" / "integrity" / "models"

FPS = 10
WINDOW = 100  # 10 seconds at 10 fps


def generate_reading_gaze(
    duration_s: float = 30.0,
    reading_speed_wpm: float = 200.0,
    chars_per_line: int = 60,
    line_height: float = 0.03,
    saccade_noise: float = 0.005,
    blink_rate: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    """Synthetic gaze for someone reading text from a screen overlay.

    Produces left-to-right sweeps across lines with return saccades,
    fixation jitter, and occasional blinks. The x/y values are in
    iris-offset units (fraction of eye width), matching the model's input.
    """
    rng = np.random.default_rng(seed)
    n_frames = int(duration_s * FPS)
    words_per_second = reading_speed_wpm / 60.0
    chars_per_second = words_per_second * 5  # ~5 chars per word
    frames_per_line = int(chars_per_line / chars_per_second * FPS)
    frames_per_line = max(frames_per_line, 10)

    gaze = np.empty((n_frames, 2), dtype=np.float32)
    frame = 0
    line = 0
    while frame < n_frames:
        line_frames = frames_per_line + rng.integers(-5, 6)
        line_frames = max(10, min(line_frames, n_frames - frame))
        for f in range(line_frames):
            if frame + f >= n_frames:
                break
            progress = f / max(line_frames - 1, 1)
            x = -0.15 + 0.30 * progress  # sweep left to right
            y = 0.02 + line * line_height  # move down with lines
            x += rng.normal(0, saccade_noise)
            y += rng.normal(0, saccade_noise * 0.5)
            # Occasional fixation pauses (saccade-fixate pattern)
            if rng.random() < 0.05:
                x += rng.normal(0, 0.01)
            gaze[frame + f] = [x, y]
        frame += line_frames
        # Return saccade (jump back to start of next line)
        if frame < n_frames:
            line += 1
            if line > 8:  # scroll / reset
                line = 0
        # Blink during return saccade
        if frame < n_frames and rng.random() < blink_rate:
            blink_len = rng.integers(2, 5)
            end = min(frame + blink_len, n_frames)
            gaze[frame:end] = np.nan
            frame = end

    return gaze[:n_frames]


def generate_normal_gaze(
    duration_s: float = 30.0,
    drift_speed: float = 0.01,
    glance_prob: float = 0.02,
    blink_rate: float = 0.03,
    seed: int = 42,
) -> np.ndarray:
    """Synthetic gaze for someone looking at the camera during an interview.

    Mostly centred with slow drift, occasional glances to the side
    (thinking, gesturing), and regular blinks. No systematic scanning.
    """
    rng = np.random.default_rng(seed)
    n_frames = int(duration_s * FPS)
    gaze = np.empty((n_frames, 2), dtype=np.float32)

    x, y = 0.0, 0.0
    for f in range(n_frames):
        # Random walk with pull toward centre
        x += rng.normal(0, drift_speed) - 0.02 * x
        y += rng.normal(0, drift_speed * 0.5) - 0.02 * y

        # Occasional glance away (thinking, looking at notes briefly)
        if rng.random() < glance_prob:
            x += rng.choice([-0.1, 0.1]) * rng.random()
            y += rng.choice([-0.05, 0.05]) * rng.random()

        # Blink
        if rng.random() < blink_rate:
            gaze[f] = np.nan
            continue

        gaze[f] = [np.clip(x, -0.3, 0.3), np.clip(y, -0.15, 0.15)]

    return gaze


def score_gaze(model: ReadingModel, gaze: np.ndarray) -> dict:
    """Score a gaze series through the reading model."""
    result = model.assess(gaze)
    if result is None:
        return {"probability": 0.0, "level": "too_short", "windows": []}
    return {
        "probability": result.probability,
        "level": result.level,
        "windows": result.windows,
        "valid_fraction": result.valid_fraction,
    }


def main() -> None:
    model = ReadingModel(MODEL_DIR)
    print(f"Model: {model.name}")
    print(f"Window thresholds: possible={model.window_possible:.4f}, high={model.window_high:.4f}")

    # --- Generate scenarios ---
    reading_configs = [
        {"label": "Fast reading (250 wpm)", "reading_speed_wpm": 250, "duration_s": 30, "seed": 1},
        {"label": "Normal reading (200 wpm)", "reading_speed_wpm": 200, "duration_s": 30, "seed": 2},
        {"label": "Slow reading (150 wpm)", "reading_speed_wpm": 150, "duration_s": 30, "seed": 3},
        {"label": "Quick scan (300 wpm, 15s)", "reading_speed_wpm": 300, "duration_s": 15, "seed": 4},
        {"label": "Reading with distractions", "reading_speed_wpm": 200, "duration_s": 30, "seed": 5, "blink_rate": 0.25},
        {"label": "Narrow text (40 chars)", "reading_speed_wpm": 200, "duration_s": 30, "seed": 6, "chars_per_line": 40},
        {"label": "Wide text (80 chars)", "reading_speed_wpm": 200, "duration_s": 30, "seed": 7, "chars_per_line": 80},
        {"label": "Reading 60s answer", "reading_speed_wpm": 200, "duration_s": 60, "seed": 8},
    ]
    normal_configs = [
        {"label": "Looking at camera", "duration_s": 30, "seed": 10},
        {"label": "Animated speaker (more glances)", "duration_s": 30, "seed": 11, "glance_prob": 0.05},
        {"label": "Nervous (more drift)", "duration_s": 30, "seed": 12, "drift_speed": 0.02},
        {"label": "Still speaker", "duration_s": 30, "seed": 13, "drift_speed": 0.005},
        {"label": "Long answer (60s)", "duration_s": 60, "seed": 14},
        {"label": "Short answer (15s)", "duration_s": 15, "seed": 15},
        {"label": "Lots of blinking", "duration_s": 30, "seed": 16, "blink_rate": 0.08},
        {"label": "Looking at notes briefly", "duration_s": 30, "seed": 17, "glance_prob": 0.08},
    ]

    print("\n" + "=" * 70)
    print("READING FROM SCREEN (should score HIGH)")
    print("=" * 70)
    reading_scores = []
    for cfg in reading_configs:
        label = cfg.pop("label")
        gaze = generate_reading_gaze(**cfg)
        result = score_gaze(model, gaze)
        reading_scores.append(result["probability"])
        level = result["level"]
        prob = result["probability"]
        n_win = len(result.get("windows", []))
        print(f"  {label:40s}  prob={prob:.3f}  level={level:8s}  windows={n_win}")

    print("\n" + "=" * 70)
    print("NORMAL INTERVIEW (should score LOW)")
    print("=" * 70)
    normal_scores = []
    for cfg in normal_configs:
        label = cfg.pop("label")
        gaze = generate_normal_gaze(**cfg)
        result = score_gaze(model, gaze)
        normal_scores.append(result["probability"])
        level = result["level"]
        prob = result["probability"]
        n_win = len(result.get("windows", []))
        print(f"  {label:40s}  prob={prob:.3f}  level={level:8s}  windows={n_win}")

    # --- Summary ---
    reading_arr = np.array(reading_scores)
    normal_arr = np.array(normal_scores)
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Reading scores:  mean={reading_arr.mean():.3f}  min={reading_arr.min():.3f}  max={reading_arr.max():.3f}")
    print(f"  Normal scores:   mean={normal_arr.mean():.3f}  min={normal_arr.min():.3f}  max={normal_arr.max():.3f}")

    # Detection rates at current thresholds
    reading_possible = (reading_arr >= model.window_possible).mean()
    reading_high = (reading_arr >= model.window_high).mean()
    normal_possible = (normal_arr >= model.window_possible).mean()
    normal_high = (normal_arr >= model.window_high).mean()
    print(f"\n  At 'possible' ({model.window_possible:.3f}): reading detected {100*reading_possible:.0f}%, normal FPR {100*normal_possible:.0f}%")
    print(f"  At 'high' ({model.window_high:.3f}):     reading detected {100*reading_high:.0f}%, normal FPR {100*normal_high:.0f}%")

    # Optimal threshold
    all_scores = np.concatenate([reading_arr, normal_arr])
    all_labels = np.concatenate([np.ones(len(reading_arr)), np.zeros(len(normal_arr))])
    thresholds = np.sort(np.unique(all_scores))
    best_ba, best_t = 0.0, 0.5
    for t in thresholds:
        tp = ((all_scores >= t) & (all_labels == 1)).sum()
        fp = ((all_scores >= t) & (all_labels == 0)).sum()
        rec = tp / max(all_labels.sum(), 1)
        spec = 1 - fp / max((1 - all_labels).sum(), 1)
        ba = (rec + spec) / 2
        if ba > best_ba:
            best_ba, best_t = ba, t
    print(f"\n  Optimal threshold: {best_t:.4f} (BA={best_ba:.3f})")

    # Combined signal
    print("\n" + "=" * 70)
    print("COMBINED SIGNAL (echo + reading together)")
    print("=" * 70)
    print("  When both echo (AI text) AND reading (gaze) flag the same answer:")
    print("  - Each is an independent signal measuring different things")
    print("  - Joint false-positive probability = P(echo FP) × P(reading FP)")
    print(f"  - Echo FPR=0% at possible, Reading FPR={100*normal_possible:.0f}% at possible")
    print("  - Combined: extremely strong evidence of answer copying")
    print("  -> For the demo: show both scores side by side, highlight when both flag")


if __name__ == "__main__":
    main()
