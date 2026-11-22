"""Evaluate reading detection on synthetic gaze patterns.

Tests the rule-based scorer against various reading and non-reading patterns,
including right-to-left reading (flipped screen) and edge cases.

    python evaluate_reading.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from demo_reading import score_early_reading, score_window_rule

FPS = 10.0
WINDOW = 100  # 10 s at 10 fps
SHORT = 30    # 3 s at 10 fps — early detection


def make_ltr_reading(speed_factor: float = 1.0, seed: int = 0) -> np.ndarray:
    """Left-to-right reading with return saccades."""
    rng = np.random.default_rng(seed)
    gaze = np.empty((WINDOW, 2), dtype=np.float32)
    frame = 0
    line = 0
    frames_per_line = int(30 / speed_factor)
    while frame < WINDOW:
        line_len = min(frames_per_line + rng.integers(-3, 4), WINDOW - frame)
        line_len = max(8, line_len)
        for f in range(line_len):
            if frame + f >= WINDOW:
                break
            progress = f / max(line_len - 1, 1)
            x = -0.15 + 0.30 * progress
            y = 0.02 + line * 0.03
            x += rng.normal(0, 0.005)
            y += rng.normal(0, 0.003)
            gaze[frame + f] = [x, y]
        frame += line_len
        line += 1
        if line > 5:
            line = 0
    return gaze[:WINDOW]


def make_rtl_reading(speed_factor: float = 1.0, seed: int = 0) -> np.ndarray:
    """Right-to-left reading (flipped screen / RTL language)."""
    gaze = make_ltr_reading(speed_factor, seed)
    gaze[:, 0] = -gaze[:, 0]  # mirror x
    return gaze


def make_zigzag_reading(seed: int = 0) -> np.ndarray:
    """Alternating L→R and R→L sweeps (like re-reading or scanning)."""
    rng = np.random.default_rng(seed)
    gaze = np.empty((WINDOW, 2), dtype=np.float32)
    frame = 0
    direction = 1  # 1 = L→R, -1 = R→L
    while frame < WINDOW:
        line_len = min(15 + rng.integers(-3, 4), WINDOW - frame)
        for f in range(line_len):
            if frame + f >= WINDOW:
                break
            progress = f / max(line_len - 1, 1)
            x = direction * (-0.15 + 0.30 * progress)
            y = rng.normal(0, 0.005)
            gaze[frame + f] = [x, y]
        frame += line_len
        direction *= -1
    return gaze[:WINDOW]


def make_camera_look(seed: int = 0) -> np.ndarray:
    """Looking at camera — normal interview, NOT reading."""
    rng = np.random.default_rng(seed)
    gaze = np.empty((WINDOW, 2), dtype=np.float32)
    x, y = 0.0, 0.0
    for f in range(WINDOW):
        x += rng.normal(0, 0.01) - 0.02 * x
        y += rng.normal(0, 0.005) - 0.02 * y
        if rng.random() < 0.03:
            gaze[f] = [np.nan, np.nan]
        else:
            gaze[f] = [np.clip(x, -0.3, 0.3), np.clip(y, -0.15, 0.15)]
    return gaze


def make_glancing(seed: int = 0) -> np.ndarray:
    """Occasional side glances — NOT reading, should not trigger."""
    rng = np.random.default_rng(seed)
    gaze = np.empty((WINDOW, 2), dtype=np.float32)
    x, y = 0.0, 0.0
    for f in range(WINDOW):
        x += rng.normal(0, 0.008) - 0.02 * x
        y += rng.normal(0, 0.004) - 0.02 * y
        if rng.random() < 0.04:
            x += rng.choice([-0.12, 0.12])
        gaze[f] = [np.clip(x, -0.3, 0.3), np.clip(y, -0.15, 0.15)]
    return gaze


def make_single_sweep(seed: int = 0) -> np.ndarray:
    """One slow sweep (looking at something) — NOT enough for reading."""
    rng = np.random.default_rng(seed)
    gaze = np.empty((WINDOW, 2), dtype=np.float32)
    for f in range(WINDOW):
        progress = f / WINDOW
        x = -0.15 + 0.30 * progress + rng.normal(0, 0.005)
        y = rng.normal(0, 0.003)
        gaze[f] = [x, y]
    return gaze


def main() -> None:
    test_cases = [
        # (name, generator, expected_reading)
        ("LTR normal speed", make_ltr_reading(1.0, 1), True),
        ("LTR fast", make_ltr_reading(1.5, 2), True),
        ("LTR slow", make_ltr_reading(0.7, 3), True),
        ("LTR seed=42", make_ltr_reading(1.0, 42), True),
        ("LTR seed=99", make_ltr_reading(1.0, 99), True),
        ("RTL normal (flipped)", make_rtl_reading(1.0, 4), True),
        ("RTL fast (flipped)", make_rtl_reading(1.5, 5), True),
        ("RTL slow (flipped)", make_rtl_reading(0.7, 50), True),
        ("RTL seed=77", make_rtl_reading(1.0, 77), True),
        ("Zigzag scanning", make_zigzag_reading(6), True),
        ("Zigzag seed=88", make_zigzag_reading(88), True),
        ("Camera look 1", make_camera_look(10), False),
        ("Camera look 2", make_camera_look(11), False),
        ("Camera look 3", make_camera_look(12), False),
        ("Camera look 4", make_camera_look(42), False),
        ("Camera look 5", make_camera_look(99), False),
        ("Camera look 6", make_camera_look(200), False),
        ("Glancing 1", make_glancing(20), False),
        ("Glancing 2", make_glancing(21), False),
        ("Glancing 3", make_glancing(42), False),
        ("Glancing 4", make_glancing(99), False),
        ("Single sweep", make_single_sweep(30), False),
        ("Single sweep 2", make_single_sweep(55), False),
    ]

    # Thresholds matching the demo's defaults
    possible_thresh = 0.14
    high_thresh = 0.59

    print(f"{'Test Case':<30} {'Score':>8} {'Detected':>10} {'Expected':>10} {'OK':>5}")
    print("=" * 70)

    tp, fp, tn, fn = 0, 0, 0, 0
    for name, gaze, expected in test_cases:
        score = score_window_rule(gaze)
        detected = score >= possible_thresh
        ok = detected == expected
        status = "PASS" if ok else "FAIL"

        if detected and expected:
            tp += 1
        elif detected and not expected:
            fp += 1
        elif not detected and expected:
            fn += 1
        else:
            tn += 1

        print(f"{name:<30} {score:>8.3f} {'READING' if detected else 'normal':>10} {'READING' if expected else 'normal':>10} {status:>5}")

    print("=" * 70)
    total = len(test_cases)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nResults: {tp + tn}/{total} correct ({accuracy:.0%})")
    print(f"  True positives:  {tp}  (reading detected correctly)")
    print(f"  True negatives:  {tn}  (normal detected correctly)")
    print(f"  False positives: {fp}  (normal flagged as reading)")
    print(f"  False negatives: {fn}  (reading missed)")
    print(f"\n  Precision: {precision:.2f}  Recall: {recall:.2f}  F1: {f1:.2f}")

    # --- Early detection tests (3-second windows) ---
    print(f"\n\n{'='*70}")
    print("EARLY DETECTION (3-second windows — short answers)")
    print(f"{'='*70}\n")

    def short_ltr(speed: float = 1.0, seed: int = 0) -> np.ndarray:
        return make_ltr_reading(speed, seed)[:SHORT]

    def short_rtl(speed: float = 1.0, seed: int = 0) -> np.ndarray:
        return make_rtl_reading(speed, seed)[:SHORT]

    def short_camera(seed: int = 0) -> np.ndarray:
        return make_camera_look(seed)[:SHORT]

    def short_glancing(seed: int = 0) -> np.ndarray:
        return make_glancing(seed)[:SHORT]

    early_cases = [
        ("Short LTR normal", short_ltr(1.0, 1), True),
        ("Short LTR fast", short_ltr(1.5, 2), True),
        ("Short LTR slow", short_ltr(0.7, 3), True),
        ("Short LTR seed=42", short_ltr(1.0, 42), True),
        ("Short RTL normal", short_rtl(1.0, 4), True),
        ("Short RTL fast", short_rtl(1.5, 5), True),
        ("Short RTL seed=77", short_rtl(1.0, 77), True),
        ("Short camera 1", short_camera(10), False),
        ("Short camera 2", short_camera(11), False),
        ("Short camera 3", short_camera(42), False),
        ("Short camera 4", short_camera(99), False),
        ("Short camera 5", short_camera(200), False),
        ("Short glancing 1", short_glancing(20), False),
        ("Short glancing 2", short_glancing(21), False),
        ("Short glancing 3", short_glancing(42), False),
    ]

    print(f"{'Test Case':<30} {'Score':>8} {'Detected':>10} {'Expected':>10} {'OK':>5}")
    print("-" * 70)

    tp2, fp2, tn2, fn2 = 0, 0, 0, 0
    for name, gaze, expected in early_cases:
        score = score_early_reading(gaze)
        detected = score >= possible_thresh
        ok = detected == expected
        status = "PASS" if ok else "FAIL"

        if detected and expected:
            tp2 += 1
        elif detected and not expected:
            fp2 += 1
        elif not detected and expected:
            fn2 += 1
        else:
            tn2 += 1

        print(f"{name:<30} {score:>8.3f} {'READING' if detected else 'normal':>10} {'READING' if expected else 'normal':>10} {status:>5}")

    print("-" * 70)
    total2 = len(early_cases)
    acc2 = (tp2 + tn2) / total2
    prec2 = tp2 / (tp2 + fp2) if (tp2 + fp2) > 0 else 0
    rec2 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0
    f1_2 = 2 * prec2 * rec2 / (prec2 + rec2) if (prec2 + rec2) > 0 else 0

    print(f"\nEarly results: {tp2 + tn2}/{total2} correct ({acc2:.0%})")
    print(f"  True positives:  {tp2}  (reading detected in 3s)")
    print(f"  True negatives:  {tn2}  (normal correctly ignored)")
    print(f"  False positives: {fp2}  (normal flagged as reading)")
    print(f"  False negatives: {fn2}  (reading missed)")
    print(f"\n  Precision: {prec2:.2f}  Recall: {rec2:.2f}  F1: {f1_2:.2f}")


if __name__ == "__main__":
    main()
