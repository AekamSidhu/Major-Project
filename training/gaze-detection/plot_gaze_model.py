"""Draw the gaze-model robustness figure from results/gaze_model_results.json.

    python plot_gaze_model.py   # -> results/gaze_model_robustness.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS = HERE / "results" / "gaze_model_results.json"
OUT = HERE / "results" / "gaze_model_robustness.png"

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CNN, MEDIAPIPE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2


def main() -> None:
    rows = json.loads(RESULTS.read_text(encoding="utf-8"))["robustness_vs_mediapipe"]
    cnn = [r for r in rows if r.get("cnn_err") is not None]
    mp = [r for r in rows if r.get("mp_err_uncal") is not None]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURFACE)
    panels = [
        ("Gaze error vs face size", "mean angular error (degrees)",
         [("Transfer-learned CNN (LOPO)", cnn, "cnn_err", CNN, "-"),
          ("MediaPipe iris offsets", mp, "mp_err_uncal", MEDIAPIPE, "-"),
          ("MediaPipe, calibrated per person", mp, "mp_err_cal", MEDIAPIPE, "--")]),
        ("Horizontal gaze signal (what the reading detector uses)", "correlation with true yaw, within person",
         [("Transfer-learned CNN (LOPO)", cnn, "cnn_r_yaw", CNN, "-"),
          ("MediaPipe iris offsets", mp, "mp_r_yaw", MEDIAPIPE, "-")]),
    ]
    for ax, (title, ylabel, series) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for label, data, key, color, style in series:
            xs = [r["face_px"] for r in data]
            ys = [r[key] for r in data]
            ax.plot(xs, ys, style, color=color, lw=2, marker="o", ms=5, label=label)
            for i in (0, -1):  # label the two ends only
                ax.annotate(f"{ys[i]:.2f}", (xs[i], ys[i]), xytext=(0, 7), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, color=INK_2)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks([32, 48, 64, 112, 224])
        ax.set_xticklabels(["32", "48", "64", "112", "224"])
        ax.set_xlim(260, 28)  # a little room for the end labels; left = full size
        ax.set_xlabel("face width in pixels")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(True, color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="center left", bbox_to_anchor=(0.02, 0.6))
    axes[1].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower left")
    axes[1].set_ylim(0.5, 1.02)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, facecolor=SURFACE)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
