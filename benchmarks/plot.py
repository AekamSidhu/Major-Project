"""Charts for docs/BENCHMARKS.md from the result files (PNG, light surface).

    python benchmarks/plot.py

Reads benchmarks/results/*.json and writes benchmarks/results/*.png; a chart
whose inputs are missing is skipped. Colours: the first three slots of the
dataviz reference palette (validated colour-blind safe as a set); every mark
carries a direct label because the third slot is below 3:1 contrast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlib  # noqa: E402

R = benchlib.RESULTS
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False,
})


def load(name: str):
    path = R / name
    return benchlib.read_json(path) if path.exists() else None


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(R / name, dpi=150)
    plt.close(fig)
    print(f"wrote {R / name}")


def asr_chart() -> None:
    runs = [(d, load(f"asr_{d}.json")) for d in ("cpu", "cuda")]
    runs = [(d, r) for d, r in runs if r]
    if not runs:
        return
    colors = {"tiny.en": S1, "base.en": S2, "small.en": S3}
    markers = {"cpu": "o", "cuda": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    for ax, split in zip(axes, ("test-clean", "test-other")):
        for device, res in runs:
            for model, m in res["models"].items():
                if split not in m:
                    continue
                x, y = m[split]["latency_ms"]["median"], 100 * m[split]["wer"]
                ax.scatter([x], [y], s=70, color=colors.get(model, INK2), marker=markers[device],
                           edgecolors=SURFACE, linewidths=2, zorder=3)
                ax.annotate(f"{model} {device.upper() if device == 'cpu' else 'GPU'}\n{y:.1f}%, {x:.0f} ms", (x, y),
                            textcoords="offset points", xytext=(8, 4), fontsize=8, color=INK2)
        ax.set_xscale("log")
        ax.set_title(f"LibriSpeech {split}")
        ax.set_xlabel("median latency per utterance, ms (log scale)")
        ax.set_ylabel("word error rate, %")
        ax.set_ylim(bottom=0)
        ax.margins(x=0.35, y=0.25)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=m, markersize=8) for m, c in colors.items()]
    handles += [plt.Line2D([], [], marker=markers[d], ls="", color=INK2, label="CPU" if d == "cpu" else "GPU (RTX 4060)",
                           markersize=8) for d, _ in runs]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Whisper model size: accuracy against latency (sidecar, 300 utterances per split)", fontsize=11)
    fig.subplots_adjust(bottom=0.2)
    save(fig, "asr_wer_latency.png")


def bars(ax, labels, values, color, fmt="{:.0%}"):
    """Horizontal bars, first label on top (lists reversed, so a shared y-axis is never inverted twice)."""
    labels, values = labels[::-1], values[::-1]
    ys = range(len(labels))
    ax.barh(list(ys), values, height=0.6, color=color, edgecolor=SURFACE, linewidth=2)
    for y, v in zip(ys, values):
        ax.text(v, y, " " + fmt.format(v), va="center", fontsize=8, color=INK)
    ax.set_yticks(list(ys), labels)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def fusion_chart() -> None:
    res = load("fusion_hc3.json")
    if not res:
        return
    configs = ["ai_text alone", "echo similarity alone", "echo similarity alone, 1 reference", "fused (ai_text + echo)",
               "fused (ai_text + echo, 1 reference)"]
    labels = ["AI-written text alone", "Echo similarity alone (2 refs)", "Echo similarity alone (1 ref)",
              "Fused flag (2 refs)", "Fused flag (1 ref)"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharey=True)
    for col, cut in enumerate(("possible", "high")):
        c = res["configs"]
        bars(axes[0][col], labels, [c[k][f"at_{cut}"]["detection_rate"] for k in configs], S1)
        bars(axes[1][col], labels, [c[k][f"at_{cut}"]["false_flag_rate"] for k in configs], S2)
        axes[0][col].set_title(f"Detection: ChatGPT answers at '{cut}'")
        axes[1][col].set_title(f"False flags: human answers at '{cut}'")
        axes[0][col].set_xlim(0, 1.15)
        axes[1][col].set_xlim(0, 0.35)
    fig.suptitle(f"HC3 test questions ({res['protocol']['questions']} questions, {res['answers']} answers): "
                 "detection rate and false flags", fontsize=11)
    save(fig, "fusion_hc3.png")

    sources = list(res["by_source"])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    width = 0.26
    for i, (key, label, color) in enumerate((("ai_text alone", "AI-written text alone", S1),
                                            ("echo similarity alone", "Echo similarity alone", S2),
                                            ("fused (ai_text + echo)", "Fused flag", S3))):
        vals = [res["by_source"][s][key]["at_high"]["false_flag_rate"] for s in sources]
        xs = [j + (i - 1) * width for j in range(len(sources))]
        ax.bar(xs, vals, width=width * 0.9, color=color, label=label, edgecolor=SURFACE, linewidth=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.005, f"{v:.0%}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(sources)), sources)
    ax.set_ylabel("human answers flagged at 'high'")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(loc="upper left")
    ax.set_title("False flags at 'high' by HC3 source (100 test questions each)")
    save(fig, "fusion_by_source.png")


def latency_chart() -> None:
    rows: list[tuple[str, dict]] = []
    for device in ("cpu", "cuda"):
        live = load(f"engine_live_{device}.json")
        if live:
            tag = "CPU" if device == "cpu" else "GPU"
            cm, im = live["candidate_mode"], live["interviewer_mode"]
            rows += [(f"Desktop ({tag} Whisper): speech end -> transcript", cm["speech_end_to_transcript_ms"]),
                     (f"Desktop ({tag} Whisper): speech end -> first answer token", cm["speech_end_to_first_token_ms"]),
                     (f"Desktop ({tag} Whisper): speech end -> answer complete", cm["speech_end_to_complete_ms"]),
                     (f"Desktop ({tag} Whisper): speech end -> follow-ups done", im["speech_end_to_followups_done_ms"])]
    room = load("room_e2e.json")
    if room:
        t = room["turns_1_to_7"]
        rows += [("Room: answer closed -> card", t["close_to_card_ms"]),
                 ("Room: card -> echo similarity", t["card_to_similarity_ms"]),
                 ("Room: card -> follow-ups", t["card_to_followups_ms"])]
    rows = [(k, v) for k, v in rows if v and v.get("n")]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 1.4))
    ys = list(range(len(rows)))
    med = [v["median"] / 1000 for _, v in rows]
    p90 = [v["p90"] / 1000 for _, v in rows]
    ax.barh(ys, med, height=0.55, color=S1, edgecolor=SURFACE, linewidth=2, label="median")
    ax.scatter(p90, ys, marker="|", s=180, color=INK, linewidths=2, zorder=3, label="90th percentile")
    for y, m, p in zip(ys, med, p90):
        ax.text(max(m, p), y, f"  {m:.1f} s (p90 {p:.1f} s)", va="center", fontsize=8, color=INK)
    ax.set_yticks(ys, [k for k, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("seconds")
    ax.set_xlim(0, max(p90) * 1.35)
    ax.legend(loc="lower right")
    ax.set_title("Where the time goes: live pipeline stages (llama3.2 on the GPU through Ollama)")
    save(fig, "pipeline_latency.png")


def main() -> None:
    asr_chart()
    fusion_chart()
    latency_chart()


if __name__ == "__main__":
    main()
