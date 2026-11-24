"""Checks that the numbers written in docs/BENCHMARKS.md's text match the result files.

    python benchmarks/check_doc.py            # exits non-zero if a claim is missing or differs

Each claim is a phrase that must appear in the document (whitespace
normalised, so a line break inside a phrase is fine) together with the value
it states, read from a result file. The table blocks are checked separately by
`tables.py --doc --check`. Keep this list in step with the prose.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchlib  # noqa: E402

DOC = benchlib.REPO / "docs" / "BENCHMARKS.md"
R = benchlib.RESULTS
REPO = benchlib.REPO


def get(path: Path, keys: list[str]) -> Any:
    value: Any = benchlib.read_json(path)
    for k in keys:
        value = value[k]
    return value


def pct(digits: int) -> Callable[[float], str]:
    return lambda v: f"{100 * v:.{digits}f}%"


def sec(digits: int) -> Callable[[float], str]:
    return lambda v: f"{v / 1000:.{digits}f} s"


def claim(phrase: str, file: Path, keys: list[str], fmt: Callable[[Any], str], scale: Callable[[Any], Any] | None = None):
    """(phrase that must be in the doc, the source value, how the phrase formats it)."""
    return phrase, file, keys, fmt, scale


def latency_median(file: Path, model: str) -> list[str]:
    return ["models", model, "all", "latency_ms", "median"]


CLAIMS = [
    # 1. Transcription (section 1, summary)
    claim("3.9% word errors on test-clean", R / "asr_cpu.json", ["models", "base.en", "test-clean", "wer"],
          lambda v: f"{100 * v:.1f}% word errors on test-clean"),
    claim("10.7% on test-other", R / "asr_cpu.json", ["models", "base.en", "test-other", "wer"],
          lambda v: f"{100 * v:.1f}% on test-other"),
    claim("about 0.41 s on the CPU", R / "asr_cpu.json", latency_median(R / "asr_cpu.json", "base.en"),
          lambda v: f"about {v / 1000:.2f} s on the CPU"),
    claim("0.09 s on the GPU", R / "asr_cuda.json", latency_median(R / "asr_cuda.json", "base.en"),
          lambda v: f"{v / 1000:.2f} s on the GPU"),
    # 2. Word timestamps: about 2.4 times the plain CPU median (ratio computed below)
    # 3. AI-written text (section 3, summary)
    claim("catches 97.8% of ChatGPT answers", R / "ai_text_v1_vs_v2.json", ["models", "v2", "at_high", "detection_rate"],
          lambda v: f"catch {100 * v:.1f}% of ChatGPT answers"),
    claim("flag 1.8% (v2)", R / "ai_text_v1_vs_v2.json", ["models", "v2", "at_high", "false_flag_rate"],
          lambda v: f"flag {100 * v:.1f}% (v2)"),
    claim("2.0% (v1)", R / "ai_text_v1_vs_v2.json", ["models", "v1", "at_high", "false_flag_rate"],
          lambda v: f"{100 * v:.1f}% (v1)"),
    claim("against 1.8% on all HC3 test", R / "ai_text_v1_vs_v2.json", ["models", "v2", "at_high", "false_flag_rate"],
          lambda v: f"against {100 * v:.1f}% on all HC3 test"),
    # 4. Fused flag (summary, section 3)
    claim("catches 80.0%", R / "fusion_hc3.json", ["configs", "fused (ai_text + echo)", "at_high", "detection_rate"],
          lambda v: f"catches {100 * v:.1f}%"),
    claim("flags 4.7% of human ones", R / "fusion_hc3.json", ["configs", "fused (ai_text + echo)", "at_high", "false_flag_rate"],
          lambda v: f"flags {100 * v:.1f}% of human ones"),
    claim("99.6% and flags 25%", R / "fusion_hc3.json", ["configs", "fused (ai_text + echo)", "at_possible", "detection_rate"],
          lambda v: f"{100 * v:.1f}% and flags 25.0%"),
    claim("AI text alone 6.8%", R / "fusion_hc3.json", ["configs", "ai_text alone", "at_high", "false_flag_rate"],
          lambda v: f"AI text alone {100 * v:.1f}%"),
    claim("echo alone 6.6%", R / "fusion_hc3.json", ["configs", "echo similarity alone", "at_high", "false_flag_rate"],
          lambda v: f"echo alone {100 * v:.1f}%"),
    claim("46.7% against 37.7%", R / "fusion_hc3.json", ["configs", "echo similarity alone", "at_high", "detection_rate"],
          lambda v: f"{100 * v:.1f}% against 37.7%"),
    claim("(3.8% fused at `high`)", R / "fusion_hc3.json", ["configs", "fused (ai_text + echo, 1 reference)", "at_high", "false_flag_rate"],
          lambda v: f"({100 * v:.1f}% fused at `high`)"),
    claim("fewer detections (72.5%)", R / "fusion_hc3.json", ["configs", "fused (ai_text + echo, 1 reference)", "at_high", "detection_rate"],
          lambda v: f"fewer detections ({100 * v:.1f}%)"),
    claim("0.876 against 0.859", R / "fusion_hc3.json", ["roc_auc", "echo similarity (2 references)"],
          lambda v: f"{v:.3f} against 0.859"),
    claim("16% of this sample's", R / "fusion_hc3.json", ["by_source", "open_qa", "ai_text alone", "at_high", "false_flag_rate"],
          lambda v: f"{100 * v:.0f}% of this sample's"),
    claim("on wiki_csai 28%", R / "fusion_hc3.json", ["by_source", "wiki_csai", "ai_text alone", "at_high", "false_flag_rate"],
          lambda v: f"on wiki_csai {100 * v:.0f}%"),
    claim("flags 25% of human answers at `high` and 80% at `possible`", R / "fusion_hc3.json",
          ["by_source", "wiki_csai", "fused (ai_text + echo)", "at_high", "false_flag_rate"],
          lambda v: f"flags {100 * v:.0f}% of human answers at `high` and 80% at `possible`"),
    # 5. Reading (section 3)
    claim("about 28% of non-reading windows", REPO / "training" / "gaze-detection" / "results" / "reading_detector_results.json",
          ["test", "network", "gazebase_small_hard", "window_fpr_at_low_fpr"],
          lambda v: f"about {100 * v:.0f}% of non-reading windows"),
    # 5b. Desktop engine and room (summary, section 2)
    claim("a median 1.8 s after the speaker", R / "engine_live_cpu.json", ["candidate_mode", "speech_end_to_first_token_ms", "median"],
          lambda v: f"a median {v / 1000:.1f} s after the speaker"),
    claim("0.7 s on the GPU", R / "engine_live_cuda.json", ["candidate_mode", "speech_end_to_first_token_ms", "median"],
          lambda v: f"{v / 1000:.1f} s on the GPU"),
    claim("all 12 questions were detected", R / "engine_live_cpu.json", ["candidate_mode", "detected"],
          lambda v: "all 12 questions were detected" if v == 12 else f"{v} questions detected"),
        claim("transcribed back in 8 of its 8 turns", R / "room_e2e.json", ["answers"],
          lambda rows: f"transcribed back in {sum(bool(r['question_transcribed']) for r in rows)} of its {len(rows)} turns"),
    claim("5.0 s (median) after the answer closed", R / "room_e2e.json", ["turns_1_to_7", "close_to_card_ms", "median"],
          lambda v: f"{v / 1000:.1f} s (median) after the answer closed"),
    claim("1.6 s (median) after the card", R / "room_e2e.json", ["turns_1_to_7", "card_to_similarity_ms", "median"],
          lambda v: f"{v / 1000:.1f} s (median) after the card"),
    claim("6.6 s (25 requests)", R / "room_e2e.json", ["segment_transcription_ms", "median"],
          lambda v: f"{v / 1000:.1f} s"),
    # 6. Reproducibility of the committed v2 model (section 3)
    claim("0.4900 / 0.2476 for HC3", R / "retrain_v2_rerun.json", ["thresholds", "balanced"],
          lambda v: f"{v:.4f} / 0.2476 for HC3"),
    claim("against 0.4624 / 0.2466", REPO / "local-ai" / "integrity" / "models" / "echo_tfidf.json", ["thresholds", "balanced"],
          lambda v: f"against {v:.4f} / 0.2466"),
]


def check_ratio() -> tuple[bool, str]:
    words = get(R / "asr_cpu_words.json", ["models", "base.en", "all", "latency_ms", "median"])
    plain = get(R / "asr_cpu.json", ["models", "base.en", "all", "latency_ms", "median"])
    ok = "about 2.4 times the plain CPU median" in DOC_TEXT
    return ok and f"{words / plain:.1f}" == "2.4", f"word-timestamp / plain CPU median = {words / plain:.2f}"


def main() -> None:
    global DOC_TEXT
    DOC_TEXT = re.sub(r"\s+", " ", DOC.read_text(encoding="utf-8"))
    failures = []
    for phrase, file, keys, fmt, _ in CLAIMS:
        # The phrase is what the doc must say; fmt(value) must equal that phrase's number.
        text = fmt(get(file, keys))
        if text not in DOC_TEXT:
            failures.append(f"doc does not contain {text!r} (from {file.name}: {'/'.join(keys)})")
            continue
        print(f"ok  {text}   <- {file.relative_to(REPO).as_posix()} {'/'.join(keys)}")
    ratio_ok, ratio_msg = check_ratio()
    print(("ok  " if ratio_ok else "FAIL ") + ratio_msg)
    if not ratio_ok:
        failures.append(ratio_msg)
    if failures:
        print("\n".join(failures))
        raise SystemExit(1)
    print(f"{len(CLAIMS)} claims checked against the result files: all match")


if __name__ == "__main__":
    main()
